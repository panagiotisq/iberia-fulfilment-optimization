"""Basemap of the Iberian Peninsula for report maps (no GIS dependencies).

Boundaries: Natural Earth 1:10m admin-0 countries and admin-1 provinces (public domain,
naturalearthdata.com), clipped to Iberia and simplified into data/iberia_geo.json.
Coordinates are plotted in an equirectangular projection with cos(40 deg) aspect, which is
accurate to a few percent over the peninsula.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.path import Path

from .config import ROOT

SEA = "#eef3f8"
LAND_FOCUS = "#fbfaf7"
LAND_OTHER = "#ecebe6"
BORDER = "#8c8a82"
PROVINCE = "#cfcdc6"
EXTENT = (-12.2, 4.2, 35.6, 44.2)   # lon_min, lon_max, lat_min, lat_max (Atlantic margin holds legends)


@lru_cache(maxsize=None)
def _geo():
    with open(ROOT / "data" / "iberia_geo.json") as f:
        return json.load(f)


def _rings(polys):
    return [np.asarray(poly[0]) for poly in polys]


def setup_axes(ax, extent=EXTENT):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1 / np.cos(np.radians(40)))
    ax.set_facecolor(SEA)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#c9c7c0"); s.set_linewidth(0.5)


def basemap(ax, provinces: bool = True, extent=EXTENT, focus_fill=LAND_FOCUS):
    """Draw sea, land, national borders and (optionally) province outlines."""
    setup_axes(ax, extent)
    g = _geo()
    for c in g["countries"]:
        focus = c["iso"] in ("ESP", "PRT")
        ax.add_collection(PolyCollection(_rings(c["polys"]), facecolor=focus_fill if focus else LAND_OTHER,
                                         edgecolor="none", zorder=0.5))
    if provinces:
        segs = [r for p in g["provinces"] for r in _rings(p["polys"])]
        ax.add_collection(PolyCollection(segs, facecolor="none", edgecolor=PROVINCE, lw=0.35, zorder=0.8))
    for c in g["countries"]:
        ax.add_collection(PolyCollection(_rings(c["polys"]), facecolor="none", edgecolor=BORDER, lw=0.6, zorder=0.9))


@lru_cache(maxsize=None)
def zone_polygons(zone_xy_key: tuple) -> list:
    """For each zone (lat, lon) the list of rings of the province that contains it."""
    g = _geo()
    provs = [(p["name"], _rings(p["polys"])) for p in g["provinces"]]
    out = []
    for lat, lon in zone_xy_key:
        found = None
        for name, rings in provs:
            if any(Path(r).contains_point((lon, lat)) for r in rings):
                found = rings
                break
        if found is None:      # capital on the coast just outside a simplified polygon: nearest centroid
            d = [min(np.hypot(r[:, 0].mean() - lon, r[:, 1].mean() - lat) for r in rings) for _, rings in provs]
            found = provs[int(np.argmin(d))][1]
        out.append(found)
    return out


def choropleth(ax, net, values, cmap, norm, edgecolor="white", lw=0.4):
    """Fill every demand zone's province with cmap(norm(value))."""
    polys = zone_polygons(tuple(map(tuple, net.zone_xy)))
    rings, colors = [], []
    for j, rs in enumerate(polys):
        for r in rs:
            rings.append(r)
            colors.append(cmap(norm(values[j])))
    ax.add_collection(PolyCollection(rings, facecolors=colors, edgecolor=edgecolor, lw=lw, zorder=0.85))
