"""Network topology, costs and service (speed) structure.

The physical flow is   FC --(line-haul truck)--> SC --(last mile)--> demand zone.
Every zone is served by exactly one sort centre (its nearest one, by road), so the
fulfilment decision "which FC ships this unit" also fixes the middle-mile lane
(FC, SC(zone)). A unit is delivered *on time* (next day) when the line-haul transit of
its lane fits in the overnight window between the FC cut-off and the SC sort start.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import geo


@dataclass(frozen=True)
class CostParams:
    truck_capacity: float = 1400.0      # package-equivalents per 53' trailer
    truck_fixed: float = 180.0          # EUR per scheduled truck (tender, loading)
    truck_per_km: float = 1.25          # EUR per km (driver, fuel, tolls)
    spot_premium: float = 1.8           # spot / LTL capacity cost relative to scheduled, per pkg-eq
    truck_speed_kmh: float = 80.0
    truck_handling_h: float = 0.5       # load + unload
    linehaul_window_h: float = 6.5      # FC cut-off -> SC sort start
    crew_capacity: float = 2000.0       # labour units per crew-shift (10 associates x 8h)
    crew_cost: float = 1760.0           # EUR per crew-shift
    overtime_frac: float = 0.30         # max overtime as fraction of planned capacity
    overtime_premium: float = 1.6       # overtime cost multiplier
    lastmile_fixed: float = 2.60        # EUR per unit
    lastmile_per_km: float = 0.012      # EUR per unit-km from SC to zone centroid
    unmet_penalty: float = 15.0         # EUR per unit not shipped today (missed promise, re-handling, cancellations)
    late_penalty: float = 1.0           # EUR per unit delivered after promise (price of speed)
    co2_truck_per_km: float = 0.90      # kg CO2 per truck-km
    co2_spot_factor: float = 1.35       # spot capacity is less consolidated
    co2_lastmile_per_unit: float = 0.18 # kg CO2 per unit (van)


@dataclass
class Network:
    """Dense array description of a fulfilment network (all arrays numpy)."""

    fc_names: list
    sc_names: list
    zone_names: list
    cat_names: list
    fc_xy: np.ndarray            # (I,2) lat/lon
    sc_xy: np.ndarray            # (H,2)
    zone_xy: np.ndarray          # (J,2)
    zone_pop: np.ndarray         # (J,) thousands
    zone_country: np.ndarray     # (J,) str
    zone_region: np.ndarray      # (J,) str
    zone_sc: np.ndarray          # (J,) int, serving SC
    lane_km: np.ndarray          # (I,H)
    lane_hours: np.ndarray       # (I,H)
    lane_late: np.ndarray        # (I,H) bool, transit exceeds the overnight window
    sc_zone_km: np.ndarray       # (J,)
    crew_max: np.ndarray         # (I,) int
    dock_max: np.ndarray         # (I,) int trucks/day
    fc_var_cost: np.ndarray      # (I,) EUR/unit
    stocks: np.ndarray           # (I,K) bool
    cat_volume: np.ndarray       # (K,)
    cat_labour: np.ndarray       # (K,)
    cat_share: np.ndarray        # (K,)
    cat_dispersion: np.ndarray   # (K,)
    params: CostParams = field(default_factory=CostParams)

    # ---- sizes
    @property
    def I(self):
        return len(self.fc_names)

    @property
    def H(self):
        return len(self.sc_names)

    @property
    def J(self):
        return len(self.zone_names)

    @property
    def K(self):
        return len(self.cat_names)

    # ---- derived costs
    @property
    def truck_cost(self) -> np.ndarray:
        p = self.params
        return p.truck_fixed + p.truck_per_km * self.lane_km

    @property
    def spot_cost(self) -> np.ndarray:
        """EUR per package-equivalent of spot capacity on each lane."""
        p = self.params
        return p.spot_premium * self.truck_cost / p.truck_capacity

    @property
    def overtime_cost(self) -> float:
        """EUR per labour unit of overtime."""
        p = self.params
        return p.overtime_premium * p.crew_cost / p.crew_capacity

    @property
    def lastmile_cost(self) -> np.ndarray:
        p = self.params
        return p.lastmile_fixed + p.lastmile_per_km * self.sc_zone_km

    @property
    def late_ij(self) -> np.ndarray:
        """(I,J) bool: shipping FC i -> zone j misses the next-day promise."""
        return self.lane_late[:, self.zone_sc]

    def with_params(self, **kw) -> "Network":
        return replace(self, params=replace(self.params, **kw))

    def add_fcs(self, rows) -> "Network":
        """Return a copy of the network with extra FCs (rows formatted like geo.FCS)."""
        fc = [(r[0], r[2], r[3], r[4], r[5], r[6], r[7]) for r in rows]
        xy = np.vstack([self.fc_xy, np.array([[r[1], r[2]] for r in fc])])
        stocks = np.vstack([self.stocks, np.array([[c in r[6] for c in self.cat_names] for r in fc])])
        km = geo.road_km(xy[:, None, 0], xy[:, None, 1], self.sc_xy[None, :, 0], self.sc_xy[None, :, 1])
        hours = km / self.params.truck_speed_kmh + self.params.truck_handling_h
        return replace(
            self,
            fc_names=self.fc_names + [r[0] for r in fc],
            fc_xy=xy,
            lane_km=km,
            lane_hours=hours,
            lane_late=hours > self.params.linehaul_window_h,
            crew_max=np.concatenate([self.crew_max, [r[3] for r in fc]]).astype(int),
            dock_max=np.concatenate([self.dock_max, [r[4] for r in fc]]).astype(int),
            fc_var_cost=np.concatenate([self.fc_var_cost, [r[5] for r in fc]]),
            stocks=stocks,
        )


def _assemble(fc_rows, sc_rows, zone_rows, cat_rows, params: CostParams) -> Network:
    fc_xy = np.array([[r[2], r[3]] for r in fc_rows])
    sc_xy = np.array([[r[2], r[3]] for r in sc_rows])
    zone_xy = np.array([[r[2], r[3]] for r in zone_rows])
    cat_codes = [c[0] for c in cat_rows]

    sz = geo.road_km(sc_xy[None, :, 0], sc_xy[None, :, 1], zone_xy[:, None, 0], zone_xy[:, None, 1])  # (J,H)
    # last mile never crosses a border: a zone is served by the nearest SC of its own country
    sc_country = np.array([r[4] if len(r) > 4 else "ES" for r in sc_rows])
    zone_country = np.array([r[1] for r in zone_rows])
    zone_sc = np.where(sc_country[None, :] == zone_country[:, None], sz, np.inf).argmin(axis=1)
    lane_km = geo.road_km(fc_xy[:, None, 0], fc_xy[:, None, 1], sc_xy[None, :, 0], sc_xy[None, :, 1])
    lane_hours = lane_km / params.truck_speed_kmh + params.truck_handling_h

    return Network(
        fc_names=[r[0] for r in fc_rows],
        sc_names=[r[0] for r in sc_rows],
        zone_names=[r[0] for r in zone_rows],
        cat_names=cat_codes,
        fc_xy=fc_xy,
        sc_xy=sc_xy,
        zone_xy=zone_xy,
        zone_pop=np.array([r[4] for r in zone_rows], dtype=float),
        zone_country=zone_country,
        zone_region=np.array([r[5] for r in zone_rows]),
        zone_sc=zone_sc,
        lane_km=lane_km,
        lane_hours=lane_hours,
        lane_late=lane_hours > params.linehaul_window_h,
        sc_zone_km=sz[np.arange(len(zone_rows)), zone_sc],
        crew_max=np.array([r[4] for r in fc_rows], dtype=int),
        dock_max=np.array([r[5] for r in fc_rows], dtype=int),
        fc_var_cost=np.array([r[6] for r in fc_rows], dtype=float),
        stocks=np.array([[c in r[7] for c in cat_codes] for r in fc_rows]),
        cat_volume=np.array([c[2] for c in cat_rows]),
        cat_labour=np.array([c[3] for c in cat_rows]),
        cat_share=np.array([c[4] for c in cat_rows]),
        cat_dispersion=np.array([c[5] for c in cat_rows]),
        params=params,
    )


def iberia(params: CostParams | None = None) -> Network:
    """The base case: 10 FCs, 12 SCs, 65 zones, 4 categories."""
    return _assemble(geo.FCS, geo.SCS, geo.ZONES, geo.CATEGORIES, params or CostParams())


def synthetic(n_fc: int, n_sc: int, n_zone: int, n_cat: int = 4, seed: int = 0,
              params: CostParams | None = None) -> Network:
    """Random geometric network over a 1200 x 1000 km box (for scalability studies).

    Capacities are scaled so that the network is roughly as loaded as the Iberian case.
    """
    rng = np.random.default_rng(seed)
    params = params or CostParams()
    # positions in degrees over a box comparable to Iberia
    lat0, lon0 = 36.5, -9.0
    zone_xy = np.c_[lat0 + 7 * rng.random(n_zone), lon0 + 12 * rng.random(n_zone)]
    pop = np.exp(rng.normal(6.0, 1.0, n_zone))
    pop *= 56000 / pop.sum()
    # FCs/SCs located near population-weighted random zones
    fc_idx = rng.choice(n_zone, n_fc, replace=n_fc > n_zone, p=pop / pop.sum())
    sc_idx = rng.choice(n_zone, n_sc, replace=n_sc > n_zone, p=pop / pop.sum())
    fc_xy = zone_xy[fc_idx] + rng.normal(0, 0.15, (n_fc, 2))
    sc_xy = zone_xy[sc_idx] + rng.normal(0, 0.10, (n_sc, 2))
    base_cats = list(geo.CATEGORIES)
    cats = []
    for k in range(n_cat):
        c = base_cats[k % len(base_cats)]
        cats.append((f"{c[0]}{k}", c[1], c[2], c[3], c[4], c[5]))
    share = np.array([c[4] for c in cats])
    share /= share.sum()
    cats = [(c[0], c[1], c[2], c[3], s, c[5]) for c, s in zip(cats, share)]
    # capacity proportional to a random weight, total ~ the Iberian total
    wts = rng.dirichlet(np.ones(n_fc) * 2.0)
    crews = np.maximum(4, np.round(wts * 426)).astype(int)
    docks = np.maximum(10, np.round(wts * 1020)).astype(int)
    fc_rows = []
    for i in range(n_fc):
        stock = "".join(c[0] for c in cats if rng.random() < 0.8) or cats[0][0]
        fc_rows.append((f"FC{i:03d}", "", fc_xy[i, 0], fc_xy[i, 1], crews[i], docks[i],
                        0.22 + 0.1 * rng.random(), None))
    net = _assemble(
        [(r[0], r[1], r[2], r[3], r[4], r[5], r[6], "") for r in fc_rows],
        [(f"SC{h:03d}", "", sc_xy[h, 0], sc_xy[h, 1]) for h in range(n_sc)],
        [(f"Z{j:04d}", "ES", zone_xy[j, 0], zone_xy[j, 1], pop[j], f"R{j % 12}") for j in range(n_zone)],
        cats, params)
    stocks = rng.random((n_fc, n_cat)) < 0.75
    stocks[np.arange(n_fc), rng.integers(0, n_cat, n_fc)] = True
    stocks[rng.integers(0, n_fc, n_cat), np.arange(n_cat)] = True
    return replace(net, stocks=stocks)
