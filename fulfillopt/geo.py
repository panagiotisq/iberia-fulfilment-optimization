"""Static geography of the stylised mainland-Iberia fulfilment network.

Demand zones are the 47 mainland Spanish provinces and the 18 mainland Portuguese
districts (islands, Ceuta and Melilla are excluded because they are served by air/sea
logistics). Coordinates are those of the provincial/district capital; populations are
rounded public census figures (INE 2023 for Spain, INE-PT Census 2021 for Portugal),
in thousands.

Fulfilment centres (FC) and sort centres (SC) are placed in well-known Iberian logistics
hubs. They are *stylised*: no site, capacity or cost in this project is real Amazon data.
"""
from __future__ import annotations

import numpy as np

# name, country, lat, lon, population (thousands), autonomous region / PT region
ZONES = [
    ("A Coruna", "ES", 43.3623, -8.4115, 1128, "GAL"),
    ("Lugo", "ES", 43.0097, -7.5560, 326, "GAL"),
    ("Ourense", "ES", 42.3358, -7.8639, 305, "GAL"),
    ("Pontevedra", "ES", 42.4310, -8.6444, 945, "GAL"),
    ("Asturias", "ES", 43.3614, -5.8494, 1011, "AST"),
    ("Cantabria", "ES", 43.4623, -3.8100, 588, "CAN"),
    ("Bizkaia", "ES", 43.2630, -2.9350, 1154, "PVA"),
    ("Gipuzkoa", "ES", 43.3183, -1.9812, 727, "PVA"),
    ("Araba", "ES", 42.8467, -2.6716, 334, "PVA"),
    ("Navarra", "ES", 42.8125, -1.6458, 672, "NAV"),
    ("La Rioja", "ES", 42.4627, -2.4450, 320, "RIO"),
    ("Huesca", "ES", 42.1401, -0.4089, 224, "ARA"),
    ("Zaragoza", "ES", 41.6488, -0.8891, 980, "ARA"),
    ("Teruel", "ES", 40.3456, -1.1065, 134, "ARA"),
    ("Barcelona", "ES", 41.3874, 2.1686, 5714, "CAT"),
    ("Girona", "ES", 41.9794, 2.8214, 786, "CAT"),
    ("Lleida", "ES", 41.6176, 0.6200, 439, "CAT"),
    ("Tarragona", "ES", 41.1189, 1.2445, 822, "CAT"),
    ("Castellon", "ES", 39.9864, -0.0513, 587, "VAL"),
    ("Valencia", "ES", 39.4699, -0.3763, 2589, "VAL"),
    ("Alicante", "ES", 38.3452, -0.4810, 1881, "VAL"),
    ("Murcia", "ES", 37.9922, -1.1307, 1531, "MUR"),
    ("Almeria", "ES", 36.8340, -2.4637, 731, "AND"),
    ("Granada", "ES", 37.1773, -3.5986, 921, "AND"),
    ("Jaen", "ES", 37.7796, -3.7849, 627, "AND"),
    ("Cordoba", "ES", 37.8882, -4.7794, 776, "AND"),
    ("Sevilla", "ES", 37.3891, -5.9845, 1947, "AND"),
    ("Huelva", "ES", 37.2614, -6.9447, 525, "AND"),
    ("Cadiz", "ES", 36.5271, -6.2886, 1245, "AND"),
    ("Malaga", "ES", 36.7213, -4.4214, 1717, "AND"),
    ("Badajoz", "ES", 38.8794, -6.9707, 669, "EXT"),
    ("Caceres", "ES", 39.4753, -6.3724, 389, "EXT"),
    ("Madrid", "ES", 40.4168, -3.7038, 6871, "MAD"),
    ("Toledo", "ES", 39.8628, -4.0273, 709, "CLM"),
    ("Ciudad Real", "ES", 38.9848, -3.9274, 492, "CLM"),
    ("Cuenca", "ES", 40.0704, -2.1374, 196, "CLM"),
    ("Guadalajara", "ES", 40.6326, -3.1600, 270, "CLM"),
    ("Albacete", "ES", 38.9943, -1.8585, 386, "CLM"),
    ("Avila", "ES", 40.6566, -4.6818, 158, "CYL"),
    ("Segovia", "ES", 40.9429, -4.1088, 154, "CYL"),
    ("Soria", "ES", 41.7636, -2.4649, 89, "CYL"),
    ("Burgos", "ES", 42.3439, -3.6969, 357, "CYL"),
    ("Palencia", "ES", 42.0095, -4.5288, 159, "CYL"),
    ("Valladolid", "ES", 41.6523, -4.7245, 519, "CYL"),
    ("Leon", "ES", 42.5987, -5.5671, 451, "CYL"),
    ("Zamora", "ES", 41.5034, -5.7467, 167, "CYL"),
    ("Salamanca", "ES", 40.9701, -5.6635, 327, "CYL"),
    ("Lisboa", "PT", 38.7223, -9.1393, 2275, "PT-LVT"),
    ("Porto", "PT", 41.1579, -8.6291, 1786, "PT-N"),
    ("Setubal", "PT", 38.5244, -8.8882, 878, "PT-LVT"),
    ("Braga", "PT", 41.5454, -8.4265, 849, "PT-N"),
    ("Aveiro", "PT", 40.6405, -8.6538, 700, "PT-C"),
    ("Leiria", "PT", 39.7436, -8.8071, 459, "PT-C"),
    ("Coimbra", "PT", 40.2033, -8.4103, 409, "PT-C"),
    ("Faro", "PT", 37.0194, -7.9304, 467, "PT-S"),
    ("Santarem", "PT", 39.2362, -8.6850, 425, "PT-LVT"),
    ("Viseu", "PT", 40.6610, -7.9097, 352, "PT-C"),
    ("Viana do Castelo", "PT", 41.6918, -8.8344, 231, "PT-N"),
    ("Vila Real", "PT", 41.3006, -7.7441, 185, "PT-N"),
    ("Castelo Branco", "PT", 39.8222, -7.4909, 177, "PT-C"),
    ("Evora", "PT", 38.5714, -7.9135, 152, "PT-S"),
    ("Guarda", "PT", 40.5373, -7.2676, 143, "PT-C"),
    ("Beja", "PT", 38.0151, -7.8632, 144, "PT-S"),
    ("Braganca", "PT", 41.8061, -6.7567, 122, "PT-N"),
    ("Portalegre", "PT", 39.2967, -7.4285, 104, "PT-S"),
]

# code, location, lat, lon, max crews/day, outbound dock-truck slots/day, variable cost (EUR/unit),
# stocked categories (subset of CATEGORY codes)
FCS = [
    ("MAD1", "San Fernando de Henares", 40.4239, -3.5300, 70, 150, 0.30, "SMA"),
    ("MAD2", "Illescas", 40.1219, -3.8456, 60, 140, 0.26, "SMBA"),
    ("BCN1", "El Prat de Llobregat", 41.3264, 2.0953, 64, 140, 0.31, "SMA"),
    ("BCN2", "Martorelles", 41.5310, 2.2369, 44, 110, 0.27, "SMB"),
    ("ZAZ1", "Zaragoza PLAZA", 41.6260, -0.9800, 44, 110, 0.24, "SMBA"),
    ("SVQ1", "Dos Hermanas", 37.2833, -5.9208, 40, 100, 0.27, "SMBA"),
    ("VLC1", "Riba-roja de Turia", 39.5460, -0.5660, 36, 90, 0.28, "SMA"),
    ("OVD1", "Siero", 43.3920, -5.6620, 22, 60, 0.29, "SMA"),
    ("BIO1", "Trapagaran", 43.2950, -3.0170, 24, 60, 0.29, "SMA"),
    ("MUR1", "Murcia", 37.9870, -1.1300, 22, 60, 0.29, "SMA"),
]

# Candidate site for the network-design what-if (not part of the base network).
CANDIDATE_FCS = [
    ("LIS1", "Azambuja", 39.0700, -8.8680, 34, 90, 0.27, "SMA"),
]

# code, location, lat, lon, country
SCS = [
    ("SC-MAD", "Getafe", 40.3050, -3.7320, "ES"),
    ("SC-BCN", "Sant Boi", 41.3450, 2.0400, "ES"),
    ("SC-VLC", "Valencia", 39.4700, -0.4200, "ES"),
    ("SC-SVQ", "Sevilla", 37.4000, -5.9500, "ES"),
    ("SC-AGP", "Malaga", 36.7000, -4.4800, "ES"),
    ("SC-BIO", "Bilbao", 43.2600, -2.9200, "ES"),
    ("SC-ZAZ", "Zaragoza", 41.6500, -0.9000, "ES"),
    ("SC-LCG", "A Coruna", 43.3300, -8.4000, "ES"),
    ("SC-VLL", "Valladolid", 41.6200, -4.7500, "ES"),
    ("SC-MJV", "Murcia", 37.9800, -1.1300, "ES"),
    ("SC-LIS", "Lisboa", 38.8000, -9.1000, "PT"),
    ("SC-OPO", "Porto", 41.2000, -8.6200, "PT"),
]

# code, name, volume (package-equivalents per unit), labour content (labour units per unit),
# national share of demand, NB dispersion (higher = closer to Poisson)
CATEGORIES = [
    ("S", "Small sortable", 1.0, 1.0, 0.52, 160.0),
    ("M", "Medium sortable", 2.2, 1.4, 0.22, 110.0),
    ("B", "Bulky / non-sortable", 7.0, 3.0, 0.06, 45.0),
    ("A", "Apparel", 1.3, 0.9, 0.20, 75.0),
]

EARTH_RADIUS_KM = 6371.0
ROAD_CIRCUITY = 1.22  # road / great-circle distance ratio, typical value for Iberia


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance (km) with numpy broadcasting."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def road_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    return ROAD_CIRCUITY * haversine_km(lat1, lon1, lat2, lon2)
