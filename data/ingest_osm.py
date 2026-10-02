"""
data/ingest_osm.py - OSM road-graph ingestion and POI enrichment.

Pipeline for one zone (controlled by ZONE env var):
1. Download the road graph with OSMnx (cache to data/cache/{zone}.graphml).
2. Store every edge as a RoadSegment row.
3. Pull POIs from Overpass (street-lamps, lit tags, shops, ATMs, petrol, pharmacies,
   bus stops, metro stations, police, hospitals, factory gates, industrial landuse).
4. Mark police, hospitals, metro, 24x7-pharmacies and bus-stands as safe havens.
5. Compute per-segment:
   - lamp_count   : street lamps within 150 m
   - lit_ratio    : fraction of segment that is "lit" (OSM lit tag + lamp proximity)
   - poi_counts   : by category at 150 m and 300 m buffers
   - dist_main_m  : distance to nearest primary/secondary road (metres)
   - dist_haven_m : distance to nearest safe haven (metres)

Usage:
    python -m data.ingest_osm               # uses ZONE from .env / env var
    python -m data.ingest_osm peenya
    python -m data.ingest_osm subramanyapura

NOTE: All data sourced from OpenStreetMap (copyright OpenStreetMap contributors, ODbL).
      Lamp counts and lit_ratio for segments without explicit OSM lit= tags are
      ESTIMATED from proximity to street-lamp nodes -- clearly flagged as ESTIMATED.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import networkx as nx
import osmnx as ox
import requests
import shapely.geometry as sg
from geoalchemy2.shape import from_shape
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

# Path bootstrap (allow running from repo root)
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import ZONE_PRESETS, Settings
from app.models.poi import POI, POICategory
from app.models.road import RoadSegment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("ingest_osm")

# Constants
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LAMP_BUFFER_M = 150
POI_NEAR_M = 150
POI_FAR_M = 300

MAIN_ROAD_TYPES = {"primary", "secondary", "trunk", "motorway"}

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Namma Metro station names per zone
METRO_STATIONS: dict[str, list[str]] = {
    "peenya": ["Peenya", "Peenya Industry", "Jalahalli", "Dasarahalli", "Subramanyapura"],
    "subramanyapura": ["Subramanyapura", "Kanakapura Road"],
}

# Approximate coordinates for Namma Metro Purple Line stations
METRO_COORDS: dict[str, tuple[float, float]] = {
    "Peenya": (13.0285, 77.5086),
    "Peenya Industry": (13.0218, 77.5143),
    "Jalahalli": (13.0136, 77.5201),
    "Dasarahalli": (13.0280, 77.5007),
    "Subramanyapura": (12.8825, 77.5523),
    "Kanakapura Road": (12.8739, 77.5497),
}

POI_SPECS: list[dict[str, Any]] = [
    {
        "category": POICategory.LAMP,
        "safe_haven": False,
        "overpass": 'node["highway"="street_lamp"]',
        "name_prefix": "Street lamp",
    },
    {
        "category": POICategory.SHOP,
        "safe_haven": False,
        "overpass": 'node["shop"]',
        "name_prefix": "Shop",
    },
    {
        "category": POICategory.ATM,
        "safe_haven": False,
        "overpass": 'node["amenity"="atm"]',
        "name_prefix": "ATM",
    },
    {
        "category": POICategory.PETROL,
        "safe_haven": False,
        "overpass": 'node["amenity"="fuel"]',
        "name_prefix": "Petrol pump",
    },
    {
        "category": POICategory.PHARMACY,
        "safe_haven": False,
        "overpass": 'node["amenity"="pharmacy"]',
        "name_prefix": "Pharmacy",
    },
    {
        "category": POICategory.BUS_STOP,
        "safe_haven": True,
        "overpass": 'node["highway"="bus_stop"]',
        "name_prefix": "Bus stop",
    },
    {
        "category": POICategory.METRO,
        "safe_haven": True,
        "overpass": 'node["railway"="station"]["station"="subway"]',
        "name_prefix": "Metro",
    },
    {
        "category": POICategory.POLICE,
        "safe_haven": True,
        "overpass": 'node["amenity"="police"]',
        "name_prefix": "Police station",
    },
    {
        "category": POICategory.HOSPITAL,
        "safe_haven": True,
        "overpass": 'node["amenity"="hospital"]',
        "name_prefix": "Hospital",
    },
    {
        "category": POICategory.FACTORY_GATE,
        "safe_haven": False,
        "overpass": 'node["industrial"]["barrier"="gate"]',
        "name_prefix": "Factory gate",
    },
]

LIT_WAYS_QUERY_TMPL = (
    "[out:json][timeout:60];\n"
    "(\n"
    '  way["lit"]({south},{west},{north},{east});\n'
    ");\n"
    "out body;\n"
)


# Overpass helpers

def _overpass_query(query: str, retries: int = 4, backoff: float = 3.0) -> dict:
    """Execute an Overpass QL query with retries + exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        for url in OVERPASS_URLS:
            try:
                resp = requests.post(
                    url,
                    data={"data": query},
                    timeout=90,
                    headers={"User-Agent": "SafeWay-DataIngestion/1.0"},
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                log.warning("Overpass attempt %d at %s failed: %s", attempt + 1, url, exc)
        wait = backoff * (2 ** attempt)
        log.info("Waiting %.1f s before retry...", wait)
        time.sleep(wait)
    raise RuntimeError(f"Overpass query failed after {retries} retries: {last_exc}")


def _build_poi_query(spec: dict, south: float, west: float, north: float, east: float) -> str:
    """Build a bounding-box Overpass query for one POI spec."""
    overpass_tag = spec["overpass"]
    # Strip leading 'node' to get the tag filter portion for way queries
    tag_part = overpass_tag[4:] if overpass_tag.startswith("node") else overpass_tag
    return (
        f"[out:json][timeout:60];\n"
        f"(\n"
        f"  {overpass_tag}({south},{west},{north},{east});\n"
        f"  way{tag_part}({south},{west},{north},{east});\n"
        f");\n"
        f"out center body;\n"
    )


# Graph helpers

def load_graph(zone: str, preset) -> nx.MultiDiGraph:
    """Load OSMnx graph from cache or download."""
    cache_path = CACHE_DIR / f"{zone}.graphml"
    if cache_path.exists():
        log.info("Loading cached graph from %s", cache_path)
        G = ox.load_graphml(str(cache_path))
    else:
        log.info("Downloading OSM road graph for zone=%s...", zone)
        G = ox.graph_from_bbox(
            north=preset.bbox_north,
            south=preset.bbox_south,
            east=preset.bbox_east,
            west=preset.bbox_west,
            network_type="all",
            simplify=True,
            retain_all=False,
        )
        ox.save_graphml(G, str(cache_path))
        log.info("Graph cached to %s", cache_path)
    log.info("Graph loaded: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


# Geometry helpers

def _metres_per_degree_lat(lat: float) -> float:
    """Approximate metres per degree of latitude at given lat."""
    return (
        111_132.954
        - 559.822 * math.cos(2 * math.radians(lat))
        + 1.175 * math.cos(4 * math.radians(lat))
    )


def _approx_buffer_deg(metres: float, center_lat: float) -> float:
    return metres / _metres_per_degree_lat(center_lat)


def _build_poi_shapely_list(elements: list[dict]) -> list[tuple[dict, sg.Point]]:
    """Extract (element, shapely Point) pairs from Overpass elements."""
    result = []
    for el in elements:
        if el.get("type") == "node":
            result.append((el, sg.Point(el["lon"], el["lat"])))
        elif el.get("type") == "way" and "center" in el:
            c = el["center"]
            result.append((el, sg.Point(c["lon"], c["lat"])))
    return result


# Segment feature computation

def compute_segment_features(
    edge_geom: sg.LineString,
    lamp_points: list[sg.Point],
    lit_way_osm_ids: set[int],
    osm_id: int,
    lit_tag: str | None,
    center_lat: float,
    safe_haven_points: list[sg.Point],
    main_road_geoms: list[sg.LineString],
) -> dict:
    """
    Compute derived safety features for one road segment.

    lit_ratio is ESTIMATED from OSM lit= tag and lamp proximity.
    It is NOT measured from official streetlight surveys.
    """
    buf_deg = _approx_buffer_deg(LAMP_BUFFER_M, center_lat)
    lamp_count = sum(1 for pt in lamp_points if edge_geom.distance(pt) <= buf_deg)

    if lit_tag == "yes":
        lit_ratio = 1.0
    elif lit_tag == "no":
        lit_ratio = 0.0
    elif lit_tag == "limited":
        lit_ratio = 0.3
    elif osm_id in lit_way_osm_ids:
        lit_ratio = 0.8
    else:
        # ESTIMATED from lamp density
        seg_len_m = edge_geom.length * _metres_per_degree_lat(center_lat)
        if lamp_count == 0:
            lit_ratio = 0.0
        else:
            lamp_spacing_m = seg_len_m / lamp_count
            if lamp_spacing_m <= 30:
                lit_ratio = 0.9
            elif lamp_spacing_m <= 60:
                lit_ratio = 0.6
            else:
                lit_ratio = min(lamp_count * 0.1, 0.5)

    dist_haven_m: float | None = None
    if safe_haven_points:
        min_deg = min(edge_geom.distance(pt) for pt in safe_haven_points)
        dist_haven_m = round(min_deg * _metres_per_degree_lat(center_lat), 1)

    dist_main_m: float | None = None
    if main_road_geoms:
        min_deg = min(edge_geom.distance(r) for r in main_road_geoms)
        dist_main_m = round(min_deg * _metres_per_degree_lat(center_lat), 1)

    return {
        "lamp_count": lamp_count,
        "lit_ratio": round(lit_ratio, 4),
        "dist_haven_m": dist_haven_m,
        "dist_main_m": dist_main_m,
    }


# Database helpers

def _upsert_road_segments(
    session: Session,
    zone: str,
    G: nx.MultiDiGraph,
    lamp_points: list[sg.Point],
    lit_way_osm_ids: set[int],
    safe_haven_points: list[sg.Point],
    main_road_geoms: list[sg.LineString],
    center_lat: float,
) -> int:
    """Delete existing segments for zone then insert fresh. Returns count inserted."""
    log.info("Clearing existing road_segments for zone=%s...", zone)
    session.execute(text("DELETE FROM road_segments WHERE zone = :z"), {"z": zone})
    session.flush()

    inserted = 0
    edges = list(G.edges(data=True, keys=True))
    log.info("Processing %d graph edges...", len(edges))

    for u, v, key, data in edges:
        geom = data.get("geometry")
        if geom is None:
            n_u, n_v = G.nodes[u], G.nodes[v]
            geom = sg.LineString([(n_u["x"], n_u["y"]), (n_v["x"], n_v["y"])])
        elif not isinstance(geom, sg.LineString):
            geom = sg.LineString(geom.coords)

        osm_id = data.get("osmid", 0)
        if isinstance(osm_id, list):
            osm_id = osm_id[0]
        osm_id = int(osm_id)

        road_type = data.get("highway", "unclassified")
        if isinstance(road_type, list):
            road_type = road_type[0]
        road_type = str(road_type)

        name = data.get("name")
        if isinstance(name, list):
            name = name[0]
        name = str(name) if name else None

        length_m = float(data.get("length", 0.0))

        lit_tag_raw = data.get("lit")
        if isinstance(lit_tag_raw, list):
            lit_tag_raw = lit_tag_raw[0]
        lit_tag = str(lit_tag_raw) if lit_tag_raw else None

        features = compute_segment_features(
            edge_geom=geom,
            lamp_points=lamp_points,
            lit_way_osm_ids=lit_way_osm_ids,
            osm_id=osm_id,
            lit_tag=lit_tag,
            center_lat=center_lat,
            safe_haven_points=safe_haven_points,
            main_road_geoms=main_road_geoms,
        )

        seg = RoadSegment(
            osm_id=osm_id,
            geometry=from_shape(geom, srid=4326),
            road_type=road_type[:60],
            name=name[:200] if name else None,
            length_m=length_m,
            lit_tag=lit_tag[:10] if lit_tag else None,
            lamp_count=features["lamp_count"],
            lit_ratio=features["lit_ratio"],
            zone=zone,
        )
        session.add(seg)
        inserted += 1

        if inserted % 500 == 0:
            session.flush()
            log.info("  ... %d segments written", inserted)

    session.flush()
    log.info("Road segments inserted: %d", inserted)
    return inserted


def _upsert_pois(session: Session, zone: str, all_pois: list[dict]) -> dict[str, int]:
    """Delete existing POIs for zone then insert fresh. Returns category counts."""
    log.info("Clearing existing POIs for zone=%s...", zone)
    session.execute(text("DELETE FROM pois WHERE zone = :z"), {"z": zone})
    session.flush()

    counts: dict[str, int] = {}
    seen: set[tuple] = set()

    for poi_dict in all_pois:
        key = (poi_dict["category"].value, round(poi_dict["lon"], 5), round(poi_dict["lat"], 5))
        if key in seen:
            continue
        seen.add(key)

        poi = POI(
            name=poi_dict["name"][:200],
            category=poi_dict["category"],
            geometry=from_shape(sg.Point(poi_dict["lon"], poi_dict["lat"]), srid=4326),
            safe_haven=poi_dict["safe_haven"],
            zone=zone,
        )
        session.add(poi)
        cat_key = poi_dict["category"].value
        counts[cat_key] = counts.get(cat_key, 0) + 1

    session.flush()
    return counts


# Main ingestion pipeline

def ingest_zone(zone: str, db_url: str) -> dict:
    """
    Full ingestion pipeline for one zone.
    Returns a summary dict (also printed to stdout).
    """
    if zone not in ZONE_PRESETS:
        raise ValueError(f"Unknown zone '{zone}'. Valid: {list(ZONE_PRESETS)}")

    preset = ZONE_PRESETS[zone]
    south, west, north, east = (
        preset.bbox_south, preset.bbox_west, preset.bbox_north, preset.bbox_east,
    )

    # 1. Road graph
    G = load_graph(zone, preset)

    # 2. Lit-way OSM IDs from Overpass
    log.info("Fetching lit-tagged ways from Overpass...")
    lit_query = LIT_WAYS_QUERY_TMPL.format(south=south, west=west, north=north, east=east)
    try:
        lit_data = _overpass_query(lit_query)
        lit_way_osm_ids: set[int] = {
            int(el["id"])
            for el in lit_data.get("elements", [])
            if el.get("tags", {}).get("lit") == "yes"
        }
        log.info("Lit-way OSM IDs: %d", len(lit_way_osm_ids))
    except Exception as exc:
        log.warning("Could not fetch lit ways: %s - continuing without", exc)
        lit_way_osm_ids = set()

    # 3. POIs from Overpass
    all_pois_raw: list[dict] = []
    lamp_points: list[sg.Point] = []
    safe_haven_points: list[sg.Point] = []

    for spec in POI_SPECS:
        cat = spec["category"]
        safe = spec["safe_haven"]
        name_pfx = spec["name_prefix"]

        log.info("Fetching %s...", name_pfx)
        query = _build_poi_query(spec, south, west, north, east)
        try:
            data = _overpass_query(query)
            element_pairs = _build_poi_shapely_list(data.get("elements", []))
        except Exception as exc:
            log.warning("POI fetch failed for %s: %s - skipping", name_pfx, exc)
            element_pairs = []

        log.info("  -> %d %s", len(element_pairs), name_pfx)

        for el, pt in element_pairs:
            tags = el.get("tags", {})
            name = (
                tags.get("name")
                or tags.get("name:en")
                or f"{name_pfx} ({pt.y:.4f},{pt.x:.4f})"
            )

            is_safe = safe
            if cat == POICategory.PHARMACY:
                oh = tags.get("opening_hours", "")
                if "24/7" in oh or "24 hours" in oh.lower():
                    is_safe = True

            poi_dict = {
                "category": cat,
                "lat": pt.y,
                "lon": pt.x,
                "name": name,
                "safe_haven": is_safe,
            }
            all_pois_raw.append(poi_dict)
            if cat == POICategory.LAMP:
                lamp_points.append(pt)
            if is_safe:
                safe_haven_points.append(pt)

    # 4. Namma Metro stations (hardcoded fallback if OSM is sparse)
    metro_names = METRO_STATIONS.get(zone, [])
    metro_in_zone = [p for p in all_pois_raw if p["category"] == POICategory.METRO]
    if len(metro_in_zone) < len(metro_names):
        log.info(
            "OSM has %d metro stations; expected %d - adding SYNTHETIC_MARKER entries",
            len(metro_in_zone), len(metro_names),
        )
        existing_names_lc = {p["name"].lower() for p in metro_in_zone}
        for mname in metro_names:
            if mname.lower() not in existing_names_lc and mname in METRO_COORDS:
                mlat, mlon = METRO_COORDS[mname]
                pd = {
                    "category": POICategory.METRO,
                    "lat": mlat,
                    "lon": mlon,
                    "name": f"{mname} Metro Station (SYNTHETIC_MARKER)",
                    "safe_haven": True,
                }
                all_pois_raw.append(pd)
                safe_haven_points.append(sg.Point(mlon, mlat))
                log.info("  Added SYNTHETIC_MARKER metro: %s", mname)

    # 5. Main-road geometries for dist_main_m computation
    main_road_geoms: list[sg.LineString] = []
    for u, v, key, data in G.edges(data=True, keys=True):
        hw = data.get("highway", "")
        if isinstance(hw, list):
            hw = hw[0]
        if str(hw) in MAIN_ROAD_TYPES:
            geom = data.get("geometry")
            if geom is None:
                n_u, n_v = G.nodes[u], G.nodes[v]
                geom = sg.LineString([(n_u["x"], n_u["y"]), (n_v["x"], n_v["y"])])
            main_road_geoms.append(geom)
    log.info("Main-road edges: %d", len(main_road_geoms))

    # 6. Persist
    engine = create_engine(db_url, echo=False)
    with Session(engine) as session:
        n_segs = _upsert_road_segments(
            session=session,
            zone=zone,
            G=G,
            lamp_points=lamp_points,
            lit_way_osm_ids=lit_way_osm_ids,
            safe_haven_points=safe_haven_points,
            main_road_geoms=main_road_geoms,
            center_lat=preset.center_lat,
        )
        poi_counts = _upsert_pois(session, zone, all_pois_raw)
        session.commit()
        log.info("Database commit complete.")

    # 7. Summary — query the DB for actual lit_ratio statistics
    #    (The raw graph edge 'lit' attribute is often absent after OSMnx
    #     simplification; the real lit_ratio was computed per-segment above.)
    with Session(engine) as session:
        row = session.execute(
            text(
                "SELECT COUNT(*) FILTER (WHERE lit_ratio > 0) AS lit_count, "
                "AVG(lit_ratio) AS avg_ratio "
                "FROM road_segments WHERE zone = :z"
            ),
            {"z": zone},
        ).one()
        n_lit = row.lit_count or 0
        avg_lit = round(float(row.avg_ratio or 0.0), 4)

    lit_share = round(n_lit / max(n_segs, 1), 3)

    summary = {
        "zone": zone,
        "segment_count": n_segs,
        "poi_counts": poi_counts,
        "lit_segment_share": lit_share,
        "avg_lit_ratio": avg_lit,
        "safe_haven_count": len(safe_haven_points),
    }
    _print_summary(summary)
    return summary


def _print_summary(s: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  OSM INGESTION SUMMARY - zone={s['zone']}")
    print("=" * 60)
    print(f"  Road segments     : {s['segment_count']:,}")
    print(f"  Lit segment share : {s['lit_segment_share']:.1%}  (lit_ratio > 0)")
    print(f"  Avg lit_ratio     : {s.get('avg_lit_ratio', 0.0):.4f}")
    print(f"  Safe havens       : {s['safe_haven_count']}")
    print("  POI counts by category:")
    for cat, cnt in sorted(s["poi_counts"].items()):
        print(f"    {cat:<20} {cnt:>5}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest OSM data for a SafeWay zone.")
    parser.add_argument(
        "zone",
        nargs="?",
        default=None,
        help="Zone name (peenya|subramanyapura). Defaults to ZONE env var.",
    )
    args = parser.parse_args()
    s = Settings()
    zone = args.zone or s.ZONE
    ingest_zone(zone, s.DATABASE_URL_SYNC)


if __name__ == "__main__":
    main()
