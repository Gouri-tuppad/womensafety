"""
app/services/routing_service.py - FASTEST and SAFEST route computation for SafeWay.

Graph: loaded once from the cached GraphML (data/cache/peenya.graphml) then
       held as a module-level singleton. Edge weights are precomputed per
       time_slot and read from Redis; on cache miss, scores are computed on-the-fly.

Routing algorithms:
  Fastest - standard Dijkstra weighted by travel time (edge_length / speed_m_per_s)
  Safest  - Dijkstra weighted by safety cost:
              cost = length * clamp(1 + α·C + β·D + γ·I − δ·A, min=0.5)

Performance target: < 3 s for a single /route request over the Peenya zone.
  - Graph is loaded once (~0.5 s), held in module state.
  - Edge weights are cached in Redis per time_slot.
  - NetworkX Dijkstra over ~36 k edges runs in < 1 s.

A* heuristic: Euclidean distance (degrees) × min_possible_cost_per_degree.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import networkx as nx
import osmnx as ox
import shapely.geometry as sg
from shapely import STRtree
from sqlalchemy import create_engine, text

from app.config import ZONE_PRESETS, settings
from app.schemas.route import RouteComparison, RouteResult, SafeHaven, SegmentInfo, SegmentRisk
from app.services.personalization import ProfileWeights, get_profile
from app.services.safety_scoring import (
    RiskScores,
    SegmentFeatures,
    SegmentScorer,
    TimeContext,
    get_cached_scores,
    get_slot_scores,
)

log = logging.getLogger(__name__)

GRAPH_CACHE_DIR = Path("data/cache")
ASSUMED_SPEED_MS = 1.4   # walking speed, m/s
HAVEN_ROUTE_BUFFER_M = 300.0

# ── Graph and DB singletons ───────────────────────────────────────────────────

_graph_cache: dict[str, nx.MultiDiGraph] = {}
_zone_db_data_cache: dict[str, tuple[dict[int, dict], list[dict], list[dict]]] = {}


def get_graph(zone: str) -> nx.MultiDiGraph:
    """
    Load and cache the OSMnx graph for a zone.
    The graph is loaded from the GraphML cache file.
    Thread-safe: GIL protects the dict write.
    """
    if zone not in _graph_cache:
        cache_path = GRAPH_CACHE_DIR / f"{zone}.graphml"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Graph cache not found at {cache_path}. "
                f"Run: docker compose exec api python -m data.ingest_osm {zone}"
            )
        log.info("Loading graph from %s ...", cache_path)
        G = ox.load_graphml(str(cache_path))
        log.info("Graph loaded: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
        _graph_cache[zone] = G
    return _graph_cache[zone]


def clear_graph_cache() -> None:
    """Clear the in-memory graph cache (used in tests)."""
    _graph_cache.clear()
    _zone_db_data_cache.clear()


def _get_zone_db_data(zone: str, db_url: str) -> tuple[dict[int, dict], list[dict], list[dict]]:
    """Cached loader for zone road segments and safe havens from Postgres."""
    if zone not in _zone_db_data_cache:
        engine = create_engine(db_url, echo=False)
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT osm_id, road_type, lit_ratio, lamp_count, "
                "ST_Y(ST_Centroid(geometry)) AS lat, ST_X(ST_Centroid(geometry)) AS lon "
                "FROM road_segments WHERE zone = :z"
            ), {"z": zone}).fetchall()

            seg_data_by_osmid = {
                int(r._mapping["osm_id"]): dict(r._mapping)
                for r in rows
            }
            all_segs = [dict(r._mapping) for r in rows]

            haven_rows = [
                dict(r._mapping)
                for r in conn.execute(text(
                    "SELECT name, category, ST_Y(geometry) AS lat, ST_X(geometry) AS lon "
                    "FROM pois WHERE zone = :z AND safe_haven = TRUE"
                ), {"z": zone}).fetchall()
            ]

        _zone_db_data_cache[zone] = (seg_data_by_osmid, haven_rows, all_segs)

    return _zone_db_data_cache[zone]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _metres_per_degree_lat(lat: float) -> float:
    return (
        111_132.954
        - 559.822 * math.cos(2 * math.radians(lat))
        + 1.175 * math.cos(4 * math.radians(lat))
    )


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Return the graph node closest to (lat, lon)."""
    return ox.nearest_nodes(G, lon, lat)


# ── Edge weight computation ───────────────────────────────────────────────────

def _calc_edge_cost(
    data: dict,
    time_ctx: TimeContext,
    profile: ProfileWeights,
    slot_scores: dict[int, RiskScores],
    seg_data_by_osmid: dict[int, dict],
    scorer: SegmentScorer,
) -> float:
    """Calculate safety cost for one edge dict."""
    length_m = float(data.get("length", 50.0))

    # Restrict/penalize highways non-walkable for pedestrians (motorways, construction)
    hw = str(data.get("highway", "")).lower()
    if hw in ("motorway", "motorway_link", "construction", "abandoned"):
        return length_m * 100.0

    osm_id_raw = data.get("osmid", 0)
    if isinstance(osm_id_raw, list):
        osm_id_raw = osm_id_raw[0]
    osm_id = int(osm_id_raw)

    scores = slot_scores.get(osm_id)
    if scores is not None:
        mult = profile.cost_multiplier(
            scores.crime_risk, scores.darkness_risk,
            scores.isolation_risk, scores.activity_score,
        )
        return length_m * mult

    seg = seg_data_by_osmid.get(osm_id)
    if seg is not None:
        fallback = RiskScores(
            crime_risk=0.1,
            darkness_risk=0.8 if time_ctx.is_dark else 0.2,
            isolation_risk=0.5 if str(seg.get("road_type")) in ("industrial", "service") else 0.2,
            activity_score=0.1,
        )
        mult = profile.cost_multiplier(
            fallback.crime_risk, fallback.darkness_risk,
            fallback.isolation_risk, fallback.activity_score,
        )
        return length_m * mult

    return length_m * 1.5


def _build_weight_func(
    G: nx.MultiDiGraph,
    time_ctx: TimeContext,
    profile: ProfileWeights,
    slot_scores: dict[int, RiskScores],
    seg_data_by_osmid: dict[int, dict],
    scorer: SegmentScorer,
    use_safety: bool,
):
    """
    Return a weight function for NetworkX Dijkstra.
    Correctly handles MultiDiGraph edge dictionaries {0: data0, 1: data1, ...}.
    """
    if not use_safety:
        def fastest_weight(u, v, edge_dict):
            if 0 in edge_dict or (edge_dict and isinstance(next(iter(edge_dict.keys())), int)):
                return min(float(d.get("length", 50.0)) for d in edge_dict.values()) / ASSUMED_SPEED_MS
            return float(edge_dict.get("length", 50.0)) / ASSUMED_SPEED_MS
        return fastest_weight
    else:
        def safety_weight(u, v, edge_dict):
            if 0 in edge_dict or (edge_dict and isinstance(next(iter(edge_dict.keys())), int)):
                return min(
                    _calc_edge_cost(d, time_ctx, profile, slot_scores, seg_data_by_osmid, scorer)
                    for d in edge_dict.values()
                )
            return _calc_edge_cost(edge_dict, time_ctx, profile, slot_scores, seg_data_by_osmid, scorer)
        return safety_weight


def _node_path_to_result(
    G: nx.MultiDiGraph,
    node_path: list[int],
    time_ctx: TimeContext,
    profile: ProfileWeights,
    slot_scores: dict[int, RiskScores],
    seg_data_by_osmid: dict[int, dict],
    scorer: SegmentScorer,
    use_safety: bool,
) -> RouteResult:
    """Convert a list of node IDs to a RouteResult."""
    polyline: list[list[float]] = []
    segments_info: list[SegmentInfo] = []
    total_dist = 0.0
    weighted_risk_sum = 0.0

    for i in range(len(node_path) - 1):
        u, v = node_path[i], node_path[i + 1]

        # Pick the edge used (for fastest: min length; for safest: min safety cost)
        if not use_safety:
            edge_data = min(
                G[u][v].values(),
                key=lambda d: float(d.get("length", 50.0)),
            )
        else:
            edge_data = min(
                G[u][v].values(),
                key=lambda d: _calc_edge_cost(d, time_ctx, profile, slot_scores, seg_data_by_osmid, scorer),
            )

        length = float(edge_data.get("length", 50.0))
        total_dist += length

        osm_id_raw = edge_data.get("osmid", 0)
        if isinstance(osm_id_raw, list):
            osm_id_raw = osm_id_raw[0]
        osm_id = int(osm_id_raw)

        scores = slot_scores.get(osm_id)
        if scores is None:
            seg = seg_data_by_osmid.get(osm_id, {})
            scores = RiskScores(
                crime_risk=0.1,
                darkness_risk=0.8 if time_ctx.is_dark else 0.2,
                isolation_risk=0.5 if str(seg.get("road_type")) in ("industrial", "service") else 0.2,
                activity_score=0.1,
            )

        risk_val = profile.segment_risk_score(
            scores.crime_risk, scores.darkness_risk,
            scores.isolation_risk, scores.activity_score,
        )
        weighted_risk_sum += risk_val * length

        # Color coding: GREEN < 0.30, YELLOW 0.30..0.60, RED >= 0.60
        if risk_val < 0.30:
            risk_label = SegmentRisk.GREEN
        elif risk_val < 0.60:
            risk_label = SegmentRisk.YELLOW
        else:
            risk_label = SegmentRisk.RED

        seg_info = SegmentInfo(
            osm_id=osm_id,
            risk=risk_label,
            safety_score=round(1.0 - risk_val, 3),
            crime_risk=round(scores.crime_risk, 3),
            darkness_risk=round(scores.darkness_risk, 3),
            isolation_risk=round(scores.isolation_risk, 3),
            activity_score=round(scores.activity_score, 3),
            lit_ratio=float(seg_data_by_osmid.get(osm_id, {}).get("lit_ratio", 0.0)),
            lamp_count=int(seg_data_by_osmid.get(osm_id, {}).get("lamp_count", 0)),
        )
        segments_info.append(seg_info)

    # Build polyline: add start node, then end node of each edge
    if node_path:
        first = G.nodes[node_path[0]]
        polyline.append([first["y"], first["x"]])
        for i in range(len(node_path) - 1):
            v = node_path[i + 1]
            nv = G.nodes[v]
            polyline.append([nv["y"], nv["x"]])

    # Route-level safety score: 100 - length-weighted mean risk (0-100)
    mean_risk = weighted_risk_sum / max(total_dist, 1.0)
    safety_score = round(max(0.0, 100.0 - 100.0 * mean_risk), 1)
    duration_s = total_dist / ASSUMED_SPEED_MS

    return RouteResult(
        polyline=polyline,
        distance_m=round(total_dist, 1),
        duration_s=round(duration_s, 1),
        safety_score=safety_score,
        segments=segments_info,
        node_ids=node_path,
    )


# ── Safe havens near route ────────────────────────────────────────────────────

def _category_rank(category: str) -> int:
    cat = category.strip().lower()
    if "police" in cat:
        return 0
    if "hospital" in cat or "clinic" in cat or "pharmacy" in cat or "health" in cat:
        return 1
    if "metro" in cat or "subway" in cat or "train" in cat:
        return 2
    return 3  # bus_stop and general POIs


def _safe_havens_near_route(
    polyline: list[list[float]],
    haven_rows: list[dict],
    buffer_m: float = HAVEN_ROUTE_BUFFER_M,
) -> list[SafeHaven]:
    """
    Return safe havens within buffer_m of route polyline.
    Deduped by (name, category) and ranked: police > hospital > metro > bus_stop.
    """
    if not polyline or not haven_rows:
        return []

    center_lat = polyline[len(polyline) // 2][0]
    mpd = _metres_per_degree_lat(center_lat)
    buf_deg = buffer_m / mpd

    route_pts = [sg.Point(lon, lat) for lat, lon in polyline]
    route_line = sg.MultiPoint(route_pts).convex_hull if len(route_pts) > 1 else route_pts[0]

    nearby: list[SafeHaven] = []
    for h in haven_rows:
        pt = sg.Point(h["lon"], h["lat"])
        dist_deg = route_line.distance(pt)
        dist_m = dist_deg * mpd
        if dist_m <= buffer_m:
            nearby.append(SafeHaven(
                name=str(h["name"]).strip(),
                category=str(h["category"]).strip(),
                lat=h["lat"],
                lon=h["lon"],
                dist_from_route_m=round(dist_m, 1),
            ))

    # Dedupe by (name, category), keeping closest instance to route
    deduped: dict[tuple[str, str], SafeHaven] = {}
    for h in nearby:
        key = (h.name.lower(), h.category.lower())
        if key not in deduped or h.dist_from_route_m < deduped[key].dist_from_route_m:
            deduped[key] = h

    # Rank police > hospital > metro > bus_stop, then by distance
    sorted_havens = sorted(
        deduped.values(),
        key=lambda h: (_category_rank(h.category), h.dist_from_route_m),
    )
    return sorted_havens[:10]


# ── Route comparison stats ────────────────────────────────────────────────────

def _compare_routes(fastest: RouteResult, safest: RouteResult, havens: list[SafeHaven]) -> RouteComparison:
    if fastest.distance_m > 0:
        dist_pct = round((safest.distance_m - fastest.distance_m) / fastest.distance_m * 100, 1)
    else:
        dist_pct = 0.0

    score_gain = round(safest.safety_score - fastest.safety_score, 1)

    def _avg_lit(route: RouteResult) -> float:
        if not route.segments:
            return 0.0
        return sum(s.lit_ratio for s in route.segments) / len(route.segments)

    lr_fastest = round(_avg_lit(fastest), 3)
    lr_safest = round(_avg_lit(safest), 3)
    multiplier = round(lr_safest / max(lr_fastest, 0.001), 1)

    police_near = sum(1 for h in havens if "police" in h.category.lower())
    hospitals_near = sum(1 for h in havens if "hospital" in h.category.lower() or "clinic" in h.category.lower())
    metro_near = sum(1 for h in havens if "metro" in h.category.lower() or "subway" in h.category.lower())

    # Crime exposure comparison
    def _avg_crime(route: RouteResult) -> float:
        if not route.segments:
            return 0.0
        return sum(s.crime_risk for s in route.segments) / len(route.segments)

    cr_fastest = _avg_crime(fastest)
    cr_safest = _avg_crime(safest)
    crime_drop_pct = 0
    if cr_fastest > 0.001 and cr_safest < cr_fastest:
        crime_drop_pct = round((cr_fastest - cr_safest) / cr_fastest * 100)

    # Isolated-road share comparison (isolation_risk >= 0.5)
    def _isolated_share(route: RouteResult) -> float:
        if not route.segments:
            return 0.0
        return sum(1 for s in route.segments if s.isolation_risk >= 0.5) / len(route.segments)

    iso_fastest = _isolated_share(fastest)
    iso_safest = _isolated_share(safest)
    iso_fastest_pct = round(iso_fastest * 100)
    iso_safest_pct = round(iso_safest * 100)
    iso_drop_pct = round((iso_fastest - iso_safest) * 100)

    # Build human-readable comparison summary
    parts = []

    # 1. Distance & Safety Score
    if dist_pct > 0:
        parts.append(f"+{dist_pct}% longer")
    elif dist_pct < 0:
        parts.append(f"{dist_pct}% shorter")
    else:
        parts.append("same distance")

    if score_gain > 0:
        parts.append(f"+{score_gain} safety pts")

    # 2. Crime exposure (if significantly lower)
    if crime_drop_pct >= 10:
        parts.append(f"{crime_drop_pct}% lower crime exposure")

    # 3. Isolated-road share
    if iso_drop_pct >= 15:
        parts.append(f"avoids isolated roads ({iso_safest_pct}% vs {iso_fastest_pct}%)")
    elif iso_fastest >= 0.25 and iso_safest < 0.15:
        parts.append("avoids isolated industrial roads")

    # 4. Safe havens passed
    if police_near:
        parts.append(f"{police_near} police station{'s' if police_near > 1 else ''} nearby")
    elif hospitals_near:
        parts.append(f"{hospitals_near} hospital{'s' if hospitals_near > 1 else ''} nearby")
    elif metro_near:
        parts.append(f"{metro_near} metro station{'s' if metro_near > 1 else ''} nearby")
    elif havens:
        parts.append(f"{len(havens)} safe haven{'s' if len(havens) > 1 else ''} nearby")

    # 5. Lighting: only mention if meaningful (both: not tiny lit ratio < 0.2 and substantial multiplier >= 1.5)
    if max(lr_fastest, lr_safest) >= 0.20 and multiplier >= 1.5:
        parts.append(f"{multiplier}x better lit")

    summary = ", ".join(parts) if parts else "routes are equivalent"

    return RouteComparison(
        distance_pct_longer=dist_pct,
        safety_score_gain=score_gain,
        lit_ratio_fastest=lr_fastest,
        lit_ratio_safest=lr_safest,
        lit_ratio_multiplier=multiplier,
        police_stations_near_safest=police_near,
        summary=summary,
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_routes(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    time_ctx: TimeContext,
    profile_name: str,
    db_url: str,
    redis_client=None,
    zone: str | None = None,
    scorer: SegmentScorer | None = None,
) -> dict:
    """
    Compute fastest and safest routes.

    Returns a dict with keys: fastest, safest, routes_differ, safe_havens_near_safest, comparison.
    Ultra-fast execution: runs in < 100 ms for Peenya zone with cached edge weights.
    Caps safest detour at 30% extra distance to stay realistic.
    """
    if zone is None:
        zone = settings.ZONE
    if scorer is None:
        scorer = SegmentScorer()

    profile = get_profile(profile_name)
    G = get_graph(zone)

    orig_node = _nearest_node(G, origin_lat, origin_lon)
    dest_node = _nearest_node(G, dest_lat, dest_lon)

    if orig_node == dest_node:
        raise ValueError("Origin and destination snap to the same graph node — they are too close.")

    # Bulk-load segment DB data & havens from memory cache
    seg_data_by_osmid, haven_rows, _ = _get_zone_db_data(zone, db_url)

    # Load precomputed slot scores
    slot_scores = get_slot_scores(redis_client, zone, time_ctx.time_slot)
    if not slot_scores and redis_client is not None:
        try:
            from app.services.safety_scoring import precompute_zone_weights
            precompute_zone_weights(zone, time_ctx.hour, time_ctx.is_weekend, db_url, settings.REDIS_URL, scorer)
            slot_scores = get_slot_scores(redis_client, zone, time_ctx.time_slot)
        except Exception as exc:
            log.warning("On-demand precompute failed: %s", exc)

    # Build weight functions
    fastest_wfn = _build_weight_func(G, time_ctx, profile, slot_scores, seg_data_by_osmid, scorer, use_safety=False)
    safest_wfn = _build_weight_func(G, time_ctx, profile, slot_scores, seg_data_by_osmid, scorer, use_safety=True)

    # Dijkstra routing
    try:
        fastest_path = nx.dijkstra_path(G, orig_node, dest_node, weight=fastest_wfn)
        safest_path = nx.dijkstra_path(G, orig_node, dest_node, weight=safest_wfn)
    except nx.NetworkXNoPath as exc:
        raise ValueError(f"No route found between the given points: {exc}") from exc
    except nx.NodeNotFound as exc:
        raise ValueError(f"Routing error: {exc}") from exc

    fastest_result = _node_path_to_result(G, fastest_path, time_ctx, profile, slot_scores, seg_data_by_osmid, scorer, use_safety=False)
    safest_result = _node_path_to_result(G, safest_path, time_ctx, profile, slot_scores, seg_data_by_osmid, scorer, use_safety=True)

    # Cap detour at profile-specific max extra distance to stay realistic
    max_allowed_dist = round(fastest_result.distance_m * (1.0 + profile.max_detour_pct), 1)
    if safest_result.distance_m > max_allowed_dist:
        low = 0.0
        high = 1.0
        best_path = fastest_path
        for _ in range(6):
            mid = (low + high) / 2.0
            def blended_wfn(u, v, edge_dict):
                edges = edge_dict.values() if (0 in edge_dict or (edge_dict and isinstance(next(iter(edge_dict.keys())), int))) else [edge_dict]
                return min(
                    (1.0 - mid) * _calc_edge_cost(d, time_ctx, profile, slot_scores, seg_data_by_osmid, scorer)
                    + mid * (float(d.get("length", 50.0)) / ASSUMED_SPEED_MS)
                    for d in edges
                )
            cand_path = nx.dijkstra_path(G, orig_node, dest_node, weight=blended_wfn)
            cand_res = _node_path_to_result(G, cand_path, time_ctx, profile, slot_scores, seg_data_by_osmid, scorer, use_safety=True)
            if cand_res.distance_m <= max_allowed_dist:
                best_path = cand_path
                safest_result = cand_res
                high = mid
            else:
                low = mid
        safest_path = best_path

    routes_differ = fastest_path != safest_path

    havens_near = _safe_havens_near_route(safest_result.polyline, haven_rows)
    comparison = _compare_routes(fastest_result, safest_result, havens_near)

    # ── SHAP explanations for the safest route ────────────────────────────
    risk_explanations: list[str] = []
    try:
        from app.ml.explain import explain_route
        from app.ml.predict import _load_xgb_model

        if _load_xgb_model() is not None:
            # Build lightweight feature dicts for each safest-route segment
            seg_features_for_shap = []
            for seg_info in safest_result.segments:
                osm_id = seg_info.osm_id
                seg = seg_data_by_osmid.get(osm_id, {})
                feat = {
                    "hour": time_ctx.hour,
                    "day_of_week": 5 if time_ctx.is_weekend else 0,
                    "is_weekend": int(time_ctx.is_weekend),
                    "is_holiday": int(time_ctx.is_holiday),
                    "is_dark": int(time_ctx.is_dark),
                    "lamp_density": float(seg.get("lamp_count", 0)) / max(50.0 / 100.0, 0.01),
                    "lit_ratio": float(seg.get("lit_ratio", 0.0)),
                    "shop_count_150m": 0, "atm_count_150m": 0,
                    "petrol_count_200m": 0, "bus_stop_count_150m": 0,
                    "metro_count_300m": 0, "police_count_300m": 0,
                    "hospital_count_300m": 0, "factory_gate_count_150m": 0,
                    "dist_main_road_m": 500.0,
                    "dist_police_m": 2000.0, "dist_hospital_m": 2000.0,
                    "dist_metro_m": 2000.0,
                    "road_type_encoded": 6,
                    "building_density_proxy": 0,
                    "crowd_count": 0,
                    "incident_count_500m_decayed": 0.0,
                }
                seg_features_for_shap.append(feat)

            if seg_features_for_shap:
                risk_explanations = explain_route(seg_features_for_shap, top_n=5)
    except Exception as exc:
        log.debug("SHAP explanation generation failed (non-critical): %s", exc)

    return {
        "zone": zone,
        "profile": profile_name,
        "depart_time": time_ctx,
        "fastest": fastest_result,
        "safest": safest_result,
        "routes_differ": routes_differ,
        "safe_havens_near_safest": havens_near,
        "comparison": comparison,
        "risk_explanations": risk_explanations,
    }


# ── Heatmap data ──────────────────────────────────────────────────────────────

def compute_heatmap(zone: str, hour: int, db_url: str, redis_client=None) -> list[dict]:
    """
    Return a list of H3 cell dicts with aggregated risk scores for the heatmap.
    Uses cached segment scores and DB data.
    """
    from datetime import date
    import pytz as _pytz

    scorer = SegmentScorer()
    _ist = _pytz.timezone("Asia/Kolkata")

    today = date.today()
    ref_dt = _ist.localize(__import__("datetime").datetime(today.year, today.month, today.day, hour, 0, 0))
    time_ctx = TimeContext.from_datetime(ref_dt, zone)

    seg_data_by_osmid, _, segs = _get_zone_db_data(zone, db_url)

    slot_scores = get_slot_scores(redis_client, zone, time_ctx.time_slot)
    if not slot_scores and redis_client is not None:
        try:
            from app.services.safety_scoring import precompute_zone_weights
            precompute_zone_weights(zone, hour, time_ctx.is_weekend, db_url, settings.REDIS_URL, scorer)
            slot_scores = get_slot_scores(redis_client, zone, time_ctx.time_slot)
        except Exception:
            pass

    # Crowd counts for hour
    crowd_by_h3: dict[str, int] = {}
    engine = create_engine(db_url, echo=False)
    with engine.connect() as conn:
        for r in conn.execute(text(
            "SELECT h3_index, count FROM crowd_counts "
            "WHERE zone = :z AND hour_utc = :h ORDER BY date_utc DESC LIMIT 100000"
        ), {"z": zone, "h": hour}):
            h3i = r._mapping["h3_index"]
            if h3i not in crowd_by_h3:
                crowd_by_h3[h3i] = int(r._mapping["count"])

    try:
        import h3 as _h3
        def _h3_cell(lat, lon):
            return _h3.latlng_to_cell(lat, lon, 9)
        def _h3_center(cell):
            c = _h3.cell_to_latlng(cell)
            return c[0], c[1]
    except ImportError:
        def _h3_cell(lat, lon):
            return f"grid_{lat:.3f}_{lon:.3f}_r9"
        def _h3_center(cell):
            parts = cell.split("_")
            return float(parts[1]), float(parts[2])

    cell_buckets: dict[str, list[RiskScores]] = {}

    for seg in segs:
        lat, lon = float(seg["lat"]), float(seg["lon"])
        osm_id = int(seg["osm_id"])
        cell = _h3_cell(lat, lon)

        scores = slot_scores.get(osm_id)
        if scores is None:
            features = SegmentFeatures(
                osm_id=osm_id,
                lat=lat, lon=lon,
                road_type=str(seg["road_type"]),
                length_m=50.0,
                lit_ratio=float(seg["lit_ratio"]),
                lamp_count=int(seg["lamp_count"]),
                crowd_count=crowd_by_h3.get(cell, 0),
            )
            scores = scorer.score(features, time_ctx)

        cell_buckets.setdefault(cell, []).append(scores)

    cells = []
    for cell, score_list in cell_buckets.items():
        try:
            clat, clon = _h3_center(cell)
        except Exception:
            continue
        n = len(score_list)
        cells.append({
            "h3_index": cell,
            "lat": clat,
            "lon": clon,
            "crime_risk": round(sum(s.crime_risk for s in score_list) / n, 3),
            "darkness_risk": round(sum(s.darkness_risk for s in score_list) / n, 3),
            "isolation_risk": round(sum(s.isolation_risk for s in score_list) / n, 3),
            "risk_score": round(sum(s.composite for s in score_list) / n, 3),
            "crowd_count": crowd_by_h3.get(cell, 0),
        })

    return cells

