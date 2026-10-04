"""
Step 1 + Step 3: Diagnose multiplier spread and find good fixture pairs.

Run inside Docker:
  docker compose exec api python scripts/find_good_pairs.py

Or from venv (with Postgres + Redis running):
  python scripts/find_good_pairs.py
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import statistics
from datetime import datetime

import pytz
import redis

from app.config import settings
from app.services.routing_service import (
    get_graph, _get_zone_db_data, _nearest_node, _haversine_m, compute_routes,
)
from app.services.safety_scoring import (
    TimeContext, RiskScores, SegmentScorer,
    precompute_zone_weights, get_slot_scores, _in_memory_slot_cache,
)
from app.services.personalization import get_profile

IST = pytz.timezone("Asia/Kolkata")
ZONE = "peenya"


def step1_diagnose_multiplier():
    """Print min/max/mean/stdev of edge cost multiplier for woman_night at 23:00."""
    print("=" * 72)
    print("STEP 1: Edge cost multiplier diagnosis (woman_night, 23:00)")
    print("=" * 72)

    rc = redis.from_url(settings.REDIS_URL, decode_responses=True)
    profile = get_profile("woman_night")
    print(f"Profile: {profile.label}")
    print(f"  alpha={profile.alpha}, beta={profile.beta}, gamma={profile.gamma}, delta={profile.delta}")
    print(f"  max_detour_pct={profile.max_detour_pct}")

    # Force re-precompute with normalisation
    _in_memory_slot_cache.clear()
    precompute_zone_weights(ZONE, 23, False, settings.DATABASE_URL_SYNC, settings.REDIS_URL)

    dt_23 = IST.localize(datetime(2026, 10, 2, 23, 0, 0))
    ctx = TimeContext.from_datetime(dt_23, ZONE)
    slot_scores = get_slot_scores(rc, ZONE, ctx.time_slot)
    print(f"\nSlot scores loaded: {len(slot_scores)} segments")

    G = get_graph(ZONE)
    seg_data_by_osmid, _, _ = _get_zone_db_data(ZONE, settings.DATABASE_URL_SYNC)

    mults = []
    mults_with_scores = []
    missing = 0

    for u, v, k, data in G.edges(keys=True, data=True):
        hw = str(data.get("highway", "")).lower()
        if hw in ("motorway", "motorway_link", "construction", "abandoned"):
            continue

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
            mults.append(mult)
            mults_with_scores.append((mult, scores))
        else:
            seg = seg_data_by_osmid.get(osm_id)
            if seg is not None:
                fallback = RiskScores(
                    crime_risk=0.1,
                    darkness_risk=0.8 if ctx.is_dark else 0.2,
                    isolation_risk=0.5 if str(seg.get("road_type")) in ("industrial", "service") else 0.2,
                    activity_score=0.1,
                )
                mult = profile.cost_multiplier(
                    fallback.crime_risk, fallback.darkness_risk,
                    fallback.isolation_risk, fallback.activity_score,
                )
                mults.append(mult)
            else:
                mults.append(1.5)
                missing += 1

    print(f"\nTotal edges analysed: {len(mults)}  (missing osm data: {missing})")
    print(f"  MIN  multiplier: {min(mults):.4f}")
    print(f"  MAX  multiplier: {max(mults):.4f}")
    print(f"  MEAN multiplier: {statistics.mean(mults):.4f}")
    print(f"  STDEV:           {statistics.stdev(mults):.4f}")
    print(f"  SPREAD (max-min): {max(mults) - min(mults):.4f}")

    # Histogram (10 buckets)
    lo, hi = min(mults), max(mults)
    n_bins = 10
    bin_width = (hi - lo) / n_bins if hi > lo else 1.0
    buckets = [0] * n_bins
    for m in mults:
        idx = min(int((m - lo) / bin_width), n_bins - 1)
        buckets[idx] += 1

    print(f"\n  Histogram ({n_bins} bins from {lo:.3f} to {hi:.3f}):")
    for i, cnt in enumerate(buckets):
        edge_lo = lo + i * bin_width
        edge_hi = edge_lo + bin_width
        bar = "#" * max(1, int(cnt / max(buckets) * 40))
        print(f"    [{edge_lo:.3f}, {edge_hi:.3f})  {cnt:5d}  {bar}")

    # Component stats
    if mults_with_scores:
        crimes = [s.crime_risk for _, s in mults_with_scores]
        darks  = [s.darkness_risk for _, s in mults_with_scores]
        isos   = [s.isolation_risk for _, s in mults_with_scores]
        acts   = [s.activity_score for _, s in mults_with_scores]

        print(f"\n  Component stats (normalised, edges with precomputed scores):")
        for name, vals in [("crime_risk", crimes), ("darkness_risk", darks),
                           ("isolation_risk", isos), ("activity_score", acts)]:
            print(f"    {name:20s}: min={min(vals):.4f}  max={max(vals):.4f}  "
                  f"mean={statistics.mean(vals):.4f}  stdev={statistics.stdev(vals):.4f}")

    return slot_scores


def step3_search_pairs(slot_scores):
    """Search 10 random OD pairs 2-5 km apart and report detour % + safety gain."""
    print("\n" + "=" * 72)
    print("STEP 3: Random pair search (woman_night, 23:00, Peenya)")
    print("=" * 72)

    rc = redis.from_url(settings.REDIS_URL, decode_responses=True)
    G = get_graph(ZONE)
    dt_23 = IST.localize(datetime(2026, 10, 2, 23, 0, 0))
    ctx = TimeContext.from_datetime(dt_23, ZONE)

    # Get all graph nodes with lat/lon
    nodes = list(G.nodes(data=True))
    random.seed(42)  # reproducible

    results = []
    attempts = 0
    max_attempts = 200

    print(f"\nSearching for pairs 2-5 km apart among {len(nodes)} nodes...\n")

    while len(results) < 10 and attempts < max_attempts:
        attempts += 1
        n1_id, n1_data = random.choice(nodes)
        n2_id, n2_data = random.choice(nodes)
        if n1_id == n2_id:
            continue

        lat1, lon1 = float(n1_data["y"]), float(n1_data["x"])
        lat2, lon2 = float(n2_data["y"]), float(n2_data["x"])
        dist_m = _haversine_m(lat1, lon1, lat2, lon2)

        if not (2000 <= dist_m <= 5000):
            continue

        try:
            res = compute_routes(
                origin_lat=lat1, origin_lon=lon1,
                dest_lat=lat2, dest_lon=lon2,
                time_ctx=ctx, profile_name="woman_night",
                db_url=settings.DATABASE_URL_SYNC,
                redis_client=rc, zone=ZONE,
            )
        except (ValueError, Exception) as e:
            continue

        fastest = res["fastest"]
        safest = res["safest"]

        if fastest.distance_m < 100:
            continue

        detour_pct = (safest.distance_m - fastest.distance_m) / fastest.distance_m * 100
        safety_gain = safest.safety_score - fastest.safety_score
        routes_differ = res["routes_differ"]

        pair_info = {
            "origin": (round(lat1, 4), round(lon1, 4)),
            "dest": (round(lat2, 4), round(lon2, 4)),
            "fastest_dist": round(fastest.distance_m, 1),
            "safest_dist": round(safest.distance_m, 1),
            "fastest_safety": fastest.safety_score,
            "safest_safety": safest.safety_score,
            "detour_pct": round(detour_pct, 1),
            "safety_gain": round(safety_gain, 1),
            "routes_differ": routes_differ,
        }
        results.append(pair_info)

        status = "+" if routes_differ and detour_pct >= 8 and safety_gain >= 10 else " "
        print(f"  [{status}] Pair {len(results):2d}: "
              f"({lat1:.4f},{lon1:.4f}) -> ({lat2:.4f},{lon2:.4f})  "
              f"fast={fastest.distance_m:.0f}m  safe={safest.distance_m:.0f}m  "
              f"detour={detour_pct:+.1f}%  safety={safety_gain:+.1f}pts  "
              f"differ={routes_differ}")

    # Pick the 2 best pairs: at least 8% longer AND at least 10 pts safer
    good = [
        p for p in results
        if p["routes_differ"] and p["detour_pct"] >= 8.0 and p["safety_gain"] >= 10.0
    ]
    good.sort(key=lambda p: p["safety_gain"], reverse=True)
    best_2 = good[:2]

    print(f"\n{'=' * 72}")
    print(f"Found {len(good)} qualifying pairs (>=8% detour, >=10 pts safer)")

    if len(best_2) < 2:
        print("WARNING: Need at least 2 qualifying pairs. Falling back to top 2 by safety gain.")
        all_sorted = sorted(results, key=lambda p: p["safety_gain"], reverse=True)
        best_2 = all_sorted[:2]

    for i, p in enumerate(best_2):
        print(f"\n  FIXTURE {i+1}:")
        print(f"    Origin:       ({p['origin'][0]}, {p['origin'][1]})")
        print(f"    Destination:  ({p['dest'][0]}, {p['dest'][1]})")
        print(f"    Fastest:      {p['fastest_dist']} m, safety {p['fastest_safety']}")
        print(f"    Safest:       {p['safest_dist']} m, safety {p['safest_safety']}")
        print(f"    Detour:       {p['detour_pct']:+.1f}%")
        print(f"    Safety gain:  {p['safety_gain']:+.1f} pts")
        print(f"    Routes differ: {p['routes_differ']}")

    return best_2


def step4_print_coordinates(best_2):
    """Print exact coordinates for manual testing."""
    print(f"\n{'=' * 72}")
    print("STEP 4: Exact coordinates for manual testing")
    print("=" * 72)

    for i, p in enumerate(best_2):
        print(f"\n  Pair {i+1}:")
        print(f"    Origin:      lat={p['origin'][0]}, lon={p['origin'][1]}")
        print(f"    Destination: lat={p['dest'][0]}, lon={p['dest'][1]}")
        print(f"    Expected detour: {p['detour_pct']:+.1f}%")
        print(f"    Expected safety gain: {p['safety_gain']:+.1f} pts")


if __name__ == "__main__":
    slot_scores = step1_diagnose_multiplier()
    best_2 = step3_search_pairs(slot_scores)
    step4_print_coordinates(best_2)
