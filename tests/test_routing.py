"""
tests/test_routing.py - Routing algorithm, demo route fixtures, acceptance tests, and performance validation.
"""

from __future__ import annotations

import time
from datetime import datetime
import pytz
import pytest
import redis

from app.config import settings
from app.services.routing_service import compute_heatmap, compute_routes, get_graph
from app.services.safety_scoring import TimeContext, precompute_zone_weights

IST = pytz.timezone("Asia/Kolkata")


# ── Fixtures for Demo Routes ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def peenya_metro_to_2nd_stage():
    """Route: Peenya Metro to Peenya 2nd Stage (NTTF / 12th Cross)."""
    return {
        "name": "Peenya Metro to Peenya 2nd Stage",
        "origin": (13.0330, 77.5332),
        "destination": (13.0220, 77.5034),
    }


@pytest.fixture(scope="session")
def industrial_cross_route():
    """Route: Across Industrial Area: Dasarahalli to Peenya 2nd Stage West."""
    return {
        "name": "Dasarahalli to Peenya 2nd Stage West (Across Industrial Area)",
        "origin": (13.0432, 77.5125),
        "destination": (13.0180, 77.4950),
    }


@pytest.fixture(scope="session")
def bagalagunte_to_peenya_1st_stage():
    """Demo Route 1: Bagalagunte to Peenya 1st Stage South-East across the industrial core."""
    return {
        "name": "Bagalagunte to Peenya 1st Stage",
        "origin": (13.0450, 77.5050),
        "destination": (13.0150, 77.5300),
    }


@pytest.fixture(scope="session")
def dasarahalli_to_peenya_1st_stage_south():
    """Demo Route 2: Dasarahalli Metro to Peenya 1st Stage South across the industrial area."""
    return {
        "name": "Dasarahalli Metro to Peenya 1st Stage South",
        "origin": (13.0440, 77.5130),
        "destination": (13.0120, 77.5320),
    }


@pytest.fixture(scope="session")
def redis_client():
    return redis.from_url(settings.REDIS_URL, decode_responses=True)


@pytest.fixture(scope="session", autouse=True)
def warm_peenya_slots(redis_client):
    """Ensure Peenya edge weights are precomputed for test hours 18 and 23."""
    precompute_zone_weights("peenya", 23, False, settings.DATABASE_URL_SYNC, settings.REDIS_URL)
    precompute_zone_weights("peenya", 18, False, settings.DATABASE_URL_SYNC, settings.REDIS_URL)
    # Also ensure graph is loaded
    get_graph("peenya")


# ── Acceptance Tests ──────────────────────────────────────────────────────────

class TestRoutingAcceptance:
    def test_demo_route_1_bagalagunte_to_1st_stage(self, bagalagunte_to_peenya_1st_stage, redis_client):
        """
        Demo Route 1: Bagalagunte to Peenya 1st Stage.
        At 11 PM: safest route differs clearly from fastest, is >= 8% longer, and >= 10 pts safer.
        """
        dt_23 = IST.localize(datetime(2026, 10, 2, 23, 0, 0))
        ctx_23 = TimeContext.from_datetime(dt_23, "peenya")
        orig = bagalagunte_to_peenya_1st_stage["origin"]
        dest = bagalagunte_to_peenya_1st_stage["destination"]

        res = compute_routes(
            origin_lat=orig[0], origin_lon=orig[1], dest_lat=dest[0], dest_lon=dest[1],
            time_ctx=ctx_23, profile_name="woman_night", db_url=settings.DATABASE_URL_SYNC,
            redis_client=redis_client, zone="peenya",
        )

        assert res["routes_differ"] is True
        assert res["comparison"].distance_pct_longer >= 8.0
        assert res["comparison"].distance_pct_longer <= 30.0
        assert res["comparison"].safety_score_gain >= 10.0
        assert res["safest"].safety_score - res["fastest"].safety_score >= 10.0
        assert res["safest"].distance_m <= 1.30 * res["fastest"].distance_m

    def test_demo_route_2_dasarahalli_to_1st_stage_south(self, dasarahalli_to_peenya_1st_stage_south, redis_client):
        """
        Demo Route 2: Dasarahalli Metro to Peenya 1st Stage South.
        At 11 PM: safest route differs clearly from fastest, is >= 8% longer, and >= 10 pts safer.
        """
        dt_23 = IST.localize(datetime(2026, 10, 2, 23, 0, 0))
        ctx_23 = TimeContext.from_datetime(dt_23, "peenya")
        orig = dasarahalli_to_peenya_1st_stage_south["origin"]
        dest = dasarahalli_to_peenya_1st_stage_south["destination"]

        res = compute_routes(
            origin_lat=orig[0], origin_lon=orig[1], dest_lat=dest[0], dest_lon=dest[1],
            time_ctx=ctx_23, profile_name="woman_night", db_url=settings.DATABASE_URL_SYNC,
            redis_client=redis_client, zone="peenya",
        )

        assert res["routes_differ"] is True
        assert res["comparison"].distance_pct_longer >= 8.0
        assert res["comparison"].distance_pct_longer <= 30.0
        assert res["comparison"].safety_score_gain >= 10.0
        assert res["safest"].safety_score - res["fastest"].safety_score >= 10.0
        assert res["safest"].distance_m <= 1.30 * res["fastest"].distance_m

    def test_safest_differs_from_fastest_at_11pm(self, peenya_metro_to_2nd_stage, redis_client):
        """
        Acceptance requirement: For seeded Peenya, the safest route differs
        from the fastest route at 11 PM (23:00) for Peenya Metro to Peenya 2nd Stage.
        """
        dt_23 = IST.localize(datetime(2026, 10, 2, 23, 0, 0))
        ctx_23 = TimeContext.from_datetime(dt_23, "peenya")

        orig = peenya_metro_to_2nd_stage["origin"]
        dest = peenya_metro_to_2nd_stage["destination"]

        res = compute_routes(
            origin_lat=orig[0],
            origin_lon=orig[1],
            dest_lat=dest[0],
            dest_lon=dest[1],
            time_ctx=ctx_23,
            profile_name="woman_night",
            db_url=settings.DATABASE_URL_SYNC,
            redis_client=redis_client,
            zone="peenya",
        )

        assert res["routes_differ"] is True
        assert res["safest"].safety_score >= res["fastest"].safety_score
        assert res["comparison"].summary != "same distance"
        assert len(res["safest"].polyline) > 0

    def test_safest_6pm_differs_from_11pm(self, peenya_metro_to_2nd_stage, redis_client):
        """
        Acceptance requirement: The 6 PM (18:00) safest route differs from
        the 11 PM (23:00) safest route for at least one demo pair.
        """
        orig = peenya_metro_to_2nd_stage["origin"]
        dest = peenya_metro_to_2nd_stage["destination"]

        dt_23 = IST.localize(datetime(2026, 10, 2, 23, 0, 0))
        ctx_23 = TimeContext.from_datetime(dt_23, "peenya")
        res_23 = compute_routes(
            origin_lat=orig[0], origin_lon=orig[1], dest_lat=dest[0], dest_lon=dest[1],
            time_ctx=ctx_23, profile_name="woman_night", db_url=settings.DATABASE_URL_SYNC,
            redis_client=redis_client, zone="peenya",
        )

        dt_18 = IST.localize(datetime(2026, 10, 2, 18, 0, 0))
        ctx_18 = TimeContext.from_datetime(dt_18, "peenya")
        res_18 = compute_routes(
            origin_lat=orig[0], origin_lon=orig[1], dest_lat=dest[0], dest_lon=dest[1],
            time_ctx=ctx_18, profile_name="woman_night", db_url=settings.DATABASE_URL_SYNC,
            redis_client=redis_client, zone="peenya",
        )

        # 6 PM path differs from 11 PM path
        assert res_18["safest"].polyline != res_23["safest"].polyline

    def test_industrial_cross_route_differs_at_11pm(self, industrial_cross_route, redis_client):
        """Second demo route: Dasarahalli to Peenya 2nd Stage West across industrial area."""
        dt_23 = IST.localize(datetime(2026, 10, 2, 23, 0, 0))
        ctx_23 = TimeContext.from_datetime(dt_23, "peenya")

        orig = industrial_cross_route["origin"]
        dest = industrial_cross_route["destination"]

        res = compute_routes(
            origin_lat=orig[0],
            origin_lon=orig[1],
            dest_lat=dest[0],
            dest_lon=dest[1],
            time_ctx=ctx_23,
            profile_name="woman_night",
            db_url=settings.DATABASE_URL_SYNC,
            redis_client=redis_client,
            zone="peenya",
        )

        assert res["routes_differ"] is True
        assert res["safest"].safety_score >= res["fastest"].safety_score


# ── Response Schema & Quality Tests ───────────────────────────────────────────

class TestRouteFeatures:
    def test_colour_coded_segments_and_havens(self, peenya_metro_to_2nd_stage, redis_client):
        """Route returns colour-coded segments (green/yellow/red) and safe havens within 300m."""
        dt = IST.localize(datetime(2026, 10, 2, 23, 0, 0))
        ctx = TimeContext.from_datetime(dt, "peenya")
        orig = peenya_metro_to_2nd_stage["origin"]
        dest = peenya_metro_to_2nd_stage["destination"]

        res = compute_routes(
            origin_lat=orig[0], origin_lon=orig[1], dest_lat=dest[0], dest_lon=dest[1],
            time_ctx=ctx, profile_name="woman_night", db_url=settings.DATABASE_URL_SYNC,
            redis_client=redis_client, zone="peenya",
        )

        safest = res["safest"]
        assert len(safest.segments) > 0
        valid_risks = {"green", "yellow", "red"}
        for s in safest.segments:
            assert s.risk.value in valid_risks
            assert 0.0 <= s.safety_score <= 1.0

        # Safe havens near safest route
        havens = res["safe_havens_near_safest"]
        assert len(havens) > 0
        for h in havens:
            assert h.dist_from_route_m <= 300.0

        # Comparison metrics
        comp = res["comparison"]
        assert hasattr(comp, "distance_pct_longer")
        assert hasattr(comp, "safety_score_gain")
        assert hasattr(comp, "summary")

    def test_safe_havens_deduped_and_ranked(self, bagalagunte_to_peenya_1st_stage, redis_client):
        """Safe havens are deduped by (name, category) and ranked: police > hospital > metro > bus_stop."""
        dt = IST.localize(datetime(2026, 10, 2, 23, 0, 0))
        ctx = TimeContext.from_datetime(dt, "peenya")
        orig = bagalagunte_to_peenya_1st_stage["origin"]
        dest = bagalagunte_to_peenya_1st_stage["destination"]

        res = compute_routes(
            origin_lat=orig[0], origin_lon=orig[1], dest_lat=dest[0], dest_lon=dest[1],
            time_ctx=ctx, profile_name="woman_night", db_url=settings.DATABASE_URL_SYNC,
            redis_client=redis_client, zone="peenya",
        )

        havens = res["safe_havens_near_safest"]
        assert len(havens) > 0
        # Check deduping: each (name.lower(), category.lower()) is unique
        keys = [(h.name.lower(), h.category.lower()) for h in havens]
        assert len(keys) == len(set(keys)), "Safe havens contain duplicates"

        # Check ranking: police/hospital/metro must precede bus stops
        def cat_rank(category: str) -> int:
            cat = category.lower()
            if "police" in cat:
                return 0
            if "hospital" in cat or "clinic" in cat or "pharmacy" in cat or "health" in cat:
                return 1
            if "metro" in cat or "subway" in cat or "train" in cat:
                return 2
            return 3

        ranks = [cat_rank(h.category) for h in havens]
        assert ranks == sorted(ranks), f"Havens are not ordered by priority: {ranks}"

    def test_detour_cap_under_30_percent(self, bagalagunte_to_peenya_1st_stage, redis_client):
        """Safest route distance is capped at <= 30% detour above fastest route."""
        dt = IST.localize(datetime(2026, 10, 2, 23, 0, 0))
        ctx = TimeContext.from_datetime(dt, "peenya")
        orig = bagalagunte_to_peenya_1st_stage["origin"]
        dest = bagalagunte_to_peenya_1st_stage["destination"]

        res = compute_routes(
            origin_lat=orig[0], origin_lon=orig[1], dest_lat=dest[0], dest_lon=dest[1],
            time_ctx=ctx, profile_name="woman_night", db_url=settings.DATABASE_URL_SYNC,
            redis_client=redis_client, zone="peenya",
        )
        assert res["safest"].distance_m <= 1.30 * res["fastest"].distance_m

    def test_comparison_summary_lighting_omitted_when_sparse(self, redis_client):
        """When lit ratio is < 0.20, summary does not lead with or mention lit ratio."""
        dt = IST.localize(datetime(2026, 10, 2, 23, 0, 0))
        ctx = TimeContext.from_datetime(dt, "peenya")
        orig = (13.0330, 77.5200)
        dest = (13.0470, 77.5190)

        res = compute_routes(
            origin_lat=orig[0], origin_lon=orig[1], dest_lat=dest[0], dest_lon=dest[1],
            time_ctx=ctx, profile_name="woman_night", db_url=settings.DATABASE_URL_SYNC,
            redis_client=redis_client, zone="peenya",
        )
        summary = res["comparison"].summary
        assert "more lit" not in summary
        assert "better lit" not in summary
        assert "police station nearby" in summary


# ── Performance Tests ─────────────────────────────────────────────────────────

class TestRoutingPerformance:
    def test_routing_latency_under_3_seconds(self, peenya_metro_to_2nd_stage, redis_client):
        """Route computation for Peenya completes well under 3 seconds."""
        dt = IST.localize(datetime(2026, 10, 2, 23, 0, 0))
        ctx = TimeContext.from_datetime(dt, "peenya")
        orig = peenya_metro_to_2nd_stage["origin"]
        dest = peenya_metro_to_2nd_stage["destination"]

        t0 = time.time()
        res = compute_routes(
            origin_lat=orig[0], origin_lon=orig[1], dest_lat=dest[0], dest_lon=dest[1],
            time_ctx=ctx, profile_name="woman_night", db_url=settings.DATABASE_URL_SYNC,
            redis_client=redis_client, zone="peenya",
        )
        elapsed = time.time() - t0

        assert elapsed < 3.0, f"Route took {elapsed:.3f}s, expected < 3.0s"
        assert res["fastest"].distance_m > 0


# ── Heatmap Tests ─────────────────────────────────────────────────────────────

class TestHeatmap:
    def test_compute_heatmap_h3_grid(self, redis_client):
        """Heatmap returns H3 cells with composite risk scores for time slider."""
        cells = compute_heatmap("peenya", 23, settings.DATABASE_URL_SYNC, redis_client)
        assert len(cells) > 0
        sample = cells[0]
        assert "h3_index" in sample
        assert "risk_score" in sample
        assert "lat" in sample and "lon" in sample
        assert 0.0 <= sample["risk_score"] <= 1.0


# ── API Endpoint Tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRoutingAPIEndpoints:
    async def test_post_route_endpoint(self, client):
        """POST /route returns 200 with full RouteResponse schema."""
        payload = {
            "origin": {"lat": 13.0330, "lon": 77.5332},
            "destination": {"lat": 13.0220, "lon": 77.5034},
            "depart_time": "2026-10-02T23:00:00+05:30",
            "profile": "woman_night"
        }
        res = await client.post("/route", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["zone"] == "peenya"
        assert data["routes_differ"] is True
        assert data["fastest"]["distance_m"] > 0
        assert data["safest"]["distance_m"] > 0

    async def test_get_safety_heatmap_endpoint(self, client):
        """GET /safety/heatmap returns 200 with H3 grid."""
        res = await client.get("/safety/heatmap?zone=peenya&hour=23")
        assert res.status_code == 200
        data = res.json()
        assert data["zone"] == "peenya"
        assert data["hour"] == 23
        assert data["cell_count"] > 0
        assert len(data["cells"]) > 0

    async def test_route_out_of_bounds_returns_400(self, client):
        """Coordinates outside active zone bounding box return 400."""
        payload = {
            "origin": {"lat": 12.0, "lon": 77.0},  # outside Peenya
            "destination": {"lat": 13.0220, "lon": 77.5034},
            "depart_time": "2026-10-02T23:00:00+05:30",
            "profile": "woman_night"
        }
        res = await client.post("/route", json=payload)
        assert res.status_code == 400
