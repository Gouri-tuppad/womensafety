"""
tests/test_safety_scoring.py - Unit and integration tests for Step 3 safety scoring.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
import pytz
import pytest

from app.config import settings
from app.services.personalization import get_profile, list_profiles, ProfileWeights
from app.services.safety_scoring import (
    CRIME_RADIUS_M,
    IncidentDensityProvider,
    RiskProvider,
    RiskScores,
    SegmentFeatures,
    SegmentScorer,
    TimeContext,
    get_cached_scores,
    get_slot_scores,
    set_cached_scores,
)

IST = pytz.timezone("Asia/Kolkata")


# ── TimeContext tests ─────────────────────────────────────────────────────────

class TestTimeContext:
    def test_time_context_sunset_and_sunrise(self):
        """Astral computes sunset for Peenya; 23:00 is dark, 12:00 noon is not."""
        dt_night = IST.localize(datetime(2026, 10, 2, 23, 0, 0))
        ctx_night = TimeContext.from_datetime(dt_night, "peenya")
        assert ctx_night.hour == 23
        assert ctx_night.is_dark is True

        dt_day = IST.localize(datetime(2026, 10, 2, 12, 0, 0))
        ctx_day = TimeContext.from_datetime(dt_day, "peenya")
        assert ctx_day.hour == 12
        assert ctx_day.is_dark is False

    def test_time_context_weekend(self):
        """Saturday/Sunday are marked as weekend with 'W' suffix."""
        # 2026-10-03 is Saturday
        dt_sat = IST.localize(datetime(2026, 10, 3, 22, 0, 0))
        ctx_sat = TimeContext.from_datetime(dt_sat, "peenya")
        assert ctx_sat.is_weekend is True
        assert ctx_sat.time_slot == "22W"

        # 2026-10-05 is Monday
        dt_mon = IST.localize(datetime(2026, 10, 5, 8, 0, 0))
        ctx_mon = TimeContext.from_datetime(dt_mon, "peenya")
        assert ctx_mon.is_weekend is False
        assert ctx_mon.time_slot == "08D"

    def test_time_context_holiday(self):
        """Republic Day (Jan 26) in Karnataka is recognized as a holiday."""
        dt_hol = IST.localize(datetime(2026, 1, 26, 10, 0, 0))
        ctx_hol = TimeContext.from_datetime(dt_hol, "peenya")
        assert ctx_hol.is_holiday is True


# ── DarknessRisk & Lighting Proxies tests ─────────────────────────────────────

class TestDarknessRisk:
    def test_darkness_risk_lighting_proxies(self):
        """DarknessRisk combines lit_ratio, lamps, shops, petrol, metro, and main roads."""
        scorer = SegmentScorer()
        dt_night = IST.localize(datetime(2026, 10, 2, 23, 0, 0))
        ctx_night = TimeContext.from_datetime(dt_night, "peenya")

        # Well-lit segment: near main road, shops, petrol, metro, lamps
        well_lit = SegmentFeatures(
            osm_id=1,
            lat=13.033, lon=77.533,
            road_type="primary",
            length_m=100.0,
            lit_ratio=1.0,
            lamp_count=8,
            shop_count_150m=3,
            petrol_count_200m=1,
            any_metro_300m=True,
            dist_main_m=0.0,
        )
        scores_lit = scorer.score(well_lit, ctx_night)

        # Unlit isolated industrial alley
        unlit = SegmentFeatures(
            osm_id=2,
            lat=13.015, lon=77.490,
            road_type="industrial",
            length_m=100.0,
            lit_ratio=0.0,
            lamp_count=0,
            shop_count_150m=0,
            petrol_count_200m=0,
            any_metro_300m=False,
            dist_main_m=1000.0,
        )
        scores_unlit = scorer.score(unlit, ctx_night)

        # Well-lit street must have much lower darkness risk than unlit alley
        assert scores_lit.darkness_risk < scores_unlit.darkness_risk
        assert scores_lit.darkness_risk == 0.0
        assert scores_unlit.darkness_risk >= 0.90

    def test_darkness_day_vs_night(self):
        """DarknessRisk is substantially attenuated during the day."""
        scorer = SegmentScorer()
        feat = SegmentFeatures(
            osm_id=3,
            lat=13.020, lon=77.500,
            road_type="residential",
            length_m=80.0,
            lit_ratio=0.0,
            lamp_count=0,
            dist_main_m=400.0,
        )
        ctx_day = TimeContext.from_datetime(IST.localize(datetime(2026, 10, 2, 12, 0, 0)), "peenya")
        ctx_night = TimeContext.from_datetime(IST.localize(datetime(2026, 10, 2, 23, 0, 0)), "peenya")

        s_day = scorer.score(feat, ctx_day)
        s_night = scorer.score(feat, ctx_night)

        assert s_day.darkness_risk < s_night.darkness_risk
        assert s_day.darkness_risk <= 0.15
        assert s_night.darkness_risk >= 0.80


# ── IsolationRisk tests ───────────────────────────────────────────────────────

class TestIsolationRisk:
    def test_isolation_road_types(self):
        """Industrial and service lanes have higher isolation risk than main roads."""
        scorer = SegmentScorer()
        ctx = TimeContext.from_datetime(IST.localize(datetime(2026, 10, 2, 23, 0, 0)), "peenya")

        industrial = SegmentFeatures(
            osm_id=10, lat=13.02, lon=77.50, road_type="industrial", length_m=50.0, lit_ratio=0.0, lamp_count=0
        )
        primary = SegmentFeatures(
            osm_id=11, lat=13.02, lon=77.50, road_type="primary", length_m=50.0, lit_ratio=0.0, lamp_count=0
        )

        assert scorer.score(industrial, ctx).isolation_risk > scorer.score(primary, ctx).isolation_risk

    def test_isolation_crowd_and_pois(self):
        """Higher crowd and nearby shops reduce isolation risk."""
        scorer = SegmentScorer()
        ctx = TimeContext.from_datetime(IST.localize(datetime(2026, 10, 2, 23, 0, 0)), "peenya")

        busy = SegmentFeatures(
            osm_id=20, lat=13.02, lon=77.50, road_type="residential", length_m=50.0,
            lit_ratio=0.0, lamp_count=5, shop_count_150m=4, crowd_count=50, dist_main_m=50.0
        )
        desolate = SegmentFeatures(
            osm_id=21, lat=13.02, lon=77.50, road_type="residential", length_m=50.0,
            lit_ratio=0.0, lamp_count=0, shop_count_150m=0, crowd_count=0, dist_main_m=800.0
        )

        assert scorer.score(busy, ctx).isolation_risk < scorer.score(desolate, ctx).isolation_risk


# ── ActivityScore tests ───────────────────────────────────────────────────────

class TestActivityScore:
    def test_activity_scaling(self):
        """ActivityScore scales from 0 to 1 with crowd count and clamps at 1.0."""
        scorer = SegmentScorer()
        ctx = TimeContext.from_datetime(IST.localize(datetime(2026, 10, 2, 12, 0, 0)), "peenya")

        zero = SegmentFeatures(osm_id=30, lat=13.0, lon=77.5, road_type="service", length_m=50.0, lit_ratio=0.0, lamp_count=0, crowd_count=0)
        moderate = SegmentFeatures(osm_id=31, lat=13.0, lon=77.5, road_type="service", length_m=50.0, lit_ratio=0.0, lamp_count=0, crowd_count=40)
        peak = SegmentFeatures(osm_id=32, lat=13.0, lon=77.5, road_type="service", length_m=50.0, lit_ratio=0.0, lamp_count=0, crowd_count=120)

        assert scorer.score(zero, ctx).activity_score == 0.0
        assert 0.4 <= scorer.score(moderate, ctx).activity_score <= 0.6
        assert scorer.score(peak, ctx).activity_score == 1.0


# ── IncidentDensityProvider & RiskProvider Protocol tests ────────────────────

class TestIncidentDensityProvider:
    def test_time_decay_and_severity(self):
        """Recent severe incidents generate higher CrimeRisk than older mild ones."""
        provider = IncidentDensityProvider()
        ctx = TimeContext.from_datetime(IST.localize(datetime(2026, 10, 2, 23, 0, 0)), "peenya")

        now = datetime.now(timezone.utc)
        recent_severe = [{
            "severity": 5,
            "occurred_at": now,
            "hour": 23,
        }]
        old_mild = [{
            "severity": 1,
            "occurred_at": now.replace(year=now.year - 1),
            "hour": 10,
        }]

        r_recent = provider.crime_risk(13.02, 77.50, CRIME_RADIUS_M, ctx, recent_severe)
        r_old = provider.crime_risk(13.02, 77.50, CRIME_RADIUS_M, ctx, old_mild)

        assert r_recent > r_old
        assert r_recent > 0.0

    def test_custom_risk_provider_protocol(self):
        """Verify custom RiskProvider satisfies Protocol and plugs into SegmentScorer."""
        class MockMLProvider:
            def crime_risk(self, lat, lon, radius_m, time_ctx, incident_rows):
                return 0.42

        mock = MockMLProvider()
        assert isinstance(mock, RiskProvider)

        scorer = SegmentScorer(risk_provider=mock)
        ctx = TimeContext.from_datetime(IST.localize(datetime(2026, 10, 2, 23, 0, 0)), "peenya")
        feat = SegmentFeatures(osm_id=99, lat=13.0, lon=77.5, road_type="residential", length_m=50.0, lit_ratio=0.0, lamp_count=0)

        scores = scorer.score(feat, ctx)
        assert scores.crime_risk == 0.42


# ── Personalization Profile tests ─────────────────────────────────────────────

class TestPersonalizationProfiles:
    def test_all_profiles_registered(self):
        profiles = list_profiles()
        assert "adult_day" in profiles
        assert "woman_night" in profiles
        assert "child" in profiles

    def test_profile_weights_distinctions(self):
        """woman_night has higher beta/gamma than adult_day; child has highest lighting & isolation weights."""
        adult = get_profile("adult_day")
        woman = get_profile("woman_night")
        child = get_profile("child")

        assert woman.beta >= adult.beta
        assert woman.gamma >= adult.gamma
        assert child.beta >= woman.beta
        assert child.gamma >= woman.gamma
        assert child.delta >= woman.delta

    def test_cost_multiplier_clamp_at_0_5(self):
        """Cost multiplier is clamped to minimum 0.5 even with high activity and zero risk."""
        adult = get_profile("adult_day")
        mult = adult.cost_multiplier(crime_risk=0.0, darkness_risk=0.0, isolation_risk=0.0, activity_score=1.0)
        assert mult >= 0.5
        assert mult == max(0.5, 1.0 - adult.delta * 1.0)
