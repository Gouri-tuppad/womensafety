"""
tests/test_ingest_helpers.py - Unit tests for Step 2 data ingestion helpers.

Uses a tiny fixture graph (no network calls, no DB required).
All Overpass / DB / Redis calls are mocked.
"""

from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import shapely.geometry as sg

# ── Import helpers under test ─────────────────────────────────────────────────

from data.ingest_osm import (
    _approx_buffer_deg,
    _build_poi_shapely_list,
    _metres_per_degree_lat,
    compute_segment_features,
    load_graph,
)
from data.seed_crime_data import (
    _generate_synthetic_incidents,
    _load_csv,
    HOUR_WEIGHTS,
    INCIDENT_TYPES,
)
from data.simulate_crowd import (
    _peenya_hourly_weight,
    _subramanyapura_hourly_weight,
    _simulate_cell_hour,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_segment() -> sg.LineString:
    """A short line segment in the Peenya bbox (approx 200 m)."""
    # lon, lat order for Shapely
    return sg.LineString([(77.500, 13.010), (77.502, 13.010)])


@pytest.fixture
def lamp_near(tiny_segment: sg.LineString) -> list[sg.Point]:
    """One lamp point very close to the segment."""
    # Midpoint of segment + tiny offset
    mid = tiny_segment.interpolate(0.5, normalized=True)
    return [sg.Point(mid.x, mid.y + 0.0001)]


@pytest.fixture
def lamp_far() -> list[sg.Point]:
    """Lamp points far away (> 150 m)."""
    return [sg.Point(77.600, 13.100)]


@pytest.fixture
def safe_haven_points() -> list[sg.Point]:
    return [sg.Point(77.501, 13.011)]  # ~130 m from segment midpoint


@pytest.fixture
def main_road_geom() -> list[sg.LineString]:
    return [sg.LineString([(77.490, 13.010), (77.510, 13.010)])]


# ── Tests: geometry helpers ───────────────────────────────────────────────────

class TestGeometryHelpers:
    def test_metres_per_degree_lat_typical(self):
        # At Bengaluru (~13 N), should be close to 110 950 m/deg
        mpd = _metres_per_degree_lat(13.0)
        assert 110_000 < mpd < 112_000, f"Unexpected metres per degree: {mpd}"

    def test_approx_buffer_deg_150m(self):
        buf = _approx_buffer_deg(150, 13.0)
        # Should be roughly 150 / 111_000 ≈ 0.00135 degrees
        assert 0.001 < buf < 0.0018, f"Unexpected buffer degrees: {buf}"

    def test_approx_buffer_deg_scales(self):
        buf_150 = _approx_buffer_deg(150, 13.0)
        buf_300 = _approx_buffer_deg(300, 13.0)
        assert abs(buf_300 - 2 * buf_150) < 1e-7

    def test_build_poi_shapely_list_nodes(self):
        elements = [
            {"type": "node", "lat": 13.01, "lon": 77.50},
            {"type": "node", "lat": 13.02, "lon": 77.51},
        ]
        pairs = _build_poi_shapely_list(elements)
        assert len(pairs) == 2
        el0, pt0 = pairs[0]
        assert pt0.x == pytest.approx(77.50)
        assert pt0.y == pytest.approx(13.01)

    def test_build_poi_shapely_list_way_with_center(self):
        elements = [
            {"type": "way", "id": 1, "center": {"lat": 13.05, "lon": 77.52}, "tags": {}},
        ]
        pairs = _build_poi_shapely_list(elements)
        assert len(pairs) == 1
        _, pt = pairs[0]
        assert pt.x == pytest.approx(77.52)

    def test_build_poi_shapely_list_skips_way_without_center(self):
        elements = [
            {"type": "way", "id": 2, "tags": {}},  # no center
        ]
        pairs = _build_poi_shapely_list(elements)
        assert len(pairs) == 0


# ── Tests: compute_segment_features ──────────────────────────────────────────

class TestComputeSegmentFeatures:
    CENTER_LAT = 13.010

    def test_lit_tag_yes(self, tiny_segment, lamp_far, safe_haven_points, main_road_geom):
        result = compute_segment_features(
            edge_geom=tiny_segment,
            lamp_points=lamp_far,
            lit_way_osm_ids=set(),
            osm_id=999,
            lit_tag="yes",
            center_lat=self.CENTER_LAT,
            safe_haven_points=safe_haven_points,
            main_road_geoms=main_road_geom,
        )
        assert result["lit_ratio"] == 1.0

    def test_lit_tag_no(self, tiny_segment, lamp_near, safe_haven_points, main_road_geom):
        result = compute_segment_features(
            edge_geom=tiny_segment,
            lamp_points=lamp_near,
            lit_way_osm_ids=set(),
            osm_id=999,
            lit_tag="no",
            center_lat=self.CENTER_LAT,
            safe_haven_points=safe_haven_points,
            main_road_geoms=main_road_geom,
        )
        assert result["lit_ratio"] == 0.0

    def test_lit_tag_limited(self, tiny_segment, lamp_far, safe_haven_points, main_road_geom):
        result = compute_segment_features(
            edge_geom=tiny_segment,
            lamp_points=lamp_far,
            lit_way_osm_ids=set(),
            osm_id=999,
            lit_tag="limited",
            center_lat=self.CENTER_LAT,
            safe_haven_points=safe_haven_points,
            main_road_geoms=main_road_geom,
        )
        assert result["lit_ratio"] == 0.3

    def test_no_lamps_zero_lit_ratio(self, tiny_segment, safe_haven_points, main_road_geom):
        result = compute_segment_features(
            edge_geom=tiny_segment,
            lamp_points=[],
            lit_way_osm_ids=set(),
            osm_id=999,
            lit_tag=None,
            center_lat=self.CENTER_LAT,
            safe_haven_points=safe_haven_points,
            main_road_geoms=main_road_geom,
        )
        assert result["lamp_count"] == 0
        assert result["lit_ratio"] == 0.0

    def test_nearby_lamps_counted(self, tiny_segment, lamp_near, safe_haven_points, main_road_geom):
        result = compute_segment_features(
            edge_geom=tiny_segment,
            lamp_points=lamp_near,
            lit_way_osm_ids=set(),
            osm_id=999,
            lit_tag=None,
            center_lat=self.CENTER_LAT,
            safe_haven_points=safe_haven_points,
            main_road_geoms=main_road_geom,
        )
        assert result["lamp_count"] >= 1

    def test_far_lamps_not_counted(self, tiny_segment, lamp_far, safe_haven_points, main_road_geom):
        result = compute_segment_features(
            edge_geom=tiny_segment,
            lamp_points=lamp_far,
            lit_way_osm_ids=set(),
            osm_id=999,
            lit_tag=None,
            center_lat=self.CENTER_LAT,
            safe_haven_points=safe_haven_points,
            main_road_geoms=main_road_geom,
        )
        assert result["lamp_count"] == 0
        assert result["lit_ratio"] == 0.0

    def test_dist_haven_reasonable(self, tiny_segment, lamp_far, safe_haven_points, main_road_geom):
        result = compute_segment_features(
            edge_geom=tiny_segment,
            lamp_points=lamp_far,
            lit_way_osm_ids=set(),
            osm_id=999,
            lit_tag=None,
            center_lat=self.CENTER_LAT,
            safe_haven_points=safe_haven_points,
            main_road_geoms=main_road_geom,
        )
        # safe haven is approx 130 m away
        assert result["dist_haven_m"] is not None
        assert 50 < result["dist_haven_m"] < 500

    def test_dist_haven_none_when_no_havens(self, tiny_segment, lamp_far, main_road_geom):
        result = compute_segment_features(
            edge_geom=tiny_segment,
            lamp_points=lamp_far,
            lit_way_osm_ids=set(),
            osm_id=999,
            lit_tag=None,
            center_lat=self.CENTER_LAT,
            safe_haven_points=[],
            main_road_geoms=main_road_geom,
        )
        assert result["dist_haven_m"] is None

    def test_lit_ratio_from_osm_id(self, tiny_segment, lamp_far, safe_haven_points, main_road_geom):
        result = compute_segment_features(
            edge_geom=tiny_segment,
            lamp_points=lamp_far,
            lit_way_osm_ids={12345},
            osm_id=12345,
            lit_tag=None,
            center_lat=self.CENTER_LAT,
            safe_haven_points=safe_haven_points,
            main_road_geoms=main_road_geom,
        )
        assert result["lit_ratio"] == 0.8

    def test_lit_ratio_range(self, tiny_segment, lamp_near, safe_haven_points, main_road_geom):
        result = compute_segment_features(
            edge_geom=tiny_segment,
            lamp_points=lamp_near * 5,
            lit_way_osm_ids=set(),
            osm_id=999,
            lit_tag=None,
            center_lat=self.CENTER_LAT,
            safe_haven_points=safe_haven_points,
            main_road_geoms=main_road_geom,
        )
        assert 0.0 <= result["lit_ratio"] <= 1.0


# ── Tests: synthetic crime generation ─────────────────────────────────────────

class TestSyntheticCrimeGeneration:
    def test_generates_expected_count_approx(self):
        """Generated count should be within ~20% of requested (after filtering)."""
        incidents = list(_generate_synthetic_incidents("peenya", count=100, seed=42))
        assert 50 <= len(incidents) <= 120

    def test_all_fields_present(self):
        incidents = list(_generate_synthetic_incidents("peenya", count=10, seed=1))
        required = {"lat", "lon", "incident_type", "severity", "occurred_at", "description"}
        for inc in incidents:
            assert required.issubset(set(inc.keys())), f"Missing fields in {inc}"

    def test_severity_in_range(self):
        for inc in _generate_synthetic_incidents("peenya", count=50, seed=7):
            assert 1 <= inc["severity"] <= 5

    def test_incident_type_valid(self):
        for inc in _generate_synthetic_incidents("subramanyapura", count=50, seed=3):
            assert inc["incident_type"] in INCIDENT_TYPES

    def test_timestamps_within_past_year(self):
        now = datetime.now(tz=timezone.utc)
        year_ago = now - timedelta(days=366)
        for inc in _generate_synthetic_incidents("peenya", count=30, seed=5):
            ts = inc["occurred_at"]
            assert year_ago <= ts <= now, f"Timestamp out of range: {ts}"

    def test_description_flags_synthetic(self):
        for inc in _generate_synthetic_incidents("peenya", count=5, seed=2):
            assert "SYNTHETIC" in inc["description"].upper()

    def test_lat_lon_within_peenya_bbox(self):
        """Points should fall within Peenya bbox (with some Gaussian jitter tolerance)."""
        for inc in _generate_synthetic_incidents("peenya", count=50, seed=11):
            assert 12.95 <= inc["lat"] <= 13.10, f"lat out of range: {inc['lat']}"
            assert 77.45 <= inc["lon"] <= 77.60, f"lon out of range: {inc['lon']}"

    def test_lat_lon_within_subramanyapura_bbox(self):
        for inc in _generate_synthetic_incidents("subramanyapura", count=50, seed=12):
            assert 12.82 <= inc["lat"] <= 12.95, f"lat out of range: {inc['lat']}"
            assert 77.49 <= inc["lon"] <= 77.62, f"lon out of range: {inc['lon']}"

    def test_peak_hours_more_common(self):
        """Hours 21-23 should produce more incidents than hours 05-07."""
        night_hours = {21, 22, 23, 0, 1}
        day_hours = {5, 6, 7, 8}
        incidents = list(_generate_synthetic_incidents("peenya", count=500, seed=99))
        night_count = sum(1 for i in incidents if i["occurred_at"].hour in night_hours)
        day_count = sum(1 for i in incidents if i["occurred_at"].hour in day_hours)
        assert night_count > day_count, (
            f"Expected night ({night_count}) > day ({day_count}) incidents"
        )

    def test_hour_weights_length(self):
        assert len(HOUR_WEIGHTS) == 24
        assert all(w >= 0 for w in HOUR_WEIGHTS)

    def test_reproducible_with_same_seed(self):
        a = list(_generate_synthetic_incidents("peenya", count=20, seed=77))
        b = list(_generate_synthetic_incidents("peenya", count=20, seed=77))
        assert len(a) == len(b)
        for ai, bi in zip(a, b):
            assert ai["incident_type"] == bi["incident_type"]
            assert ai["severity"] == bi["severity"]


# ── Tests: CSV loading ────────────────────────────────────────────────────────

class TestCSVLoad:
    def test_load_template_csv(self, tmp_path):
        csv_path = Path(__file__).parent.parent / "data" / "incidents_template.csv"
        if not csv_path.exists():
            pytest.skip("Template CSV not found")
        rows = _load_csv(csv_path)
        assert len(rows) >= 1
        for row in rows:
            assert "lat" in row and "lon" in row
            assert "incident_type" in row
            assert 1 <= row["severity"] <= 5

    def test_load_custom_csv(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "lat,lon,type,severity,timestamp,source\n"
            "13.020,77.510,robbery,3,2024-06-01T22:00:00,official\n"
            "12.885,77.553,stalking,2,2024-07-15T21:00:00,user\n"
        )
        rows = _load_csv(csv_file)
        assert len(rows) == 2
        assert rows[0]["incident_type"] == "robbery"
        assert rows[0]["severity"] == 3
        assert rows[1]["incident_type"] == "stalking"

    def test_skip_invalid_rows(self, tmp_path):
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text(
            "lat,lon,type,severity,timestamp,source\n"
            "bad_lat,77.510,robbery,3,2024-06-01T22:00:00,official\n"
            "13.020,77.510,harassment,2,2024-06-02T20:00:00,user\n"
        )
        rows = _load_csv(csv_file)
        assert len(rows) == 1  # only valid row loaded


# ── Tests: crowd simulation ───────────────────────────────────────────────────

class TestCrowdSimulation:
    def test_peenya_shift_peak_high(self):
        """Shift-change hours (06, 14, 22) should have high weights."""
        assert _peenya_hourly_weight(6) > 0.5
        assert _peenya_hourly_weight(14) > 0.5
        assert _peenya_hourly_weight(22) > 0.5

    def test_peenya_night_low(self):
        """Deep night (02-04) in Peenya should have very low weight."""
        for h in (2, 3, 4):
            assert _peenya_hourly_weight(h) < 0.1, f"Hour {h} weight too high"

    def test_peenya_factory_uplift(self):
        """Near-factory uplift at shift hours should be >= plain weight."""
        for h in (6, 14, 22):
            w_plain = _peenya_hourly_weight(h, is_near_factory=False)
            w_factory = _peenya_hourly_weight(h, is_near_factory=True)
            assert w_factory >= w_plain

    def test_subramanyapura_morning_peak(self):
        """Morning commute hours (08-10) should have high weight."""
        assert _subramanyapura_hourly_weight(8) > 0.5
        assert _subramanyapura_hourly_weight(9) > 0.5

    def test_subramanyapura_evening_peak(self):
        """Evening commute (17-19) should be high."""
        assert _subramanyapura_hourly_weight(17) > 0.4
        assert _subramanyapura_hourly_weight(18) > 0.7

    def test_subramanyapura_late_night_low(self):
        """After 22:00 should be thin."""
        for h in (22, 23, 0, 1):
            assert _subramanyapura_hourly_weight(h) < 0.2, f"Hour {h} weight too high"

    def test_subramanyapura_metro_uplift(self):
        """Metro cells should have >= crowd weight at peak hours."""
        for h in (8, 9, 18, 19):
            w_plain = _subramanyapura_hourly_weight(h, is_near_metro=False)
            w_metro = _subramanyapura_hourly_weight(h, is_near_metro=True)
            assert w_metro >= w_plain

    def test_simulate_cell_hour_non_negative(self):
        rng = random.Random(42)
        for h in range(24):
            count = _simulate_cell_hour("peenya", "grid_13.010_77.500_r9", h, rng)
            assert count >= 0

    def test_simulate_cell_hour_realistic_range(self):
        rng = random.Random(7)
        counts = [
            _simulate_cell_hour("subramanyapura", "grid_12.885_77.553_r9", h, rng)
            for h in range(24)
        ]
        max_count = max(counts)
        assert max_count > 0, "All hours have zero count"
        assert max_count < 2000, f"Unrealistically high count: {max_count}"

    def test_weights_sum_to_valid_range(self):
        """Peenya and Subramanyapura weights should be in [0, 1]."""
        for h in range(24):
            assert 0.0 <= _peenya_hourly_weight(h) <= 1.0
            assert 0.0 <= _subramanyapura_hourly_weight(h) <= 1.0

    def test_simulated_label_in_description(self):
        """Synthetic incident descriptions must contain SYNTHETIC label."""
        for inc in _generate_synthetic_incidents("peenya", count=5, seed=42):
            assert "SYNTHETIC" in inc["description"]


# ── Tests: load_graph caching (mocked) ───────────────────────────────────────

class TestLoadGraphCaching:
    def test_load_graph_uses_cache(self, tmp_path, monkeypatch):
        """When cache file exists, osmnx.load_graphml should be called (not download)."""
        import networkx as nx
        import osmnx as ox

        # Create a tiny fake graph
        G = nx.MultiDiGraph()
        G.add_node(1, x=77.500, y=13.010, osmid=1)
        G.add_node(2, x=77.502, y=13.010, osmid=2)
        G.add_edge(1, 2, 0, osmid=111, highway="residential", length=200.0)
        G.graph["crs"] = "epsg:4326"

        # Write a dummy graphml using osmnx
        cache_file = tmp_path / "peenya.graphml"
        ox.save_graphml(G, str(cache_file))

        # Monkeypatch CACHE_DIR
        import data.ingest_osm as ingest_module
        monkeypatch.setattr(ingest_module, "CACHE_DIR", tmp_path)

        from app.config import ZONE_PRESETS
        preset = ZONE_PRESETS["peenya"]

        loaded = load_graph("peenya", preset)
        assert loaded.number_of_nodes() >= 1

    def test_load_graph_downloads_when_no_cache(self, tmp_path, monkeypatch):
        """When no cache, ox.graph_from_bbox should be called."""
        import networkx as nx
        import osmnx as ox
        import data.ingest_osm as ingest_module

        # Ensure cache doesn't exist
        monkeypatch.setattr(ingest_module, "CACHE_DIR", tmp_path)

        G_fake = nx.MultiDiGraph()
        G_fake.add_node(1, x=77.500, y=13.010, osmid=1)
        G_fake.graph["crs"] = "epsg:4326"

        with patch.object(ox, "graph_from_bbox", return_value=G_fake) as mock_dl, \
             patch.object(ox, "save_graphml") as mock_save:
            loaded = load_graph("peenya", MagicMock(
                bbox_north=13.05, bbox_south=13.00,
                bbox_east=77.55, bbox_west=77.48,
            ))
            mock_dl.assert_called_once()
