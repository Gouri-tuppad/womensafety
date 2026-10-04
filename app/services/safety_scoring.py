"""
app/services/safety_scoring.py - Per-segment risk computation for SafeWay.

Components (all in [0, 1]):
  CrimeRisk      - incident density within 500 m, time-decayed + hour-weighted
  DarknessRisk   - composite: lit_ratio + lamp density + POI proximity proxies
  IsolationRisk  - road type, POI count, distance to main road, crowd count
  ActivityScore  - crowd density from H3 crowd_counts (subtracted in cost formula)

DarknessRisk uses a composite proxy because Peenya's OSM lighting data is sparse
(~1.8% lit segment share). The composite combines:
  - lit_ratio      (40 %) - direct OSM measurement / lamp proximity estimate
  - lamp_count     (20 %) - street lamps within 150 m
  - shop_nearby    (15 %) - retail lights up the street
  - petrol_nearby  (10 %) - 24h fuel stations are always lit
  - main_road      (15 %) - primary/secondary roads are usually lit

Behind RiskProvider interface so the ML model in Step 4 can drop in.

Redis cache key: safety:{zone}:{osm_id}:{time_slot}
  time_slot = f"{hour:02d}{'W' if is_weekend else 'D'}"  e.g. "23W"
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import pytz
from astral import LocationInfo
from astral.sun import sun

from app.config import ZONE_PRESETS, settings

log = logging.getLogger(__name__)

_IST = pytz.timezone("Asia/Kolkata")

# ── Constants ─────────────────────────────────────────────────────────────────

CRIME_RADIUS_M = 500.0
LAMP_BUFFER_M = 150.0
SHOP_BUFFER_M = 150.0
PETROL_BUFFER_M = 200.0
MAIN_ROAD_BUFFER_M = 300.0

# Road types that imply high isolation in the Peenya industrial context
ISOLATED_ROAD_TYPES = {"industrial", "service", "track", "path",
                       "footway", "cycleway", "steps", "unclassified"}
WELL_CONNECTED_TYPES = {"primary", "secondary", "tertiary", "residential",
                        "primary_link", "secondary_link", "tertiary_link"}

CACHE_TTL_S = 3600  # 1 hour


# ── Time context ──────────────────────────────────────────────────────────────

@dataclass
class TimeContext:
    """Encapsulates all time-related factors used in risk computation."""
    hour: int           # 0-23 local
    is_dark: bool       # True after civil sunset, before civil sunrise
    is_weekend: bool    # Saturday or Sunday
    is_holiday: bool    # Indian national / Karnataka state holiday
    zone: str

    @property
    def time_slot(self) -> str:
        """Compact key for caching: '23W' (hour + W/D for weekend/day)."""
        return f"{self.hour:02d}{'W' if self.is_weekend else 'D'}"

    @classmethod
    def from_datetime(cls, dt: datetime, zone: str) -> "TimeContext":
        """Build a TimeContext from a timezone-aware datetime."""
        # Convert to IST
        local_dt = dt.astimezone(_IST)
        hour = local_dt.hour
        weekday = local_dt.weekday()  # 0=Mon, 6=Sun
        is_weekend = weekday >= 5

        # Compute sunset/sunrise for zone centre via astral
        preset = ZONE_PRESETS[zone]
        loc = LocationInfo(
            name=preset.display_name,
            region="India",
            timezone="Asia/Kolkata",
            latitude=preset.center_lat,
            longitude=preset.center_lon,
        )
        s = sun(loc.observer, date=local_dt.date(), tzinfo=_IST)
        is_dark = local_dt < s["dawn"] or local_dt > s["dusk"]

        # Holidays (Indian + Karnataka state)
        is_holiday = False
        try:
            import holidays as hol_lib
            ka_holidays = hol_lib.country_holidays("IN", subdiv="KA", years=local_dt.year)
            is_holiday = local_dt.date() in ka_holidays
        except Exception:
            pass  # graceful degradation

        return cls(
            hour=hour,
            is_dark=is_dark,
            is_weekend=is_weekend,
            is_holiday=is_holiday,
            zone=zone,
        )


# ── RiskProvider interface ────────────────────────────────────────────────────

@runtime_checkable
class RiskProvider(Protocol):
    """
    Interface for CrimeRisk providers.

    Step 3 uses IncidentDensityProvider (spatial + time decay).
    Step 4 will plug in an XGBoost-based ML provider behind this interface.
    """

    def crime_risk(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        time_ctx: TimeContext,
        incident_rows: list[dict],
    ) -> float:
        """
        Return CrimeRisk in [0, 1] for a location given incident data.

        Args:
            lat, lon: segment midpoint
            radius_m: spatial neighbourhood radius
            time_ctx: time factors
            incident_rows: pre-fetched incidents within radius_m (list of dicts)
        """
        ...


# ── Default provider: incident density ───────────────────────────────────────

class IncidentDensityProvider:
    """
    CrimeRisk from incident density within `radius_m`, with:
      - Exponential time decay (half-life = 60 days)
      - Severity weighting (severity 1-5 → weight 0.2-1.0)
      - Hour-of-day match: incidents at similar hours get 1.5x weight
      - Normalised against an empirical zone maximum density
    """

    HALF_LIFE_DAYS = 60.0
    MAX_EXPECTED_WEIGHTED_INCIDENTS = 15.0  # empirical cap for Peenya seeded data

    def crime_risk(
        self,
        lat: float,
        lon: float,
        radius_m: float,
        time_ctx: TimeContext,
        incident_rows: list[dict],
    ) -> float:
        if not incident_rows:
            return 0.0

        now = datetime.now(tz=timezone.utc)
        total_weight = 0.0

        for inc in incident_rows:
            # Time decay: recent incidents count more
            occurred = inc.get("occurred_at")
            if occurred is None:
                days_ago = 180.0
            elif hasattr(occurred, "timestamp"):
                days_ago = (now - occurred.astimezone(timezone.utc)).days
            else:
                days_ago = 180.0

            decay = math.exp(-0.693 * days_ago / self.HALF_LIFE_DAYS)  # 0.693 = ln(2)

            # Severity: 1→0.2, 2→0.4, ..., 5→1.0
            severity = max(1, min(5, inc.get("severity", 3)))
            sev_weight = severity / 5.0

            # Hour match: incidents at the same hour ±2 count 1.5x
            inc_hour = inc.get("hour")
            if inc_hour is not None:
                hour_diff = abs(inc_hour - time_ctx.hour)
                hour_diff = min(hour_diff, 24 - hour_diff)  # circular
                hour_factor = 1.5 if hour_diff <= 2 else 1.0
            else:
                hour_factor = 1.0

            total_weight += decay * sev_weight * hour_factor

        # Normalise against empirical max
        risk = total_weight / self.MAX_EXPECTED_WEIGHTED_INCIDENTS
        return max(0.0, min(1.0, risk))


# ── Segment feature bundle ────────────────────────────────────────────────────

@dataclass
class SegmentFeatures:
    """All pre-fetched features needed to score one segment."""
    osm_id: int
    lat: float          # midpoint latitude
    lon: float          # midpoint longitude
    road_type: str
    length_m: float
    lit_ratio: float
    lamp_count: int
    # POI proximity (counts within buffers)
    shop_count_150m: int = 0
    petrol_count_200m: int = 0
    any_metro_300m: bool = False
    dist_main_m: float = 9999.0   # metres to nearest primary/secondary
    # Crowd
    crowd_count: int = 0
    # Incidents within CRIME_RADIUS_M (pre-fetched list of dicts)
    incidents: list[dict] = field(default_factory=list)


# ── Scoring engine ────────────────────────────────────────────────────────────

@dataclass
class RiskScores:
    """Per-segment risk scores (all in [0, 1])."""
    crime_risk: float
    darkness_risk: float
    isolation_risk: float
    activity_score: float

    @property
    def composite(self) -> float:
        """Unweighted composite risk (equal weights), used for heatmap display."""
        return (self.crime_risk + self.darkness_risk + self.isolation_risk) / 3.0


class SegmentScorer:
    """
    Computes all four risk components for a road segment at a given time.

    Args:
        risk_provider: CrimeRisk provider (IncidentDensityProvider by default).
                       Step 4 replaces this with an ML model.
    """

    def __init__(self, risk_provider: RiskProvider | None = None) -> None:
        self.risk_provider: RiskProvider = risk_provider or IncidentDensityProvider()

    # ── Public API ────────────────────────────────────────────────────────────

    def score(self, features: SegmentFeatures, time_ctx: TimeContext) -> RiskScores:
        """Compute all four risk components for one segment at given time."""
        crime = self.risk_provider.crime_risk(
            lat=features.lat,
            lon=features.lon,
            radius_m=CRIME_RADIUS_M,
            time_ctx=time_ctx,
            incident_rows=features.incidents,
        )
        darkness = self._darkness_risk(features, time_ctx)
        isolation = self._isolation_risk(features, time_ctx)
        activity = self._activity_score(features)

        return RiskScores(
            crime_risk=crime,
            darkness_risk=darkness,
            isolation_risk=isolation,
            activity_score=activity,
        )

    # ── Component calculations ────────────────────────────────────────────────

    def _darkness_risk(self, f: SegmentFeatures, ctx: TimeContext) -> float:
        """
        Composite darkness risk using lit_ratio + multiple lighting proxies.

        Because Peenya OSM lit data covers only ~1.8% of segments, we use
        indirect indicators (shops, petrol pumps, main roads, metro) as proxies.

        Intrinsic lighting score (0-1):
          40% lit_ratio        - direct OSM / lamp proximity measurement
          20% lamp density     - street lamps per 150 m buffer (capped at 8)
          15% shop proximity   - retail uses lights even at night
          10% petrol proximity - 24h fuel stations always lit
          15% main road prox   - primary/secondary roads usually well lit

        Then modulated by time-of-day:
          night (is_dark): darkness_risk = 1 - intrinsic  (max effect)
          day:             darkness_risk = 0.15 * (1 - intrinsic)  (minimal)
        """
        lamp_score = min(f.lamp_count / 8.0, 1.0)
        shop_score = min(f.shop_count_150m / 3.0, 1.0)
        petrol_score = 1.0 if f.petrol_count_200m > 0 else 0.0
        metro_score = 1.0 if f.any_metro_300m else 0.0
        main_road_score = max(0.0, 1.0 - f.dist_main_m / MAIN_ROAD_BUFFER_M)

        intrinsic = (
            0.40 * f.lit_ratio
            + 0.20 * lamp_score
            + 0.15 * shop_score
            + 0.10 * petrol_score
            + 0.15 * main_road_score
            # metro contributes via shop_score path but also directly
        ) + 0.05 * metro_score  # small direct bonus; re-normalise below

        # Cap at 1.0 (the extra 0.05 for metro might push it over)
        intrinsic = min(intrinsic, 1.0)

        # Night uplift factor: additional crime-time darkness boost for specific hours
        # Peenya is very dark 22:00-05:00 (no shift activity, no shops open)
        night_hours_factor = 1.0
        if ctx.is_dark:
            if 22 <= ctx.hour or ctx.hour <= 5:
                night_hours_factor = 1.2  # extra darkness at deepest night

        if ctx.is_dark:
            base = 1.0 - intrinsic
            darkness = min(1.0, base * night_hours_factor)
        else:
            # During the day darkness_risk is small but still varies
            darkness = 0.15 * (1.0 - intrinsic)

        return round(max(0.0, min(1.0, darkness)), 4)

    def _isolation_risk(self, f: SegmentFeatures, ctx: TimeContext) -> float:
        """
        Isolation risk: penalises industrial roads, distant from main roads,
        few POIs, and low crowd count.

        Road type  (40%):
          industrial/service/track → 0.8
          unclassified             → 0.6
          residential              → 0.3
          tertiary                 → 0.2
          primary/secondary        → 0.1

        POI density (30%): fewer shops + lamps = more isolated
        Main road distance (20%): far from arterial = more isolated
        Crowd (10%): low crowd = more isolated
        """
        rt = f.road_type.lower().replace("_link", "")

        if rt in ("industrial", "track", "path", "footway", "cycleway", "steps"):
            road_score = 0.85
        elif rt in ("service", "unclassified"):
            road_score = 0.65
        elif rt in ("living_street", "residential"):
            road_score = 0.35
        elif rt in ("tertiary",):
            road_score = 0.20
        elif rt in ("secondary",):
            road_score = 0.12
        elif rt in ("primary", "trunk", "motorway"):
            road_score = 0.05
        else:
            road_score = 0.50  # unknown

        # POI density: use lamp_count + shop_count as proxy
        total_pois = f.lamp_count + f.shop_count_150m
        poi_score = max(0.0, 1.0 - total_pois / 8.0)

        # Main road proximity
        main_score = min(f.dist_main_m / 500.0, 1.0)

        # Crowd
        # Peenya night baseline is ~5 people/cell; well-attended areas ~60
        crowd_score = max(0.0, 1.0 - f.crowd_count / 50.0)

        # Night uplift: industrial roads more isolating at night
        night_bonus = 0.0
        if ctx.is_dark and rt in ("industrial", "service", "track"):
            night_bonus = 0.15

        isolation = (
            0.40 * road_score
            + 0.30 * poi_score
            + 0.20 * main_score
            + 0.10 * crowd_score
            + night_bonus
        )
        return round(max(0.0, min(1.0, isolation)), 4)

    def _activity_score(self, f: SegmentFeatures) -> float:
        """
        Activity score from crowd_counts (subtracted in cost formula).
        Peenya peak at shift changes: ~80-120 people/cell.
        Night baseline: ~3-8 people/cell.
        """
        # Peenya zone baseline: 80 people/cell at shift change
        return round(min(f.crowd_count / 80.0, 1.0), 4)


# ── Redis cache helpers ───────────────────────────────────────────────────────

# In-process memory cache for blazing-fast Dijkstra routing (< 10 ms)
_in_memory_slot_cache: dict[tuple[str, str], dict[int, RiskScores]] = {}
_static_features_cache: dict[str, dict[int, dict]] = {}


def _cache_key(zone: str, osm_id: int, time_slot: str) -> str:
    return f"safety:{zone}:{osm_id}:{time_slot}"


def _slot_hash_key(zone: str, time_slot: str) -> str:
    return f"safety_slot:{zone}:{time_slot}"


def get_cached_scores(redis_client, zone: str, osm_id: int, time_slot: str) -> RiskScores | None:
    """Return cached RiskScores from Redis or in-memory cache, or None on miss."""
    # Check in-memory cache first
    slot_dict = _in_memory_slot_cache.get((zone, time_slot))
    if slot_dict and osm_id in slot_dict:
        return slot_dict[osm_id]

    if redis_client is None:
        return None

    try:
        raw = redis_client.get(_cache_key(zone, osm_id, time_slot))
        if raw is None:
            return None
        d = json.loads(raw)
        return RiskScores(
            crime_risk=d["crime_risk"],
            darkness_risk=d["darkness_risk"],
            isolation_risk=d["isolation_risk"],
            activity_score=d["activity_score"],
        )
    except Exception as exc:
        log.debug("Cache get failed for %s/%s/%s: %s", zone, osm_id, time_slot, exc)
        return None


def get_slot_scores(redis_client, zone: str, time_slot: str) -> dict[int, RiskScores]:
    """
    Return all cached RiskScores for an entire time slot as a dict {osm_id: RiskScores}.
    Ultra-fast: uses in-memory cache first, then Redis hash 'safety_slot:{zone}:{time_slot}'.
    """
    if (zone, time_slot) in _in_memory_slot_cache:
        return _in_memory_slot_cache[(zone, time_slot)]

    if redis_client is None:
        return {}

    try:
        raw_hash = redis_client.hgetall(_slot_hash_key(zone, time_slot))
        if raw_hash:
            scores_dict: dict[int, RiskScores] = {}
            for osm_id_str, val in raw_hash.items():
                d = json.loads(val)
                scores_dict[int(osm_id_str)] = RiskScores(
                    crime_risk=float(d["crime_risk"]),
                    darkness_risk=float(d["darkness_risk"]),
                    isolation_risk=float(d["isolation_risk"]),
                    activity_score=float(d["activity_score"]),
                )
            _in_memory_slot_cache[(zone, time_slot)] = scores_dict
            return scores_dict
    except Exception as exc:
        log.warning("Failed to read slot hash for %s/%s: %s", zone, time_slot, exc)

    return {}


def set_cached_scores(redis_client, zone: str, osm_id: int, time_slot: str,
                      scores: RiskScores, ttl: int = CACHE_TTL_S) -> None:
    """Store RiskScores in Redis and in-memory cache."""
    # In-memory
    _in_memory_slot_cache.setdefault((zone, time_slot), {})[osm_id] = scores

    if redis_client is None:
        return

    try:
        payload = json.dumps({
            "crime_risk": scores.crime_risk,
            "darkness_risk": scores.darkness_risk,
            "isolation_risk": scores.isolation_risk,
            "activity_score": scores.activity_score,
        })
        redis_client.set(_cache_key(zone, osm_id, time_slot), payload, ex=ttl)
    except Exception as exc:
        log.debug("Cache set failed: %s", exc)


# ── Bulk precompute (called at startup / make precompute) ─────────────────────

def precompute_zone_weights(
    zone: str,
    hour: int,
    is_weekend: bool,
    db_url: str,
    redis_url: str,
    scorer: SegmentScorer | None = None,
) -> int:
    """
    Precompute and cache safety scores for ALL segments in a zone for one time slot.

    Returns the number of segments processed.

    Optimized for Docker memory & latency constraints:
      1. Uses fast box queries instead of slow circular buffer polygons.
      2. Caches static segment spatial features in-memory across calls.
      3. Pipelines Redis writes in batches of 1,000 for sub-second database caching.
      4. Stores both individual keys and a Redis hash for fast bulk lookup.
    """
    import math as _math
    from datetime import date, timedelta

    import shapely.geometry as sg
    from shapely import STRtree, box
    from sqlalchemy import create_engine, text

    if scorer is None:
        scorer = SegmentScorer()

    # Synthesise a representative datetime for this time slot
    today = date.today()
    if is_weekend:
        days_to_sat = (5 - today.weekday()) % 7
        ref_date = today + timedelta(days=days_to_sat)
    else:
        days_to_mon = (0 - today.weekday()) % 7
        ref_date = today + timedelta(days=days_to_mon)

    import pytz as _pytz
    _ist = _pytz.timezone("Asia/Kolkata")
    ref_dt = _ist.localize(
        __import__("datetime").datetime(ref_date.year, ref_date.month, ref_date.day, hour, 0, 0)
    )
    time_ctx = TimeContext.from_datetime(ref_dt, zone)

    engine = create_engine(db_url, echo=False)

    with engine.connect() as conn:
        # 1. Road segments
        segs_result = conn.execute(text(
            "SELECT osm_id, road_type, length_m, lit_ratio, lamp_count, "
            "ST_X(ST_Centroid(geometry)) AS lon, ST_Y(ST_Centroid(geometry)) AS lat "
            "FROM road_segments WHERE zone = :z"
        ), {"z": zone})
        segments = [dict(row._mapping) for row in segs_result]
        log.info("Loaded %d segments for zone=%s", len(segments), zone)

        # 2. POIs by category (if static cache not warm)
        pois = []
        if zone not in _static_features_cache:
            pois_result = conn.execute(text(
                "SELECT category, ST_X(geometry) AS lon, ST_Y(geometry) AS lat "
                "FROM pois WHERE zone = :z"
            ), {"z": zone})
            pois = [dict(row._mapping) for row in pois_result]

        # 3. Crowd counts for this hour
        crowd_result = conn.execute(text(
            "SELECT h3_index, count FROM crowd_counts "
            "WHERE zone = :z AND hour_utc = :h "
            "ORDER BY date_utc DESC LIMIT 100000"
        ), {"z": zone, "h": hour})
        crowd_by_h3: dict[str, int] = {}
        for row in crowd_result:
            h3i = row._mapping["h3_index"]
            if h3i not in crowd_by_h3:
                crowd_by_h3[h3i] = int(row._mapping["count"])

        # 4. Incidents: bulk load and bucket by proximity
        inc_result = conn.execute(text(
            "SELECT "
            "  ST_X(geometry) AS lon, ST_Y(geometry) AS lat, "
            "  severity, occurred_at, "
            "  EXTRACT(HOUR FROM occurred_at AT TIME ZONE 'Asia/Kolkata') AS hour "
            "FROM incidents WHERE zone = :z AND source = 'synthetic' "
            "  AND occurred_at >= NOW() - INTERVAL '365 days'"
        ), {"z": zone})
        incidents_raw = [dict(row._mapping) for row in inc_result]
        log.info("Loaded %d incidents for scoring", len(incidents_raw))

    def _mpd(lat: float) -> float:
        return 111_132.954 - 559.822 * _math.cos(2 * _math.radians(lat)) + 1.175 * _math.cos(4 * _math.radians(lat))

    # Pre-compute static spatial features if not cached
    if zone not in _static_features_cache:
        log.info("Computing static spatial features for %s segments...", zone)
        shop_pts = [sg.Point(p["lon"], p["lat"]) for p in pois if p["category"] in ("shop", "atm")]
        petrol_pts = [sg.Point(p["lon"], p["lat"]) for p in pois if p["category"] == "petrol"]
        metro_pts = [sg.Point(p["lon"], p["lat"]) for p in pois if p["category"] == "metro"]
        main_pts = [sg.Point(s["lon"], s["lat"]) for s in segments
                    if s["road_type"].lower() in ("primary", "secondary", "trunk")]

        shop_tree = STRtree(shop_pts) if shop_pts else None
        petrol_tree = STRtree(petrol_pts) if petrol_pts else None
        metro_tree = STRtree(metro_pts) if metro_pts else None
        main_tree = STRtree(main_pts) if main_pts else None

        static_dict: dict[int, dict] = {}
        for seg in segments:
            osm_id = int(seg["osm_id"])
            lat, lon = float(seg["lat"]), float(seg["lon"])
            mpd = _mpd(lat)
            d_shop = SHOP_BUFFER_M / mpd
            d_petrol = PETROL_BUFFER_M / mpd
            d_metro = 300.0 / mpd

            shop_count = len(shop_tree.query(box(lon - d_shop, lat - d_shop, lon + d_shop, lat + d_shop))) if shop_tree else 0
            petrol_count = len(petrol_tree.query(box(lon - d_petrol, lat - d_petrol, lon + d_petrol, lat + d_petrol))) if petrol_tree else 0
            any_metro = len(metro_tree.query(box(lon - d_metro, lat - d_metro, lon + d_metro, lat + d_metro))) > 0 if metro_tree else False

            dist_main_m = 9999.0
            if main_tree is not None:
                pt = sg.Point(lon, lat)
                idx = main_tree.nearest(pt)
                if idx is not None:
                    dist_main_m = float(pt.distance(main_pts[idx])) * mpd

            static_dict[osm_id] = {
                "shop_count": shop_count,
                "petrol_count": petrol_count,
                "any_metro": any_metro,
                "dist_main_m": dist_main_m,
            }
        _static_features_cache[zone] = static_dict

    static_features = _static_features_cache[zone]

    # Incident spatial tree
    inc_pts = [sg.Point(i["lon"], i["lat"]) for i in incidents_raw]
    inc_tree = STRtree(inc_pts) if inc_pts else None

    # H3 helper
    try:
        import h3 as _h3
        def _h3_cell(lat: float, lon: float) -> str:
            return _h3.latlng_to_cell(lat, lon, 9)
    except ImportError:
        def _h3_cell(lat: float, lon: float) -> str:
            return f"grid_{lat:.3f}_{lon:.3f}_r9"

    import redis as _redis_lib
    rc = _redis_lib.from_url(redis_url, decode_responses=True)

    slot_key = time_ctx.time_slot
    slot_scores: dict[int, RiskScores] = {}
    pipe = rc.pipeline(transaction=False)
    slot_hash_data: dict[str, str] = {}

    processed = 0
    for seg in segments:
        osm_id = int(seg["osm_id"])
        lat, lon = float(seg["lat"]), float(seg["lon"])
        mpd = _mpd(lat)
        d_crime = CRIME_RADIUS_M / mpd

        # Incidents within CRIME_RADIUS_M via fast box query
        nearby_incidents: list[dict] = []
        if inc_tree is not None:
            nearby_idxs = inc_tree.query(box(lon - d_crime, lat - d_crime, lon + d_crime, lat + d_crime))
            nearby_incidents = [incidents_raw[i] for i in nearby_idxs]

        # Crowd from H3
        h3_cell = _h3_cell(lat, lon)
        crowd_count = crowd_by_h3.get(h3_cell, 0)

        s_feat = static_features.get(osm_id, {})

        features = SegmentFeatures(
            osm_id=osm_id,
            lat=lat,
            lon=lon,
            road_type=str(seg["road_type"]),
            length_m=float(seg["length_m"]),
            lit_ratio=float(seg["lit_ratio"]),
            lamp_count=int(seg["lamp_count"]),
            shop_count_150m=s_feat.get("shop_count", 0),
            petrol_count_200m=s_feat.get("petrol_count", 0),
            any_metro_300m=s_feat.get("any_metro", False),
            dist_main_m=s_feat.get("dist_main_m", 9999.0),
            crowd_count=crowd_count,
            incidents=nearby_incidents,
        )

        scores = scorer.score(features, time_ctx)
        slot_scores[osm_id] = scores
        processed += 1

    # ── Min-max normalise each risk component across the entire slot ──
    # Raw scores cluster in narrow bands (e.g. darkness_risk 0.7-1.0 at night
    # because Peenya OSM lighting data is sparse).  Normalisation stretches
    # each component to [0, 1] so the cost multiplier actually differentiates
    # safe from unsafe edges and Dijkstra can find meaningfully different routes.
    slot_scores = _normalize_slot_scores(slot_scores)

    # ── Write normalised scores to Redis ──────────────────────────────
    for osm_id, scores in slot_scores.items():
        payload = json.dumps({
            "crime_risk": scores.crime_risk,
            "darkness_risk": scores.darkness_risk,
            "isolation_risk": scores.isolation_risk,
            "activity_score": scores.activity_score,
        })
        pipe.set(_cache_key(zone, osm_id, slot_key), payload, ex=CACHE_TTL_S * 24 * 7)
        slot_hash_data[str(osm_id)] = payload

        if len(slot_hash_data) % 2000 == 0:
            pipe.execute()

    # Flush remaining pipeline
    pipe.execute()

    # Store full slot hash in Redis
    if slot_hash_data:
        try:
            rc.hset(_slot_hash_key(zone, slot_key), mapping=slot_hash_data)
            rc.expire(_slot_hash_key(zone, slot_key), CACHE_TTL_S * 24 * 7)
        except Exception as exc:
            log.warning("Failed to store slot hash: %s", exc)

    # Save to memory cache
    _in_memory_slot_cache[(zone, slot_key)] = slot_scores

    log.info("Precompute complete: %d segments, time_slot=%s", processed, slot_key)
    return processed


def _normalize_slot_scores(slot_scores: dict[int, RiskScores]) -> dict[int, RiskScores]:
    """
    Min-max normalise each risk component across all segments in a time slot.

    Without this, risk components (especially darkness_risk at night) cluster in
    a narrow band because Peenya's sparse OSM lighting data makes most segments
    look similarly dark.  The result is that the cost multiplier barely varies
    across edges and Dijkstra returns nearly identical fastest/safest routes.

    After normalisation every component spans [0, 1], giving the cost formula
    real discriminating power.  A 5% floor/ceiling trim avoids extreme outlier
    sensitivity.
    """
    if len(slot_scores) < 2:
        return slot_scores

    crimes = [s.crime_risk for s in slot_scores.values()]
    darks  = [s.darkness_risk for s in slot_scores.values()]
    isos   = [s.isolation_risk for s in slot_scores.values()]
    acts   = [s.activity_score for s in slot_scores.values()]

    def _minmax(vals: list[float]) -> tuple[float, float]:
        """Return (min, max) with 5th/95th percentile trim for robustness."""
        sv = sorted(vals)
        n = len(sv)
        lo = sv[max(0, int(n * 0.05))]
        hi = sv[min(n - 1, int(n * 0.95))]
        return (lo, hi)

    def _scale(val: float, lo: float, hi: float) -> float:
        if hi - lo < 1e-6:
            return 0.5  # no variance → neutral
        return max(0.0, min(1.0, (val - lo) / (hi - lo)))

    c_lo, c_hi = _minmax(crimes)
    d_lo, d_hi = _minmax(darks)
    i_lo, i_hi = _minmax(isos)
    a_lo, a_hi = _minmax(acts)

    log.info(
        "Normalising slot scores: crime [%.3f,%.3f] dark [%.3f,%.3f] "
        "iso [%.3f,%.3f] act [%.3f,%.3f]",
        c_lo, c_hi, d_lo, d_hi, i_lo, i_hi, a_lo, a_hi,
    )

    normalised: dict[int, RiskScores] = {}
    for osm_id, s in slot_scores.items():
        normalised[osm_id] = RiskScores(
            crime_risk=round(_scale(s.crime_risk, c_lo, c_hi), 4),
            darkness_risk=round(_scale(s.darkness_risk, d_lo, d_hi), 4),
            isolation_risk=round(_scale(s.isolation_risk, i_lo, i_hi), 4),
            activity_score=round(_scale(s.activity_score, a_lo, a_hi), 4),
        )

    return normalised
