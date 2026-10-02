"""
data/seed.py - Full Step 2 seed pipeline for one zone.

Runs: ingest_osm -> seed_crime_data -> simulate_crowd
and prints a combined summary.

Usage:
    make seed ZONE=peenya
    make seed ZONE=subramanyapura
    make seed ZONE=peenya CSV=data/real_incidents.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import Settings, ZONE_PRESETS
from data.ingest_osm import ingest_zone
from data.seed_crime_data import seed_zone
from data.simulate_crowd import simulate_zone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("seed")


def run_all(zone: str, csv_path: Path | None = None) -> None:
    s = Settings()

    if zone not in ZONE_PRESETS:
        raise SystemExit(f"Unknown zone '{zone}'. Valid: {list(ZONE_PRESETS)}")

    log.info("=" * 60)
    log.info("STEP 2 SEED PIPELINE  zone=%s", zone)
    log.info("=" * 60)

    # 1. OSM ingestion
    log.info("[1/3] Ingesting OSM road graph and POIs...")
    osm_summary = ingest_zone(zone, s.DATABASE_URL_SYNC)

    # 2. Crime data seeding
    log.info("[2/3] Seeding synthetic crime incidents...")
    crime_summary = seed_zone(zone, s.DATABASE_URL_SYNC, csv_path=csv_path)

    # 3. Crowd simulation
    log.info("[3/3] Simulating crowd counts...")
    crowd_summary = simulate_zone(zone, s.DATABASE_URL_SYNC, s.REDIS_URL, days=7)

    # Combined summary
    print("\n" + "#" * 60)
    print(f"  STEP 2 COMPLETE  zone={zone}")
    print("#" * 60)
    print(f"  Segments          : {osm_summary['segment_count']:,}")
    print(f"  Lit share         : {osm_summary['lit_segment_share']:.1%}")
    print(f"  Safe havens       : {osm_summary['safe_haven_count']}")
    print("  POIs by category  :")
    for cat, cnt in sorted(osm_summary["poi_counts"].items()):
        print(f"    {cat:<20} {cnt:>5}")
    print(f"  Incidents (synth) : {crime_summary['synthetic_count']:,}")
    print(f"  Incidents (CSV)   : {crime_summary['csv_count']:,}")
    print(f"  Crowd H3 cells    : {crowd_summary['h3_cells']:,}")
    print(f"  Crowd DB rows     : {crowd_summary['db_rows']:,}")
    print("#" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Step 2 seed pipeline.")
    parser.add_argument("zone", nargs="?", default=None)
    parser.add_argument("--csv", type=Path, default=None, help="Real incident CSV path")
    args = parser.parse_args()

    s = Settings()
    zone = args.zone or s.ZONE
    run_all(zone, csv_path=args.csv)


if __name__ == "__main__":
    main()
