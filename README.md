# 🛡️ SafeWay – Safest-Route Navigation API

> **Backend API** that returns the **safest** walking route instead of the fastest, using crime history, street lighting, crowd activity, and time of day.  Built for women and children's safety in urban environments.

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Safest Route** | Dijkstra's algorithm on safety-weighted street graph (OSMnx + NetworkX) |
| **Safety Cost Formula** | `cost = distance × clamp(1 + α·Crime + β·Darkness + γ·Isolation − δ·Activity, min=0.5)` |
| **Shake-Gesture SOS** | One-tap emergency alert → SMS + push notifications to contacts & authorities |
| **Live Trip Monitoring** | Real-time deviation detection, staleness alerts, destination proximity |
| **Authority Dashboard** | Crime heatmaps (H3), zone safety overviews, SOS alert management |
| **ML Risk Prediction** | XGBoost model trained on synthetic crime data with SHAP explanations |
| **Time-Aware** | Darkness risk via `astral` (sunset/sunrise), Karnataka holidays via `holidays` |
| **Zone Presets** | Peenya (industrial) and Subramanyapura (residential) in Bengaluru |

## 🏗️ Architecture

```
app/
├── main.py              # FastAPI application & lifespan
├── config.py            # Settings, zone presets, env vars
├── deps.py              # Dependency injection (DB, Redis)
├── api/v1/              # REST endpoints
│   ├── routes.py        #   POST /routes/safest
│   ├── sos.py           #   POST /sos/trigger, GET/PATCH /sos/alerts
│   ├── trips.py         #   POST /trips/start, /update, /end
│   ├── dashboard.py     #   GET /dashboard/stats, /heatmap, /safety-overview
│   └── users.py         #   CRUD user management
├── core/                # Exceptions, security helpers
├── models/              # SQLAlchemy 2 ORM models (PostGIS-ready)
├── schemas/             # Pydantic v2 request/response schemas
├── services/            # Business logic
│   ├── graph_service.py #   OSMnx graph loading + caching
│   ├── safety_service.py#   Safety-cost formula engine
│   ├── route_service.py #   Route computation (Dijkstra)
│   ├── trip_service.py  #   Live trip monitoring
│   ├── sos_service.py   #   SOS alert handling
│   └── notification_service.py  # SMS/Push/LLM interfaces (mock)
├── ml/                  # Machine learning pipeline
│   ├── synthetic_data.py#   Synthetic crime data generator
│   ├── features.py      #   Feature engineering (H3, temporal, spatial)
│   └── model.py         #   XGBoost training, SHAP, persistence
└── data/
    └── seed.py          # Database seeding script
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose (for PostgreSQL + Redis)

### Option 1: Docker Compose (recommended)

```bash
cp .env.example .env
docker compose up --build
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Option 2: Local Development

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate on Windows
pip install -r requirements.txt
cp .env.example .env

# Start PostgreSQL + Redis separately, then:
uvicorn app.main:app --reload
```

## 🧪 Testing

SafeWay includes comprehensive unit, integration, and end-to-end smoke test suites.

### 1. Pytest Test Suite
To run the automated Python test suite inside Docker or locally:

```powershell
# Inside Docker container (recommended)
docker compose exec api pytest -v

# Or quiet mode
docker compose exec -T api pytest -q

# Locally (if venv is active and DB/Redis are running)
pytest tests/ -v
```



### 3. Unified Verification Runner (`scripts/run_all_checks.ps1`)
Runs the full verification pipeline in sequence:
1. `docker compose ps` (container availability and health checks)
2. `pytest -q` (full 160-test pytest suite)
3. `scripts/smoke.ps1` (end-to-end API test suite)
4. Displays a consolidated summary table with pass/fail counts and final verdict.

**Run All Checks:**
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_all_checks.ps1
```


## 🗺️ Demo Zone & Fixture Routes (Peenya Only)

> **SCOPE**: SafeWay backend currently focuses **strictly on the Peenya zone**.

### Peenya Industrial Area
- **Bounding Box**: North 13.05, South 13.00, East 77.55, West 77.48
- **Character**: Factories, warehouses, garment units, long compound walls, sparse lighting
- **Shift Changes**: Peak crowd and factory shift turnover (~06:00–07:00, 14:00–15:00, 22:00–23:00)

### Demo Route : Bagalagunte to Peenya 1st Stage South-East (Across Industrial Area)
- **Origin**: `(13.0450, 77.5050)` [Bagalagunte / MEI Colony]
- **Destination**: `(13.0150, 77.5300)` [Peenya 1st Stage South-East]
- **Behavior at 11 PM (23:00)**:
  - **Fastest**: 5,718.1 m (safety score 56.2) – cuts directly through unlit, isolated industrial alleys.
  - **Safest**: 6,395.9 m (safety score 67.7) – detours along well-lit, active arterial corridors.
  - **Metrics**: **+11.9% longer**, **+11.5 safety pts gain**, 59% lower crime exposure, avoids isolated roads (0% vs 15%), 1 police station nearby.

<img width="1934" height="838" alt="Screenshot 2026-10-07 172550" src="https://github.com/user-attachments/assets/4e79f5de-c2c0-4b13-85e0-d07737eab928" />


## 🧪 Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI + Pydantic v2 |
| Database | PostgreSQL + PostGIS, SQLAlchemy 2 + Alembic |
| Cache | Redis |
| Routing | OSMnx + NetworkX |
| ML | XGBoost, scikit-learn, SHAP |
| Spatial | H3, Shapely |
| Time | astral (sunset), holidays (KA public holidays) |
| Notifications | Mock SMS/Push/LLM (interface-based, swappable) |
| Testing | pytest, httpx |
| Deployment | Docker + docker-compose |

## 📝 License

MIT
