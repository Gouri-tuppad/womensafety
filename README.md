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

### Run Tests

```bash
pytest tests/ -v
```

### Train ML Model

```bash
python -m app.ml.model          # train + evaluate
python -m app.ml.model --shap   # with SHAP explanations
```

## 📡 API Endpoints (Step 3: Safety Scoring & Routing)

### 1. Compute Route (Fastest vs Safest)
```http
POST /route
Content-Type: application/json

{
  "origin": {"lat": 13.0330, "lon": 77.5332},
  "destination": {"lat": 13.0220, "lon": 77.5034},
  "depart_time": "2026-10-02T23:00:00+05:30",
  "profile": "woman_night"
}
```

Response includes:
- `fastest`: Distance, duration, safety score, colour-coded segments, polyline
- `safest`: Distance, duration, safety score, colour-coded segments, polyline
- `routes_differ`: `true` when safest navigates away from high-risk streets
- `safe_havens_near_safest`: Hospitals, police stations, metro & bus stops within 300 m
- `comparison`: Summary statistics (e.g., `"+2.6% longer, 1.6x more lit, +8.2 safety pts"`)

*(Also available under `/api/v1/route`)*

### 2. Safety Heatmap Grid (H3 Resolution 9)
```http
GET /safety/heatmap?zone=peenya&hour=23
```
Returns an H3 risk grid (resolution 9) with `risk_score` (0–1 composite), `crime_risk`, `darkness_risk`, `isolation_risk`, and `crowd_count`, designed for time-slider UI map overlays.

*(Also available under `/api/v1/safety/heatmap`)*

### 3. Edge Weight Precomputation
```http
GET /safety/precompute?hour=23&is_weekend=false
```
Precomputes and caches safety edge weights in Redis for sub-second routing performance (< 100 ms).

## ⚙️ Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ZONE` | `peenya` | Active zone: `peenya` (Scope: Peenya only) |
| `ALPHA_CRIME` | `0.4` | Default CrimeRisk weight in safety formula |
| `BETA_DARKNESS` | `0.3` | Default DarknessRisk weight |
| `GAMMA_ISOLATION` | `0.2` | Default IsolationRisk weight |
| `DELTA_ACTIVITY` | `0.1` | Default ActivityScore weight (reduces cost) |

## 🗺️ Demo Zone & Fixture Routes (Peenya Only)

> **SCOPE**: SafeWay backend currently focuses **strictly on the Peenya zone**.

### Peenya Industrial Area
- **Bounding Box**: North 13.05, South 13.00, East 77.55, West 77.48
- **Character**: Factories, warehouses, garment units, long compound walls, sparse lighting
- **Shift Changes**: Peak crowd and factory shift turnover (~06:00–07:00, 14:00–15:00, 22:00–23:00)

### Demo Route 1: Bagalagunte to Peenya 1st Stage South-East (Across Industrial Area)
- **Origin**: `(13.0450, 77.5050)` [Bagalagunte / MEI Colony]
- **Destination**: `(13.0150, 77.5300)` [Peenya 1st Stage South-East]
- **Behavior at 11 PM (23:00)**:
  - **Fastest**: 5,718.1 m (safety score 56.2) – cuts directly through unlit, isolated industrial alleys.
  - **Safest**: 6,395.9 m (safety score 67.7) – detours along well-lit, active arterial corridors.
  - **Metrics**: **+11.9% longer**, **+11.5 safety pts gain**, 59% lower crime exposure, avoids isolated roads (0% vs 15%), 1 police station nearby.

### Demo Route 2: Dasarahalli Metro to Peenya 1st Stage South (Across Industrial Area)
- **Origin**: `(13.0440, 77.5130)` [Dasarahalli Metro]
- **Destination**: `(13.0120, 77.5320)` [Peenya 1st Stage South]
- **Behavior at 11 PM (23:00)**:
  - **Fastest**: 4,920.3 m (safety score 50.6) – traverses high-crime, dark factory roads.
  - **Safest**: 5,410.3 m (safety score 65.0) – navigates around dangerous segments via active corridors.
  - **Metrics**: **+10.0% longer**, **+14.4 safety pts gain**, 48% lower crime exposure, 1 police station nearby.

### Additional Peenya Route: Peenya Metro to Peenya 2nd Stage (NTTF / 12th Cross)
- **Origin**: `(13.0330, 77.5332)` [Peenya Metro]
- **Destination**: `(13.0220, 77.5034)` [Peenya 2nd Stage]
- **Behavior**: Safest route differs at 11 PM vs 6 PM due to nightfall and crowd shifts.

## 📊 Safety Cost Formula

```
cost = distance × clamp(1 + α·CrimeRisk + β·DarknessRisk + γ·IsolationRisk − δ·ActivityScore, min=0.5)
```

- All four inputs normalised to [0, 1]
- Multiplier clamped to minimum 0.5 (a segment can be at most 2× "faster" than raw distance)
- **Safety score** of a route (0–100) = 100 − length-weighted mean risk

## 🤖 ML Risk Prediction & Mitigating Reporting Bias

### 1. ML Backend Architecture (Step 4)
- **Primary Model**: Gradient-boosted decision trees via **XGBoost** (`ml/models/xgb_risk.json`).
- **Baseline Comparison**: **Random Forest** Regressor (`ml/models/rf_risk.joblib`).
- **Time-Based Splitting**: Data is split strictly chronologically (training on older time periods and testing on the latest ~25% of time slots) to prevent temporal data leakage. No random splitting.
- **Performance & Metrics**: Model metrics (AUC, RMSE, baseline comparisons, feature importances) are recorded in `ml/reports/metrics.json` via `python -m app.ml.train_risk_model` (wired to `make train`).
- **Redis-Cached Batch Inference**: `predict_risk(segment_ids, timestamp)` performs batch prediction and caches results in Redis (`ml_pred:{zone}:{time_slot}`) for sub-second routing performance.
- **SHAP TreeExplainer**: Generates plain-English contributing factors (e.g., *"Well lit: 92% of road lit"*, *"Metro station within 300 m"*, *"Late night: very few people around"*) returned in the `risk_explanations` field of the `POST /route` response.

### 2. How Environmental Features Reduce Reporting Bias
Traditional crime-prediction and routing systems that depend exclusively on historical police incident records suffer from **severe reporting bias**:
1. **Under-Reporting**: Low-income, marginalized, or industrial areas often have lower reporting rates due to lack of police presence, lower trust in law enforcement, or fewer reporting channels.
2. **Over-Policing / Patrol Bias**: Higher-income commercial and residential belts often feature higher police visibility and more recorded FIRs, creating an artificial signal of higher risk despite better lighting and emergency access.

SafeWay addresses this bias by grounding risk prediction primarily in **observable, objectively verifiable environmental and spatial features**:
- **Physical Lighting**: Lamp density per 100 m and measured segment `lit_ratio`.
- **Infrastructure Proximity**: Measured network distances to nearest police stations, hospitals, and transit hubs (metro stations and bus stops).
- **Road Typology**: Structural road classes (e.g. isolated service/industrial lanes with compound walls vs active arterial avenues).
- **Urban Density Proxies**: POI counts (shops, ATMs, pharmacies) providing passive natural surveillance ("eyes on the street").
- **Temporal & Dynamic Crowd Activity**: Solar time-based darkness calculations and hourly crowd volume proxies.

Because the XGBoost model learns risk relationships from these physical attributes, **segments with zero incident history still receive accurate, safety-reflective risk assessments**.

### 3. Fairness Audit & Disparity Monitoring
The fairness audit (`app/ml/fairness_audit.py`) compares model predictions across socio-economically diverse neighborhoods (e.g. Peenya's industrial core vs residential Subramanyapura) grouped by affluence proxies (amenity density, road type distributions):
- Automatically flags disparities where mean predicted risk diverges beyond `DISPARITY_THRESHOLD = 0.15`.
- Writes full audit metrics and pairwise comparisons to `ml/reports/fairness.json`.
- Accessible to municipal authorities via `GET /dashboard/fairness` (requires authority role).

## ⚠️ Synthetic Data Disclaimer

**All crime incident data, street lighting scores, crowd levels, and CCTV indicators
are SYNTHETIC / SIMULATED.**  They are generated programmatically to reflect realistic
spatial and temporal patterns for demonstration purposes.  No real crime data is used.

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
