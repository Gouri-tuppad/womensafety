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

## 📡 API Endpoints

### Safest Route
```http
POST /api/v1/routes/safest
{
  "origin": {"lat": 13.02, "lon": 77.51},
  "destination": {"lat": 13.03, "lon": 77.52},
  "departure_time": "2024-06-15T23:30:00+05:30"  // optional
}
```

### SOS Trigger
```http
POST /api/v1/sos/trigger
{
  "user_id": "uuid",
  "latitude": 13.025,
  "longitude": 77.515
}
```

### Start Trip
```http
POST /api/v1/trips/start
{
  "user_id": "uuid",
  "origin_lat": 13.02, "origin_lon": 77.51,
  "dest_lat": 13.03, "dest_lon": 77.52
}
```

### Dashboard
```http
GET /api/v1/dashboard/stats
GET /api/v1/dashboard/heatmap
GET /api/v1/dashboard/safety-overview
```

## ⚙️ Configuration

All settings via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ZONE` | `peenya` | Active zone: `peenya` or `subramanyapura` |
| `ALPHA_CRIME` | `0.4` | Crime risk weight in safety formula |
| `BETA_DARKNESS` | `0.3` | Darkness risk weight |
| `GAMMA_ISOLATION` | `0.2` | Isolation risk weight |
| `DELTA_ACTIVITY` | `0.1` | Activity score weight (reduces cost) |
| `SOS_COOLDOWN_SECONDS` | `30` | Minimum interval between SOS triggers |
| `TRIP_DEVIATION_METERS` | `200` | Route deviation alert threshold |
| `SMS_PROVIDER` | `mock` | SMS backend (`mock` = no API keys needed) |

## 🗺️ Demo Zones

### Peenya Industrial Area
- **Character**: Factories, warehouses, garment units, long compound walls
- **Risk pattern**: Very empty and dark at night; busy during shift changes (~06-07, 14-15, 22-23)
- **Bbox**: N 13.05, S 13.00, E 77.55, W 77.48

### Subramanyapura
- **Character**: Residential with metro station, Kanakapura Road, interior lanes, vacant plots
- **Risk pattern**: Thin crowds after ~22:00; incidents near vacant plots and lake edges
- **Bbox**: N 12.91, S 12.86, E 77.58, W 77.52

## 📊 Safety Cost Formula

```
cost = distance × clamp(1 + α·CrimeRisk + β·DarknessRisk + γ·IsolationRisk − δ·ActivityScore, min=0.5)
```

- All four inputs normalised to [0, 1]
- Multiplier clamped to minimum 0.5 (a segment can be at most 2× "faster" than raw distance)
- **Safety score** of a route (0–100) = 100 − length-weighted mean risk

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
