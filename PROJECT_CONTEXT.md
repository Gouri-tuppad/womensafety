PROJECT: "SafeWay" backend. A safest-route navigation API for women and children. It returns the SAFEST route instead of the fastest, using crime history, street lighting, crowd/activity and time of day. It includes a shake-gesture SOS, live trip monitoring and an authority dashboard API. Backend only, no mobile app or frontend.

STACK: Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2 + Alembic, PostgreSQL + PostGIS, Redis, OSMnx + NetworkX, XGBoost, scikit-learn, SHAP, pandas, numpy, H3, astral, holidays, pytest, Docker + docker-compose. SMS/push/LLM providers sit behind interfaces with MOCK implementations so everything runs without API keys.

STRUCTURE:
app/ main.py, config.py, deps.py, api/, core/, models/, schemas/, services/, ml/, data/
tests/, docker-compose.yml, Dockerfile, Makefile, README.md, requirements.txt, .env.example

DEMO LOCATION: Bengaluru, Karnataka, India. Two zone presets in config.py, selectable via env ZONE:
- `peenya`: Peenya Industrial Area, Peenya 1st/2nd/3rd/4th Stage, Jalahalli Cross, Dasarahalli, Peenya metro. Approx bbox N 13.05, S 13.00, E 77.55, W 77.48
- `subramanyapura`: Subramanyapura, Subramanyapura metro, Kanakapura Road stretch, Vasanthapura, Uttarahalli. Approx bbox N 12.91, S 12.86, E 77.58, W 77.52
Bounding boxes are approximate: verify against OpenStreetMap and adjust. Default ZONE=peenya. TIMEZONE=Asia/Kolkata. Compute sunset with `astral` for each zone's centre. Holidays: `holidays` lib, country IN, subdiv KA.

CONTEXT TO MODEL REALISTICALLY:
- Peenya is a large industrial belt: factories, warehouses, garment units. It is busy during shift changes (about 6-7 AM, 2-3 PM, 10-11 PM), then very empty and dark at night, with long compound walls and few shops. Many women workers commute at shift-change hours.
- Subramanyapura is mostly residential with a metro station, Kanakapura Road as the main artery, interior lanes, vacant plots and lake/park edges, and thin crowds after about 10 PM.

SAFETY COST FORMULA (use exactly):
cost = distance * (1 + alpha*CrimeRisk + beta*DarknessRisk + gamma*IsolationRisk - delta*ActivityScore)
All four inputs normalized 0-1; clamp the multiplier to a minimum of 0.5.
Safety score of a route (0-100) = 100 minus the length-weighted mean risk.

RULES: typed, documented code; config via env vars; no secrets in code; clearly label every simulated or synthetic data source; at the end of each step, run the code and its tests, and fix failures before reporting done. Make sensible assumptions and document them rather than asking questions.







Using the project context, build STEP 1 only: infrastructure and foundation.

1. docker-compose with services: api (FastAPI), db (postgis/postgis), redis. Dockerfile, requirements.txt, .env.example, Makefile with targets: up, down, migrate, seed, train, test.
2. config.py using pydantic-settings: DB/Redis URLs, ZONE, the zone presets (bbox, centre, name) for `peenya` and `subramanyapura`, JWT secret, SOS countdown seconds (default 5), volunteer radius (default 1000 m), alpha/beta/gamma/delta defaults, provider switches (SMS_PROVIDER=mock|twilio, PUSH_PROVIDER=mock|fcm, LLM_PROVIDER=mock|openai|anthropic).
3. SQLAlchemy models + Alembic migration (PostGIS geometry columns with spatial indexes):
   users (role: woman/child/parent/volunteer/authority, trust_score, profile_type), trusted_contacts, road_segments (osm_id, geometry LineString, road_type, length_m, lit_tag, lamp_count, lit_ratio, zone), pois (category: lamp/shop/atm/petrol/bus_stop/metro/pharmacy/factory_gate; safe_haven bool), incidents (geometry Point, type, severity, timestamp, source: official|news|user|synthetic), reports, trips, trip_locations, sos_events (status: COUNTDOWN/ACTIVE/CANCELLED/RESOLVED, trigger_type, confidence), sos_locations, evidence_files, geofences, volunteers, audit_log, feedback.
4. Auth: register, login (JWT), role-based dependencies, trusted contacts CRUD, profile endpoints. Basic rate limiting middleware.
5. GET /health endpoint, which also checks DB and Redis.

Acceptance: `docker compose up` starts everything, migrations apply, /docs shows the endpoints, and pytest covers auth and the health check.