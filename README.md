# PHARAOH GUARD

Crowd-risk prediction + guidance for Egyptian heritage sites — FastAPI +
LangGraph multi-agent pipeline (Monitoring → Reasoning → Recommendation →
Dispatch), backed by a trained scikit-learn risk classifier.

Repo layout:
```
Pharaoh_Guard/
├── datascience/     risk_model.joblib, feature_manifest.json, dataset_summary.json (site registry)
├── notebooks/       01 data generation, 02 cleaning
└── src/             the FastAPI app — run everything from inside this folder
```

> **Everything below is run from inside `src/`** — the Python packages
> (`api`, `core`, `agents`, `config`, ...) live at `src/`, not at the repo
> root, so `pip install`, `uvicorn`, and `streamlit` all need `src/` as the
> working directory.

## 1. Setup FastAPI Environment

```bash
cd src
python -m venv pharaoh-guard
```

Activate it (`source pharaoh-guard/bin/activate` on macOS/Linux,
`pharaoh-guard\Scripts\activate` on Windows).

Install the backend dependencies:

```bash
pip install -r requirements.txt
```

Create the environment file by copying the example:

### Windows
```bash
copy .env.example .env
```

### macOS / Linux
```bash
cp .env.example .env
```

Open `.env` and add your **Groq API key**. Note the `RRS_` prefix — every
setting is read as `RRS_<FIELD_NAME>` (see `config/settings.py`):

```env
RRS_GROQ_API_KEY=your_groq_api_key
```

The risk model path (`RRS_RISK_MODEL_PATH`) does **not** need to be set —
it defaults to `../datascience/risk_model.joblib` relative to `src/`.

Start the FastAPI server (still from inside `src/`):

```bash
uvicorn api.main:app --reload
```

Quick smoke test once it's up:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-12-27T11:00:00",
    "site_name": "Giza Pyramids",
    "current_visitors": 17250,
    "security_staff": 40,
    "medical_team": 6,
    "police_units": 4,
    "temperature": 19,
    "humidity": 50,
    "wind_speed": 12,
    "visibility": 10,
    "queue_length": 2600,
    "queue_time": 95
  }'
```
Expected: `"risk_level": "Critical"`, `"proba_critical"` around `0.9997`.

---

## 2. Setup Streamlit Environment

Open a **new terminal**, from inside `src/` again.

```bash
python -m venv pharaoh-streamlit
```

Activate it, then:

```bash
pip install streamlit==1.39.0
```

Run the dashboard:

```bash
streamlit run streamlit_app.py
```

---

## Notes

- Start the FastAPI server before launching the Streamlit dashboard.
- The backend and Streamlit use **separate virtual environments**.
- Make sure `RRS_GROQ_API_KEY` is set in `.env` before running the server —
  Monitoring still runs without it, but Reasoning/Recommendation (LLM
  calls) will fail on escalated (High/Critical) situations.
- `POST /predict` only needs *live* telemetry (visitors, staffing, weather,
  queue) — site capacity, gates, sensitivity, and all engineered model
  features are derived internally from `datascience/dataset_summary.json`'s
  site registry and `core/risk_model.py`. See that module's docstring for
  the full feature list if you need to extend it.
- `POST /predict/batch` scores a list of readings (e.g. all 8 sites for
  one simulated hour) concurrently in one request — this is what the
  Streamlit "Run Full-Day Simulation" button uses. `/predict/manual` has
  been removed now that `/predict` is the trusted, model-backed path.
- `POST /chat/admin` (`{"message": "...", "session_id": "..."}`) is the
  admin chat assistant: it can look up any site's live status, search the
  emergency protocol library, pull the daily report, or — only when
  explicitly asked — dispatch a real operational action through the same
  `tools/dispatch_tools.ACTION_REGISTRY` the automatic pipeline uses.
  Conversation history is kept in memory per `session_id` (not persisted
  across restarts); `POST /chat/admin/clear?session_id=...` resets it.
