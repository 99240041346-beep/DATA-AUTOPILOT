# DATA AUTOPILOT

Future-facing autonomous Data Science workspace: upload data, profile it, explore patterns, prepare ML experiments, explain findings, and test what-if scenarios.

## Stack
Next.js + TypeScript + Tailwind + Recharts; FastAPI + pandas + scikit-learn.

## Run
npm install && npm run dev

Optional API: cd services/api && pip install -r requirements.txt && uvicorn main:app --reload --port 8000

## Production configuration

Set these environment variables when deploying the API:

- `ALLOWED_ORIGINS`: comma-separated frontend origins; defaults to `http://localhost:3000`.
- `ANALYSIS_API_URL`: FastAPI base URL used by the Next.js API proxies.
- `MAX_ROWS`: maximum rows accepted by analysis endpoints; defaults to `50000`.
- `MAX_COLUMNS`: maximum columns accepted; defaults to `200`.
- `AUTOPILOT_DB_PATH`: SQLite database path; defaults to `autopilot.db`.

The API exposes `/health` for liveness and `/ready` for database readiness. Render is configured to use `/health`.

## Test

From `services/api`:

```bash
pip install -r requirements.txt
pytest -q
```
