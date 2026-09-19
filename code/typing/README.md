# Typing capture

First brick of the account-holder-verification harness.
Research notes: `~/.yoyo/arshad_mfsdsai/big_idea/`

## Run

```bash
conda activate bigidea
uvicorn server:app --reload
```

Then open <http://localhost:8000>. Ctrl-C to stop.

## Test

```bash
node --test          # metrics.js only - pure functions, no browser needed
```

## Files

| File | Role | Lifespan |
|---|---|---|
| `capture.js` | Records key events. Nothing else. | becomes the real collector |
| `metrics.js` | Pure functions: events to numbers. No DOM. | grows into the feature vector |
| `app.js` | Wires them to the page. | throwaway UI |
| `index.html` | Structure + styles. | throwaway UI |
| `server.py` | FastAPI: serves files, accepts `POST /api/session`. | |
| `sessions.db` | SQLite. Created on first run. | |

## Data

`keystrokes` holds the raw event log and is the record of truth.
`session_metrics` is a convenience copy of what the page displayed.
Feature extraction will change as the research moves on; recompute from
`keystrokes` rather than re-collecting from people.

Only `event.code` (which physical key) is stored, never `event.key` (which
character), and never the contents of the text box.

Inspect the data from Python:

```python
import sqlite3, pandas as pd
con = sqlite3.connect('sessions.db')
pd.read_sql_query('SELECT * FROM session_metrics', con)
pd.read_sql_query('SELECT * FROM keystrokes WHERE session_id = 1', con)
```

## Environment

```bash
conda env create -f environment.yml    # first time only
```
