# BioPrint application

The final typing-pointer app is on `main`. Follow the repository-root
[README](../../README.md) for installation, model details, evidence and privacy notes.

From the repository root, after activating the Conda environment:

```bash
python launch.py
```

Open http://localhost:8000. This launcher configures the bundled trained pointer
encoder and typing calibration bank, keeps local data in `runtime/`, and serves
only loopback. Running bare `uvicorn server:app` bypasses that configuration.

`server.py` provides the API; `db.py` owns SQLite persistence; `engine/` contains
feature extraction and scoring; `pointer_neural.py` integrates the frozen encoder;
`static/` contains the browser app. `tests/` includes isolated API and scoring tests.
Research API modules are optional and are not mounted by the normal login server.
