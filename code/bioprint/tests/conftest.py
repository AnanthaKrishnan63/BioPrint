
import pytest



@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A server on a throwaway database. Never touches bioprint.db."""
    monkeypatch.setenv("BIOPRINT_DB", str(tmp_path / "test.db"))
    import importlib

    import db
    importlib.reload(db)
    import server
    importlib.reload(server)
    from fastapi.testclient import TestClient

    return TestClient(server.app)
