from fastapi.testclient import TestClient

from concall.api import app

client = TestClient(app)


def test_health_reports_optional_components():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body) >= {"diarization_configured", "llm_configured"}


def test_unknown_call_is_404():
    assert client.get("/calls/NOT_A_CALL/report").status_code == 404
