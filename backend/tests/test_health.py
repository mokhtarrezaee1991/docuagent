from fastapi.testclient import TestClient

from app.main import app


def test_health():
    # Sendet eine Anfrage an die Anwendung ohne eigenen Webserver.
    with TestClient(app) as client:
        response = client.get("/health")

    # Prueft den Statuscode und die genaue JSON-Antwort.
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "DocuAgent"}
