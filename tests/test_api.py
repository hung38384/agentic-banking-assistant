from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_endpoint_uses_tool_and_returns_schema():
    response = client.post(
        "/chat",
        headers={"X-API-Key": "dev-secret"},
        json={
            "conversation_id": "test-chat",
            "message": "Calculate a 100000 VND loan at 6 percent for 24 months",
            "user_profile": {},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "loan_calculation"
    assert body["tool_calls"][0]["name"] == "loan_calculator"
    assert "monthly_payment" in body["tool_calls"][0]["output"]

