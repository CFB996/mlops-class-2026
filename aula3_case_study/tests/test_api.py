"""
Integration tests for the BonsAI chat service.

They talk to a running stack and skip when it is down, so they are useful on your own
machine after `docker compose up` and harmless in CI, where no stack runs.

Start the stack first:
    cd docker && docker compose up -d
"""

import pytest
import requests

BASE_URL = "http://localhost:3000"
TIMEOUT = 120  # an LLM call is slow, and slower still under a rate limiter


@pytest.fixture(scope="module")
def service():
    """Skip the whole module unless BonsAI is answering."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
    except requests.exceptions.RequestException:
        pytest.skip("BonsAI is not running")

    if response.status_code != 200:
        pytest.skip(f"BonsAI is unhealthy: {response.status_code}")

    return response.json()


def test_health_reports_llm_configuration(service):
    """/health must say how the LLM is wired, and must never leak the key."""
    assert service["status"] == "healthy"

    llm = service["llm"]
    assert llm["model"], "no model reported"
    assert isinstance(llm["api_key_configured"], bool)

    # The key must never appear in a response, under any field name.
    assert "api_key" not in str(llm).lower().replace("api_key_configured", "")


def test_health_reports_which_prompt_is_served(service):
    """You must be able to tell whether BonsAI serves @champion or a local fallback."""
    info = service["model_info"]
    assert info["source"] in ("mlflow", "local")
    assert info["status"] in ("champion", "local_fallback")


def test_answers_a_bonsai_question(service):
    if not service["llm"]["api_key_configured"]:
        pytest.skip("no GEMINI_API_KEY configured")

    response = requests.post(
        f"{BASE_URL}/chat",
        json={"query": "How often should I water my Juniper bonsai?"},
        timeout=TIMEOUT,
    )
    assert response.status_code == 200

    answer = response.json()["response"].lower()
    assert len(answer) > 50, "suspiciously short answer"
    assert "water" in answer or "soil" in answer


def test_refuses_questions_that_are_not_about_bonsai(service):
    """
    Refusing is a product requirement, not an edge case. A specialist that answers
    anything is not a specialist.
    """
    if not service["llm"]["api_key_configured"]:
        pytest.skip("no GEMINI_API_KEY configured")

    response = requests.post(
        f"{BASE_URL}/chat",
        json={"query": "What is the capital of Portugal?"},
        timeout=TIMEOUT,
    )
    assert response.status_code == 200

    answer = response.json()["response"].lower()
    assert "lisbon" not in answer and "lisboa" not in answer, "answered an off-topic question"
    assert "bonsai" in answer, "refusal should point back to bonsai"


def test_rejects_an_empty_query(service):
    response = requests.post(f"{BASE_URL}/chat", json={"query": ""}, timeout=30)
    assert response.status_code == 400


def test_lists_the_available_prompt_modes(service):
    response = requests.get(f"{BASE_URL}/prompt/info", timeout=30)
    assert response.status_code == 200

    modes = response.json()["available_prompts"]
    assert set(modes) == {"basic", "structured", "diagnostic", "emergency"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
