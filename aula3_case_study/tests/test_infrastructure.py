"""
Connectivity checks for the class 3 stack.

These talk to live services and skip when nothing is running, so they are useful on your
own machine after `docker compose up` and harmless in CI.

The checks that read the compose file and need nothing running live in the repository's
top-level tests/ directory, because they apply to both classes.
"""

import time

import pytest
import requests

# Host ports, as published by docker/docker-compose.yml.
MLFLOW_URL = "http://localhost:5001"
BONSAI_URL = "http://localhost:3000"
JUPYTER_URL = "http://localhost:8888"

SERVICES = {
    "mlflow": f"{MLFLOW_URL}/health",
    "bonsai": f"{BONSAI_URL}/health",
    "jupyter": f"{JUPYTER_URL}/api",
}


@pytest.fixture(scope="module")
def services_ready():
    """Report which services answered before their deadline."""
    ready = {}

    # Each service gets its own budget, so a slow one cannot starve the others.
    for name, health_url in SERVICES.items():
        deadline = time.time() + 30
        ready[name] = False

        while time.time() < deadline:
            try:
                if requests.get(health_url, timeout=5).status_code == 200:
                    ready[name] = True
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

    return ready


@pytest.mark.parametrize("service", sorted(SERVICES))
def test_service_is_healthy(service, services_ready):
    if not services_ready.get(service):
        pytest.skip(f"{service} is not running")

    assert requests.get(SERVICES[service], timeout=10).status_code == 200


def test_stack_is_usable(services_ready):
    """The stack only does anything useful when MLflow and BonsAI are both up."""
    if not any(services_ready.values()):
        pytest.skip("no services running")

    assert services_ready.get("mlflow"), "MLflow is not reachable"
    assert services_ready.get("bonsai"), "BonsAI is not reachable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
