"""
Rules that both class stacks must obey.

These read the compose files and Dockerfiles from disk. Nothing has to be running, so
they finish in under a second and can run on every pull request.

Each test here encodes a lesson from the course. If one of them fails on your branch, the
failure message is the lesson.
"""

import os

import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (name, compose file, services it must define, its Dockerfiles)
STACKS = [
    (
        "aula2",
        "aula2_mlop_infra/docker/docker-compose.yml",
        {"mlflow", "jupyter", "api"},
        ["aula2_mlop_infra/docker/Dockerfile", "aula2_mlop_infra/docker/Dockerfile.jupyter"],
    ),
    (
        "aula3",
        "aula3_case_study/docker/docker-compose.yml",
        {"mlflow", "jupyter", "bonsai"},
        ["aula3_case_study/docker/Dockerfile.api"],
    ),
]

STACK_IDS = [s[0] for s in STACKS]

# Dockerfiles allowed to finish as root, and why. Anything not listed here must end as an
# unprivileged user. An exemption needs a reason someone can go and read.
ROOT_USER_EXEMPTIONS = {
    "aula2_mlop_infra/docker/Dockerfile.jupyter": (
        "starts as root only so its entrypoint can match the container user to the host's "
        "uid on the bind mount, then drops privileges before JupyterLab runs — "
        "see docs/adr/0003-jupyter-adopts-host-uid.md"
    ),
}


def load(compose_path):
    with open(os.path.join(REPO_ROOT, compose_path)) as f:
        return yaml.safe_load(f)


@pytest.fixture(params=STACKS, ids=STACK_IDS)
def stack(request):
    name, compose_path, services, dockerfiles = request.param
    return {
        "name": name,
        "path": compose_path,
        "config": load(compose_path),
        "expected_services": services,
        "dockerfiles": dockerfiles,
    }


def test_compose_is_valid_yaml(stack):
    """A stack nobody can parse is a stack nobody can run."""
    assert stack["config"] is not None, f"{stack['path']} is empty or invalid"
    assert "services" in stack["config"], f"{stack['path']} defines no services"


def test_expected_services_are_defined(stack):
    defined = set(stack["config"]["services"])
    missing = stack["expected_services"] - defined
    assert not missing, f"{stack['path']} is missing services: {sorted(missing)}"


def test_host_ports_do_not_collide(stack):
    """Two services cannot publish the same host port — the second one fails to start."""
    seen = {}
    for service_name, service in stack["config"]["services"].items():
        for mapping in service.get("ports", []):
            host_port = str(mapping).split(":")[0]
            assert host_port not in seen, (
                f"host port {host_port} is published by both '{seen[host_port]}' and "
                f"'{service_name}' in {stack['path']}"
            )
            seen[host_port] = service_name


def test_no_pinned_container_names(stack):
    """
    container_name makes a name global to the Docker daemon, so the other class's stack
    refuses to start while this one is up. Letting Compose derive names per project keeps
    the two independent.
    """
    pinned = [
        name
        for name, service in stack["config"]["services"].items()
        if "container_name" in service
    ]
    assert not pinned, f"{stack['path']} pins container_name on: {pinned}"


def test_secrets_come_from_the_environment(stack):
    """An API key belongs in .env, which is gitignored — never in a tracked file."""
    with open(os.path.join(REPO_ROOT, stack["path"])) as f:
        for number, line in enumerate(f, start=1):
            if "API_KEY" in line and "${" not in line:
                pytest.fail(
                    f"{stack['path']}:{number} looks like a hardcoded secret: {line.strip()}"
                )


def test_dockerfiles_do_not_end_as_root(stack):
    """
    The last USER instruction decides who the container runs as. Ending as root means
    every process inside runs as root.
    """
    for dockerfile in stack["dockerfiles"]:
        full_path = os.path.join(REPO_ROOT, dockerfile)
        assert os.path.exists(full_path), f"{dockerfile} is referenced but missing"

        with open(full_path) as f:
            users = [
                line.split(maxsplit=1)[1].strip()
                for line in f
                if line.strip().upper().startswith("USER ")
            ]

        if not users:
            continue  # inherits the base image's user

        if users[-1].lower() == "root":
            assert dockerfile in ROOT_USER_EXEMPTIONS, (
                f"{dockerfile} ends as root with no documented reason. If that is "
                f"deliberate, add it to ROOT_USER_EXEMPTIONS with an explanation."
            )


def test_no_dockerfile_uses_insecure_add(stack):
    """ADD with a URL fetches over the network at build time, unverified."""
    for dockerfile in stack["dockerfiles"]:
        with open(os.path.join(REPO_ROOT, dockerfile)) as f:
            content = f.read().upper()
        assert "ADD HTTP" not in content, f"{dockerfile} uses ADD with a URL; use COPY"
