# MLOps / LLMOps — 2026

[![CI](https://github.com/luismrmaia-dev/mlops-class-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/luismrmaia-dev/mlops-class-2026/actions/workflows/ci.yml)

Teaching repository for a two-part course built around one running product: a bonsai
e-commerce site.

- **Class 2 — MLOps.** Stand up a containerised stack, train a classifier that identifies
  a bonsai's species, register it, promote it, and serve it. → [aula2_mlop_infra/](aula2_mlop_infra/)
- **Class 3 — LLMOps.** Take the same product into prompt engineering: BonsAI, an
  assistant that advises on caring for that species. → [aula3_case_study/](aula3_case_study/)

Class 3 extends Class 2 rather than restarting. Class 2 names the species; Class 3 advises
on it.

## Before you read the code

- [CONTEXT.md](CONTEXT.md) — the vocabulary. What a *Classifier*, a *Prompt Mode*, an
  *Evaluation Set* and a *Champion* mean here, and which words to avoid.
- [docs/adr/](docs/adr/) — decisions that are not obvious from the code, and why they were
  made.

## Getting started

```sh
git clone https://github.com/luismrmaia-dev/mlops-class-2026.git
cd mlops-class-2026/aula2_mlop_infra/docker
docker compose up -d --build
```

Each class folder has its own README with the full walkthrough.

## Requirements

Docker and Docker Compose. Nothing else — no local Python, no MLflow, no conda.

## Layout

```
aula2_mlop_infra/       Class 2 — MLOps
├── api/                Serves the champion Classifier
├── docker/             The stack: compose file, images, pinned requirements
└── notebooks/          bonsai_classifier.ipynb

aula3_case_study/       Class 3 — LLMOps
├── api/                BonsAI chat application
├── docker/             The stack
├── notebooks/          Prompt engineering exploration
├── src/                Prompt evaluation pipeline
└── tests/

docs/adr/               Architecture decision records
CONTEXT.md              Domain glossary
```

## Ports

Both classes publish MLflow on 5001 and JupyterLab on 8888, so **run one class at a time**
— bring one stack down before starting the other.

| Service | URL |
|---|---|
| MLflow | http://localhost:5001 |
| JupyterLab | http://localhost:8888 |
| Class 2 API | http://localhost:8080 |
| Class 3 BonsAI | http://localhost:3000 |

## Automation

Every push and pull request runs [`tests/`](tests/), which needs no secrets and therefore
gives the same feedback on a fork as it does here. Two things are checked:

- **Stack rules** — both compose files must define their services, publish host ports that
  do not collide, pin no `container_name`, and keep secrets in `${ENV_VAR}` indirections;
  no Dockerfile may end as root without a documented reason.
- **Notebook hygiene** — notebooks must be committed with their outputs cleared, or the
  diffs become unreadable and the repository grows without bound.

Each of those rules is a lesson from the course. When one fails, the failure message is
the lesson.

You do not have to run them yourself — pushing is enough, and the result appears on your
commit and in the badge above. If you want the answer before pushing, run them the same
way you run everything else in this course, in a container:

```sh
docker run --rm -v "$PWD":/repo -w /repo python:3.13-slim \
  sh -c "pip install -q pytest pyyaml requests && pytest tests/ -v"
```

Still no Python on your machine. The container is thrown away when it finishes.

## License

MIT.
