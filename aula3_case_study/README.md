# Class 3 — LLMOps and Prompt Engineering

Class 2 built a model that tells you *which* bonsai you have. This class builds **BonsAI**,
the assistant that tells you *how to keep it alive* — and applies the same operational
discipline to prompts that class 2 applied to models.

The shape is deliberately identical:

| | Class 2 | Class 3 |
|---|---|---|
| The thing you version | a trained model | a **Prompt Mode** |
| Where versions live | MLflow Model Registry | MLflow **Prompt** Registry |
| How you compare them | a held-out test set | the **Evaluation Set** |
| How you ship one | move the `@champion` alias | move the `@champion` alias |
| What serves it | the prediction API | the BonsAI chat service |

What changed is the thing being versioned. The discipline did not.

## What you need

- Docker and Docker Compose
- A browser
- A free Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
  — no credit card, no approval process

## Quick start

```bash
cd aula3_case_study/docker
cp .env.example .env        # then paste your key into it
docker compose up -d --build
```

| Service | URL | What it is |
|---|---|---|
| BonsAI | http://localhost:3000 | The chat interface |
| MLflow | http://localhost:5001 | Prompt registry and evaluation results |
| JupyterLab | http://localhost:8888 | Scratch space — optional |

> **Bring class 2's stack down first.** Both classes publish MLflow on 5001 and JupyterLab
> on 8888.

Ask BonsAI something. Then ask it about a houseplant, and watch it refuse — being
specialised is a product requirement here, and it is one of the things we measure.

## The cycle

### 1. Four Prompt Modes

[`src/prompt_modes.py`](src/prompt_modes.py) holds four ways of instructing BonsAI:

| Mode | Approach |
|---|---|
| `basic` | Answer the question conversationally |
| `structured` | Problem, immediate actions, long-term care, prevention |
| `diagnostic` | Work symptoms → causes → diagnosis → treatment |
| `emergency` | Urgent actions first, keep it short |

They live in one file for a reason. The evaluation pipeline and the chat service both
import it, so the modes being *scored* are byte-for-byte the modes being *served*. When
those drift apart, an evaluation proves nothing about what customers receive.

### 2. The Evaluation Set

[`src/evaluation_set.py`](src/evaluation_set.py) is the fixed bar. Six real bonsai
questions with reference answers, and four questions BonsAI must **refuse** — because a
specialist that happily answers about tomatoes is broken, however fluent it sounds.

It is not training data. Nothing is fitted to it. It exists so two Prompt Modes can be
compared on identical input, which means it has to stay fixed: change the ruler and
yesterday's numbers stop meaning anything.

### 3. Evaluate and promote

```bash
docker compose exec bonsai python -m src.evaluate_prompts
```

Each mode is registered as a version of the prompt `bonsai-care`, run over every case, and
scored on:

| Metric | What it catches |
|---|---|
| `key_point_coverage` | Did it say the things that actually matter? |
| `refusal_accuracy` | Did it refuse everything outside bonsai? |
| `rougeL` | Word overlap with the reference answer |
| `actionability` | Can the customer *do* something with this? |
| `truncated_responses` | Did the reply get cut off mid-sentence? |
| `failed_calls` | Did the provider reject the call? |

Compare them in MLflow at http://localhost:5001.

`refusal_accuracy` has a hard gate at 1.0. A mode that answers a question it was told to
refuse is not promoted, however well it scores elsewhere — a confident, fluent, off-topic
answer is worse than no answer.

Add `--promote` to move the alias once a mode clears the gate:

```bash
docker compose exec bonsai python -m src.evaluate_prompts --promote
curl -X POST http://localhost:3000/prompt/reload
```

BonsAI now answers with the new champion. Nothing was rebuilt or redeployed — exactly as
in class 2, where moving a model alias was the deployment.

## Rate limits are part of the problem

The Gemini free tier caps you twice: **5 requests per minute**, and a **daily ceiling per
model**. A full evaluation is 40 calls, so the per-minute limit alone makes a run take
about eight minutes, and the pipeline paces itself to stay under it.

The daily cap is the one that bites. `gemini-3-flash` allows 20 requests a day on the free
tier — fewer than a single evaluation run needs. That is why this class defaults to a
`flash-lite` model: not because it is better, but because you can actually use it twice.

This is worth sitting with. The first version of this pipeline had no pacing, and most
calls came back `429`. A rejected call still produced a score — of zero. The ranking that
came out looked like a statement about prompt quality and was really a statement about
quota, with nothing on the screen to say so.

That is the failure mode to remember from this class: LLM evaluation fails *quietly*. It
does not crash, it returns a number. Which is why the pipeline records `failed_calls` and
`truncated_responses` alongside every score.

Raise the pace if you are on a paid tier:

```bash
docker compose exec bonsai python -m src.evaluate_prompts --rpm 60
```

## Token budgets fail quietly too

Gemini spends tokens *thinking* before it writes anything, and they count against
`max_tokens`. At `max_tokens=300` a reply came back after 11 visible tokens with 289 spent
reasoning, and `finish_reason: length`. The answer was a truncated sentence — which ROUGE
scores perfectly happily.

`LLM_MAX_TOKENS` defaults to 2000 for this reason, and the service logs a warning whenever
a reply hits the ceiling.

## Which model, and why it is pinned

`LLM_MODEL` defaults to `gemini-3.5-flash-lite`. Measured while preparing this class:

| Model | Result |
|---|---|
| `gemini-flash-latest` | `503 UNAVAILABLE`, every attempt |
| `gemini-2.5-flash` | `404` — *"no longer available to new users"* |
| `gemini-3-flash` | 20 requests per day on the free tier — one run needs 40 |
| `gemini-3.6-flash` | 13.0s, and still truncated at `max_tokens=2000` |
| `gemini-3.5-flash-lite` | 5.2s, finished cleanly in 947 tokens |

Aliases like `-latest` move. Pin the model for the same reason class 2 pins `mlflow` and
the Python version: what you rehearsed must be what runs.

## How the provider is wired

[`src/llm_client.py`](src/llm_client.py) is the only file that knows who answers. Gemini
speaks the OpenAI wire format, so the `openai` SDK is used unchanged and only the base URL
differs. Swapping provider means editing that one file.

## Layout

```
aula3_case_study/
├── api/bonsai_app.py          # Chat service; serves prompts:/bonsai-care@champion
├── docker/
│   ├── docker-compose.yml     # mlflow + jupyter + bonsai
│   ├── Dockerfile.api
│   ├── requirements.txt       # pinned
│   └── .env.example           # copy to .env and add your key
├── src/
│   ├── llm_client.py          # the only file that knows the provider
│   ├── prompt_modes.py        # the four modes, shared by pipeline and service
│   ├── evaluation_set.py      # the fixed bar
│   └── evaluate_prompts.py    # score, rank, gate, promote
└── tests/
```

## Secrets

`.env` is gitignored, and `docker-compose.yml` only ever names the variable that holds
your key, never the key itself. Never commit it, never paste it into a chat window. Anyone
who obtains it can spend your quota.

## Troubleshooting

**BonsAI answers "I'm not configured right now"** — `GEMINI_API_KEY` is missing. It must
be in `docker/.env`, next to the compose file; Compose does not read a `.env` from the
repository root.

```bash
curl -s http://localhost:3000/health
```

**`429 RESOURCE_EXHAUSTED`** — the free tier's 5 requests per minute. Lower `--rpm`, or
wait a minute.

**`503 UNAVAILABLE`** — the model is overloaded, not your key. Try
`LLM_MODEL=gemini-3.6-flash` in `.env`.

**BonsAI ignores a newly promoted champion** — it caches the prompt after loading it:

```bash
curl -X POST http://localhost:3000/prompt/reload
```

**"Invalid Host header - possible DNS rebinding attack detected"** — MLflow rejects Host
headers it does not recognise. `init-mlflow.sh` passes `--allowed-hosts`; if you changed
the service name in the compose file, add it there too.

**Start over**

```bash
docker compose down -v
rm -rf ../db ../artifacts
docker compose up -d --build
```
