"""
The one place where this course decides which LLM it talks to.

Everything else in class 3 calls `client.chat.completions.create(...)` and never needs to
know who is answering. Swapping provider means editing this file and nothing else — which
is the point: a provider is a dependency, and dependencies belong behind a seam.

We use Gemini through its OpenAI-compatible endpoint, so the `openai` SDK works unchanged.
"""

import os

from openai import OpenAI

# Gemini speaks the OpenAI wire format at this address. Point LLM_BASE_URL somewhere else
# and the same code talks to any other OpenAI-compatible provider.
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Pinned on purpose, and a "lite" model on purpose.
#
#   - `gemini-flash-latest` is an alias, and it returned 503 UNAVAILABLE every time
#   - `gemini-2.5-flash` answers 404: "no longer available to new users"
#   - `gemini-3-flash` allows 20 requests PER DAY on the free tier, and one evaluation
#     run is 40 calls, so a student could not complete a single run
#   - this model answered the same prompt in 5.2s where gemini-3.6-flash took 13.0s and
#     still hit the token ceiling
#
# An alias can also change under you between a rehearsal and a class, which is the same
# reason class 2 pins mlflow and the Python version.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Gemini spends tokens thinking before it answers, and they count against this budget.
# At 300 a reply came back cut off after 11 visible tokens, with 289 spent thinking. A
# truncated answer still gets scored by ROUGE, so a budget that is too small does not
# raise an error — it quietly corrupts the evaluation.
#
# 2000 is comfortable for this model, which used 947 tokens total on a structured answer.
# A heavier model may need more: watch for finish_reason == "length".
DEFAULT_MAX_TOKENS = 2000


def get_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY")


def get_model() -> str:
    return os.getenv("LLM_MODEL", DEFAULT_MODEL)


def get_max_tokens() -> int:
    return int(os.getenv("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS))


def get_max_retries() -> int:
    return int(os.getenv("LLM_MAX_RETRIES", 5))


def is_configured() -> bool:
    """True when there is a key to use. Check this before building a client."""
    return bool(get_api_key())


def build_client() -> OpenAI | None:
    """
    Return a ready OpenAI client, or None when no key is configured.

    Returning None rather than raising lets the app start and explain itself. A service
    that refuses to boot because a key is missing tells you nothing; one that boots and
    reports "no key" on /health tells you exactly what to fix.
    """
    api_key = get_api_key()
    if not api_key:
        return None

    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        # The SDK retries twice by default, which is not enough here. A shared model goes
        # through spells of 503 "high demand", and two attempts a second apart routinely
        # fall inside one — killing a live demo at the worst moment. The SDK backs off
        # between attempts and honours Retry-After, so raising this costs nothing when
        # the provider is healthy.
        max_retries=get_max_retries(),
    )


def describe() -> dict:
    """Configuration summary for health endpoints and MLflow params. Never the key."""
    return {
        "provider": "gemini (OpenAI-compatible endpoint)",
        "base_url": os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
        "model": get_model(),
        "max_tokens": get_max_tokens(),
        "api_key_configured": is_configured(),
    }
