"""Model backends. Each returns (text, input_tokens, output_tokens)."""
from __future__ import annotations

import json
import os
import random
import time

from .prompt import SYSTEM, messages, user_message


class Backend:
    name = "base"

    def __call__(self, note: str, gold: dict | None = None) -> tuple[str, int, int]:
        raise NotImplementedError


def _retry(fn, tries=6):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # rate limits, overloads, timeouts
            status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            if status in (400, 401, 403, 404) or i == tries - 1:
                raise
            time.sleep(min(30, 2 ** i) + random.random())


class Anthropic(Backend):
    """Claude via the Anthropic API (needs ANTHROPIC_API_KEY)."""

    def __init__(self, model: str, max_tokens: int = 1024):
        import anthropic
        self.client = anthropic.Anthropic(max_retries=0)
        self.model, self.max_tokens, self.name = model, max_tokens, model

    def __call__(self, note, gold=None):
        r = _retry(lambda: self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=SYSTEM,
            messages=[{"role": "user", "content": user_message(note)}]))
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return text, r.usage.input_tokens, r.usage.output_tokens


class OpenAICompatible(Backend):
    """Any OpenAI-compatible server, e.g. `vllm serve Qwen/Qwen3-8B --enable-lora ...`.

    Qwen3 thinking is switched off so the model answers directly, matching how it was fine-tuned.
    """

    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1", max_tokens: int = 1024,
                 disable_thinking: bool = True):
        import requests
        self.s = requests.Session()
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.max_tokens, self.name = model, max_tokens, model
        self.extra = {"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else {}

    def __call__(self, note, gold=None):
        body = {"model": self.model, "messages": messages(note), "max_tokens": self.max_tokens,
                "temperature": 0.0, **self.extra}

        def go():
            r = self.s.post(self.url, json=body, timeout=300)
            r.raise_for_status()
            return r.json()
        j = _retry(go)
        u = j.get("usage", {})
        return j["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


class Mock(Backend):
    """Noisy oracle for dry runs: returns the gold answer with deliberate mistakes.

    Lets the whole pipeline (scoring, costs, report) run with no GPU and no API key.
    Its scores mean NOTHING about real models.
    """

    def __init__(self, error_rate: float = 0.1, latency: float = 0.0, seed: int = 0, name: str = "mock"):
        self.p, self.latency, self.rng, self.name = error_rate, latency, random.Random(seed), name

    def __call__(self, note, gold=None):
        time.sleep(self.latency)
        g = json.loads(json.dumps(gold))
        r = self.rng
        if r.random() < self.p / 3:
            return "Sorry, I can't parse this note.", 700, 10
        if r.random() < self.p and g["diagnoses"]:
            g["diagnoses"][0]["icd10"] = "R69"
        if r.random() < self.p:
            g["vitals"]["temperature_f"] = round(g["vitals"]["temperature_f"] - 1.5, 1)
        if r.random() < self.p and g["medications"]:
            g["medications"][0]["frequency"] = "once daily" if g["medications"][0]["frequency"] != "once daily" \
                else "twice daily"
        text = json.dumps(g)
        return text, 700 + len(note) // 4, len(text) // 4


def make(spec: str) -> Backend:
    """Backend from a short spec:
         anthropic:claude-haiku-5-5
         openai:<model>@http://host:8000/v1     (vLLM; model is the served name, e.g. 'ft' for the LoRA)
         mock[:error_rate]
    """
    kind, _, rest = spec.partition(":")
    if kind == "anthropic":
        return Anthropic(rest)
    if kind == "openai":
        model, _, url = rest.partition("@")
        return OpenAICompatible(model, url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    if kind == "mock":
        return Mock(float(rest) if rest else 0.1, name=spec)
    raise ValueError(f"unknown backend spec {spec!r}")
