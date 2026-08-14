"""Shared helper for models served behind an OpenAI-compatible HTTP endpoint
(vLLM on Modal, etc.). cope_b, cope_a, and safeguard adapters build on this.

Auto-detects chat vs. raw completions from the endpoint URL. Answer parsing:
short-output mode (max_tokens<=4) takes the first 0/1; CoT mode scans for the
last 0/1 in the response (the verdict after reasoning).
"""

from __future__ import annotations

import os

import requests

CoT_THRESHOLD = 4


def parse_answer(raw: str, max_tokens: int) -> str:
    stripped = (raw or "").strip()
    if max_tokens <= CoT_THRESHOLD:
        if stripped.startswith("1"):
            return "1"
        if stripped.startswith("0"):
            return "0"
        return ""
    for ch in reversed(stripped):
        if ch in ("0", "1"):
            return ch
    return ""


def make_http_classifier(prompt_builder, opts: dict):
    """prompt_builder(policy_text, content) -> str (the full prompt)."""
    endpoint = opts["endpoint"]
    model = opts["model"]
    base_max = int(opts.get("max_tokens", 1))
    retry_max = int(opts.get("retry_max_tokens", base_max * 2 if base_max > 4 else base_max))
    timeout = int(opts.get("timeout", 300))
    api_key = os.environ.get("VLLM_API_KEY", "")
    is_chat = endpoint.rstrip("/").endswith("/chat/completions")

    def classify(policy_text: str, content: str, attempt: int = 0):
        prompt = prompt_builder(policy_text, content)
        max_tokens = retry_max if attempt else base_max
        if is_chat:
            body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0}
        else:
            body = {"model": model, "prompt": prompt,
                    "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0}
        resp = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body, timeout=timeout,
        )
        resp.raise_for_status()
        choice = resp.json()["choices"][0]
        raw = (choice["message"]["content"] if is_chat else choice["text"]) or ""
        return parse_answer(raw, max_tokens), raw

    return classify
