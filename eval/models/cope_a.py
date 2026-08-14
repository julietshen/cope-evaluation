"""zentropi-ai/cope-a-9b — Gemma-2-9B + LoRA adapter. No chat template, so it
must use /v1/completions (raw). Emits a single 0/1 token. 8k context ceiling —
policies over ~8k tokens (e.g. very_long) are physically incompatible."""

from ._openai_http import make_http_classifier
from .cope_b import PROMPT_TEMPLATE

NAME = "cope-a-9b"
SUPPORTS_CONCURRENCY = True

DEFAULTS = {
    "endpoint": "https://juliet--cope-a-9b-serve.modal.run/v1/completions",
    "model": "cope-a",
    "max_tokens": "1",
}


def make_classifier(opts: dict):
    o = {**DEFAULTS, **opts}
    return make_http_classifier(
        lambda policy, content: PROMPT_TEMPLATE.format(policy=policy, content=content), o)
