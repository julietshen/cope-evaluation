"""mistralai/Shieldstral-1.0-3B — multimodal (image) variant of the shieldstral
adapter. The `content` passed by the harness is an **image file path** (or a
local path to a downloaded Bluesky blob); the policy is text in <Instruct> and
the Document being judged is the image.

Same single-forward-pass scoring as the text adapter: yes/no token logits at the
final position are renormalized into a 0-1 score, thresholded at 0.5. Needs
torchvision (Pixtral image processor). Runs sequentially.
"""
from __future__ import annotations
import math

NAME = "Shieldstral-1.0-3B-mm"
SUPPORTS_CONCURRENCY = False

MODEL_ID = "mistralai/Shieldstral-1.0-3B"
SYSTEM = ("Judge whether the Document meets the requirements based on the Query "
          "and the Instruction provided.")
QUERY = "Does the image Document violate the policy?"
INSTRUCT_PREAMBLE = ("You are a content safety classifier for an online platform. "
                     "Judge the image Document strictly according to the policy below.\n\n")


def _yes_no_ids(tok):
    ids = {"yes": set(), "no": set()}
    for word in ("yes", "no"):
        for v in (word, word.capitalize(), word.upper()):
            for prefix in ("", " "):
                enc = tok.encode(prefix + v, add_special_tokens=False)
                if len(enc) == 1:
                    ids[word].add(enc[0])
    assert ids["yes"] and ids["no"], "no single-token yes/no variants found"
    return sorted(ids["yes"]), sorted(ids["no"])


def make_classifier(opts: dict):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    model_id = opts.get("model", MODEL_ID)
    query = opts.get("query", QUERY)
    threshold = float(opts.get("threshold", 0.5))
    if "device" in opts:
        device = opts["device"]
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    proc = AutoProcessor.from_pretrained(model_id)
    tok = proc.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    YES, NO = _yes_no_ids(tok)

    def classify(policy_text: str, content: str, attempt: int = 0):
        instruct = INSTRUCT_PREAMBLE + policy_text
        try:
            img = Image.open(content.strip()).convert("RGB")
        except Exception as e:
            return "", f"ERROR loading image {content!r}: {e}"
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "text", "text": f"<Instruct>: {instruct}\n\n<Query>: {query}\n\n<Document>:"},
                {"type": "image"},
            ]},
        ]
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
        inputs = proc(text=prompt, images=[img], return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1].float()
        z_yes = max(logits[j].item() for j in YES)
        z_no = max(logits[j].item() for j in NO)
        score = math.exp(z_yes) / (math.exp(z_yes) + math.exp(z_no))
        return ("1" if score >= threshold else "0"), f"score={score:.4f}"

    return classify
