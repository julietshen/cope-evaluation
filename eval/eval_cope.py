"""
Evaluate zentropi-ai/cope-b-a4b on the self-harm test set across multiple
policy prompts of varying detail.

Usage:
    export VLLM_API_KEY=sk-...                       # the key you set in cope-secrets
    python eval_cope.py                              # full run
    python eval_cope.py --limit 5                    # smoke test on 5 rows
    python eval_cope.py --policies minimal simple    # subset of policies

Outputs:
    results/predictions_<timestamp>.csv  - wide table: one row per content sample
    results/summary_<timestamp>.csv      - per-policy precision/recall/F1
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

DEFAULT_ENDPOINT = "https://juliet--cope-b-a4b-serve.modal.run/v1/chat/completions"
DEFAULT_MODEL = "zentropi-ai/cope-b-a4b"
HERE = Path(__file__).parent
POLICIES_DIR = HERE / "policies"
RESULTS_DIR = HERE / "results"
DEFAULT_TEST_SET = HERE / "test_set.csv"

PROMPT_TEMPLATE = """INSTRUCTIONS
============

Examine the given POLICY and determine if the given CONTENT meets the criteria for ANY of the LABELS. Answer "1" if yes, and "0" if no.


POLICY
======

{policy}


CONTENT
=======

{content}


ANSWER
======

"""


def load_policies(names: list[str]) -> dict[str, str]:
    out = {}
    for name in names:
        path = POLICIES_DIR / f"{name}.md"
        if not path.exists():
            sys.exit(f"missing policy file: {path}")
        out[name] = path.read_text()
    return out


def load_test_set(path: Path, limit: int | None) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    # Drop rows missing a ground_truth — supports labelling-in-progress CSVs
    rows = [r for r in rows if r.get("ground_truth", "").strip() in ("0", "1")]
    return rows[:limit] if limit else rows


def call_cope(policy: str, content: str, api_key: str, endpoint: str, model: str,
              max_tokens: int = 1, timeout: int = 300) -> tuple[str, str]:
    prompt = PROMPT_TEMPLATE.format(policy=policy, content=content)
    is_chat = endpoint.rstrip("/").endswith("/chat/completions")
    if is_chat:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0}
    else:
        body = {"model": model, "prompt": prompt,
                "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0}
    resp = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=timeout,
    )
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    raw = (choice["message"]["content"] if is_chat else choice["text"]) or ""
    # Cope: max_tokens=1 returns "0" or "1" directly.
    # Reasoning models (e.g. gpt-oss-safeguard): output is CoT followed by final answer.
    # Take the LAST "0" or "1" appearing in the response — that's the answer after reasoning.
    stripped = raw.strip()
    pred = ""
    if max_tokens <= 4:
        # Short-output mode: first 0/1 token wins.
        if stripped.startswith("1"): pred = "1"
        elif stripped.startswith("0"): pred = "0"
    else:
        # CoT mode: scan for last standalone 0 or 1 character.
        for ch in reversed(stripped):
            if ch in ("0", "1"):
                pred = ch
                break
    return pred, raw


def warmup(api_key: str, endpoint: str, model: str, max_tokens: int = 1) -> None:
    print(f"warming up {endpoint} (cold start can take 2-3 min)...", flush=True)
    t0 = time.time()
    pred, raw = call_cope("Flag harmful content.", "hello", api_key, endpoint, model,
                          max_tokens=max_tokens, timeout=600)
    print(f"  warm in {time.time()-t0:.1f}s, pred={pred!r}, raw={raw[:200]!r}", flush=True)


def run_one(args):
    row_id, content, policy_name, policy_text, api_key, endpoint, model, max_tokens = args
    try:
        pred, raw = call_cope(policy_text, content, api_key, endpoint, model, max_tokens=max_tokens)
        return row_id, policy_name, pred, raw, None
    except Exception as e:
        return row_id, policy_name, "", "", str(e)


def metrics(preds: list[str], truth: list[str]) -> dict:
    tp = sum(1 for p, t in zip(preds, truth) if p == "1" and t == "1")
    fp = sum(1 for p, t in zip(preds, truth) if p == "1" and t == "0")
    fn = sum(1 for p, t in zip(preds, truth) if p == "0" and t == "1")
    tn = sum(1 for p, t in zip(preds, truth) if p == "0" and t == "0")
    errors = sum(1 for p in preds if p == "")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(len(preds) - errors, 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, errors=errors,
                precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET,
                    help="path to CSV with columns id, content, ground_truth")
    ap.add_argument("--label", default=None,
                    help="optional label prefix for output files (e.g., 'sex' or 'selfharm')")
    ap.add_argument("--limit", type=int, default=None, help="limit number of rows")
    ap.add_argument("--policies", nargs="+", default=["minimal", "simple", "medium", "full"])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--skip-warmup", action="store_true")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                    help="vLLM /v1/completions or /v1/chat/completions URL (default: cope-b-a4b on Modal)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="model name in request body (cope-b: 'zentropi-ai/cope-b-a4b'; cope-a LoRA: 'cope-a')")
    ap.add_argument("--max-tokens", type=int, default=1,
                    help="max output tokens (1 for cope; bump to 2048 for reasoning models like gpt-oss-safeguard)")
    args = ap.parse_args()

    api_key = os.environ.get("VLLM_API_KEY")
    if not api_key:
        sys.exit("set VLLM_API_KEY env var (the key you used in `modal secret create cope-secrets`)")

    policies = load_policies(args.policies)
    rows = load_test_set(args.test_set, args.limit)
    print(f"loaded {len(rows)} test rows from {args.test_set}, {len(policies)} policies: {list(policies)}")

    if not args.skip_warmup:
        warmup(api_key, args.endpoint, args.model, args.max_tokens)

    # Build (row_id, content, policy_name, policy_text, ...) jobs
    jobs = []
    for row in rows:
        for pname, ptext in policies.items():
            jobs.append((row["id"], row["content"], pname, ptext, api_key, args.endpoint, args.model, args.max_tokens))

    print(f"firing {len(jobs)} requests with concurrency={args.concurrency}...", flush=True)
    results: dict[tuple[str, str], tuple[str, str, str | None]] = {}
    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(run_one, j) for j in jobs]
        for fut in as_completed(futures):
            row_id, pname, pred, raw, err = fut.result()
            results[(row_id, pname)] = (pred, raw, err)
            completed += 1
            if completed % 25 == 0 or completed == len(jobs):
                elapsed = time.time() - t0
                rate = completed / elapsed
                print(f"  {completed}/{len(jobs)} ({rate:.1f}/s, {elapsed:.0f}s elapsed)", flush=True)

    # Write predictions CSV
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"{args.label}_{ts}" if args.label else ts
    pred_path = RESULTS_DIR / f"predictions_{suffix}.csv"

    policy_names = list(policies)
    header = ["id", "content", "ground_truth"]
    for p in policy_names:
        header += [f"{p}_pred", f"{p}_raw"]
    header += [f"{p}_correct" for p in policy_names]

    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            line = [row["id"], row["content"], row["ground_truth"]]
            for p in policy_names:
                pred, raw, err = results[(row["id"], p)]
                line += [pred, raw if not err else f"ERROR: {err}"]
            for p in policy_names:
                pred, _, _ = results[(row["id"], p)]
                line.append("1" if pred == row["ground_truth"] else "0")
            w.writerow(line)
    print(f"\nwrote predictions: {pred_path}")

    # Write summary CSV
    summary_path = RESULTS_DIR / f"summary_{suffix}.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "n", "tp", "fp", "fn", "tn", "errors",
                    "precision", "recall", "f1", "accuracy"])
        print(f"\n{'policy':10s} {'n':>4s} {'tp':>4s} {'fp':>4s} {'fn':>4s} {'tn':>4s} "
              f"{'err':>4s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'acc':>6s}")
        print("-" * 70)
        for p in policy_names:
            preds = [results[(row["id"], p)][0] for row in rows]
            truth = [row["ground_truth"] for row in rows]
            m = metrics(preds, truth)
            w.writerow([p, len(rows), m["tp"], m["fp"], m["fn"], m["tn"], m["errors"],
                        f"{m['precision']:.3f}", f"{m['recall']:.3f}",
                        f"{m['f1']:.3f}", f"{m['accuracy']:.3f}"])
            print(f"{p:10s} {len(rows):>4d} {m['tp']:>4d} {m['fp']:>4d} {m['fn']:>4d} {m['tn']:>4d} "
                  f"{m['errors']:>4d} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                  f"{m['f1']:>6.3f} {m['accuracy']:>6.3f}")
    print(f"\nwrote summary: {summary_path}")


if __name__ == "__main__":
    main()
