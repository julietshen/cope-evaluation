# Model Testing Guide

A walkthrough of how we serve and evaluate open-weight safety models, written for someone who doesn't write code day-to-day. No jargon without explanation.

The recipe here is general — it works for any open-weight LLM-based safety classifier you can download from Hugging Face. We use **`zentropi-ai/cope-b-a4b`** as the running example throughout because we're actively evaluating it for inclusion in the ROOST Model Community. Where something is cope-specific (the prompt template, the exact flags, the file names) it's called out so you can adapt cleanly to a different model.

This guide has two halves:

- **Part 1 — Serving**: getting an open-weight model deployed on a rented GPU and reachable from the laptop.
- **Part 2 — Evaluating**: feeding test content through the deployed model under different "policy" prompts and measuring how well it labels things.

---

## Part 1: Serving

### What we're trying to do

We want a deployed AI model — running on a remote GPU — that our laptop can send test inputs to. "Running" an AI model means loading its weights (a large file full of numbers that define what the model "knows") into the memory of a powerful computer with a graphics card, and then having a program take a user's input, do the math, and produce a response.

Our worked example: **cope-b-a4b** (made by a company called Zentropi AI). It's pre-release — not available through any hosted API like OpenAI or Anthropic — so we have to run it ourselves. Its weights are ~50 GB. Most useful open-weight safety models are in the 8–80 GB range; the steps below scale up or down accordingly.

### Why we can't just run it on the laptop

Three problems with running it on the MacBook:

1. **Memory.** A typical safety model needs 10–100 GB of memory just to load. The MacBook has 36 GB. Larger models physically can't fit.
2. **GPU.** AI models run thousands of times faster on NVIDIA graphics cards than on a regular CPU. The MacBook has Apple's own GPU, which most AI inference tools don't support yet.
3. **The serving software.** The standard tool for serving large AI models, called **vLLM**, doesn't run on Macs at all. It's built for Linux servers with NVIDIA GPUs.

So we need a remote computer with the right hardware.

### Why we picked Modal

We considered several options for renting GPU computers:

| Option | Best for | Why we didn't pick it |
|---|---|---|
| **RunPod** | Long testing sessions (3+ hrs at a stretch) | You pay even when the computer is idle. Easy to forget to turn off. |
| **Lambda Labs** | Reliable, well-documented usage | Slightly more expensive; no auto-scaling |
| **AWS / GCP / Azure** | Big enterprise deployments | Slow to set up, most expensive, paperwork |
| **Modal** ✓ | **Intermittent testing** | Pay only when you're actively using it. Scales to zero. |

Modal is what's called a **serverless GPU platform**. The idea: you write a Python script describing what you want to run. Modal handles the rest — provisioning a computer with a GPU when a request comes in, charging you per second of actual use, and turning everything off when you're done. You never SSH into anything or babysit a running server.

**Cost expectation:** an H100 GPU on Modal is ~$4/hour while actively serving a request. For light testing (a few requests, then walking away for hours), you'd spend maybe $1–5/day. For a heavy day of nonstop testing, more like $20–40.

### The pieces involved

A few proper nouns it helps to know:

- **Hugging Face (HF).** A website/registry where AI researchers publish their models. Like GitHub but for AI weights. Cope lives at `huggingface.co/zentropi-ai/cope-b-a4b`; other safety models you might evaluate include `openai/gpt-oss-safeguard-20b`, `meta-llama/Llama-Guard-3-8B`, `google/shieldgemma-9b`.
- **HF token.** A password-like string that proves you have permission to download a specific model. Some models (including cope) are "gated" — you have to click "I agree to the terms" on the model's page first, then your token unlocks downloads.
- **vLLM.** The software that loads the model into GPU memory and exposes a familiar OpenAI-style API (`POST /v1/chat/completions` and `POST /v1/completions`) so we can talk to it. It does a lot of optimization tricks under the hood to make inference fast. Works for most open-weight transformer models.
- **Modal Secret.** A safe place to store sensitive strings (like the HF token). The code references the secret by name; the actual values never appear in files we commit.
- **Modal Volume.** A persistent disk that survives between runs. We use it to cache model weights so we don't have to re-download them every time we start the server.

### The workflow, step by step

The steps below use cope as the example. For a different model, change three things: the Hugging Face model name, the Modal app name, and the secret name. Everything else stays the same.

#### Step 1: Accept the model's terms (if gated)

On Hugging Face, visit the model page while logged in and click "Agree and access repository." This is a one-time thing per model. After this, your account is on the allowlist. Non-gated models skip this step.

#### Step 2: Create an HF token

Go to Hugging Face settings → tokens, and make a new "Fine-grained" token scoped to just this one model with read access. Copy the long `hf_...` string. Treat it like a password — anyone with it can download things as you.

#### Step 3: Install Modal locally

We need a small tool on the laptop that talks to Modal's servers:

```bash
uv tool install modal     # one-time install
modal token new           # one-time login, opens browser
```

#### Step 4: Tell Modal our secrets

We store the HF token (so Modal's computer can download the model) and a custom API key (so random people on the internet can't use our endpoint):

```bash
modal secret create cope-secrets HF_TOKEN=hf_yourtoken VLLM_API_KEY=sk-pick-something
```

🔧 **Adapt for a different model**: change `cope-secrets` to `<your-model>-secrets`, and reference that name in the Python file (Step 5).

Modal saves these on their servers. The Python script reads them at runtime through environment variables (`os.environ["HF_TOKEN"]`, etc.).

#### Step 5: The Python serving recipe (`serve_cope.py`)

This file is the recipe Modal follows. The important parts in plain English:

- "Build a Linux machine with NVIDIA's CUDA toolkit and Python 3.12 installed, then install vLLM on it."
- "Mount a persistent disk at a specific path so cached files survive between runs."
- "When a web request comes in, give the machine an H100 GPU and run `vllm serve <MODEL_NAME>` to start the model."
- "After 10 minutes of no requests, shut the GPU machine down to save money."

The cope-specific bits are at the top of the file:

```python
MODEL_NAME = "zentropi-ai/cope-b-a4b"
GPU_CONFIG = "H100:1"        # 80GB GPU; smaller models fit on A100:1 or L40S:1
MAX_MODEL_LEN = 8192          # max prompt+response length, in tokens
```

We pass a few flags to vLLM:

- `--trust-remote-code`: many open-weight models ship their own custom Python loader. We're saying "yes, run it." Required for cope.
- `--enforce-eager`: skip a slow but optional GPU-kernel compilation step. Cope's pre-release weights had a bug in that compilation step; safe to drop this flag once a model is well-shaken.
- `--max-model-len 8192`: how long an input the model accepts. 8192 "tokens" is roughly 6,000 words. Set this to the largest prompt you actually need — higher values use more GPU memory.
- `--api-key`: require requests to include our custom API key.

🔧 **Adapt for a different model**: change `MODEL_NAME`, possibly `GPU_CONFIG` (rule of thumb: model weights in GB × 1.3 ≤ GPU memory), and check the model card for any required vLLM flags.

#### Step 6: Pre-download the weights (one-time, optional)

```bash
modal run serve_cope.py::download_model
```

This runs a tiny cheap machine (no GPU, ~$0.01) that downloads the model weights and stores them on the persistent volume. For cope's 50 GB it takes 10–20 minutes. After this, every future startup is fast because the weights are already there. **Optional**, but saves ~$1.50 per future cold start.

#### Step 7: Deploy the server

```bash
modal deploy serve_cope.py
```

This publishes the recipe to Modal. Modal prints a public URL like `https://juliet--cope-b-a4b-serve.modal.run`. The server isn't *running* yet — Modal will spin up a machine only when a request arrives. From this point on the laptop can be closed; the endpoint stays available.

(`modal deploy` is the production-style command. There's also `modal serve` which is for interactive development — that one tears down when you close the terminal.)

#### Step 8: Send a test request

The cope prompt format is unusual — see Part 2 for the full template. For a quick smoke test against most vLLM-served models, the chat-completions endpoint works:

```bash
curl https://juliet--cope-b-a4b-serve.modal.run/v1/chat/completions \
  -H "Authorization: Bearer sk-your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zentropi-ai/cope-b-a4b",
    "messages": [{"role":"user","content":"say hi"}],
    "max_tokens": 64
  }'
```

The very first request after a long idle period is a **cold start** — Modal provisions a GPU machine, loads the weights into GPU memory, and starts serving. Takes ~2–3 minutes for a 50 GB model. Subsequent requests are fast (a few seconds). If 10 minutes go by with no requests, the machine spins down, and the next request pays the cold-start cost again.

#### Step 9: Shut it down when fully done

```bash
modal app stop cope-b-a4b
```

This guarantees the GPU machine can't spin up. The volume (with cached weights) sticks around for a few cents per month. To delete the volume too, use `modal volume delete hf-cache`.

### Errors we hit along the way, decoded

These are common when standing up a new model. Some background on each:

1. **"Deprecated on 2025-02-24: 'container_idle_timeout' renamed to 'scaledown_window'."**
   Modal renamed a parameter in their library. Our Python file used the old name. One-line fix.

2. **"Each item should be of the form `<KEY>=VALUE`."**
   When we tried to type the secret-creation command across multiple lines using backslash-continuation, blank lines between the lines broke the continuation. The shell ran a command without any actual values. Fix: write it on one line.

3. **"modal-http: invalid function call."**
   We tried the wrong URL — guessed at it instead of looking it up on the dashboard.

4. **"modal-http: app for invoked web endpoint is stopped."**
   We had stopped the app earlier and never re-deployed it.

5. **"RuntimeError: Engine core initialization failed."**
   The big one. vLLM was trying to JIT-compile a small piece of GPU code on-the-fly the first time someone asked for a response. The Linux machine we'd configured had CUDA's runtime libraries but not the compiler (`nvcc`). Fix: switch to a beefier base image that includes the compiler. Costs about 5 minutes of one-time image-rebuild.

### What "good" looks like

When everything is working, the test curl above returns JSON in the shape:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "choices": [{
    "message": {"role": "assistant", "content": "Hi there!"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
}
```

That's the OpenAI Chat Completions API shape — meaning anything that already speaks "OpenAI" (the OpenAI SDK, LangChain, evaluation harnesses, etc.) can be pointed at our endpoint by changing the `base_url` and `api_key` and will work without other code changes.

### Quick reference

```bash
# Start of session
modal deploy serve_cope.py
# … test things, send requests …
# End of session (optional, saves the cents)
modal app stop cope-b-a4b

# Check what's happening on the GPU machine
modal app logs cope-b-a4b

# Update the code, redeploy
modal deploy serve_cope.py     # same command, picks up changes

# Replace a secret value
modal secret create --force cope-secrets HF_TOKEN=hf_new VLLM_API_KEY=sk-new
```

### Costs in one place

| What | When billed | Rough cost |
|---|---|---|
| Modal GPU (H100, 1×) | Per second, only while serving a request | ~$4/hour active |
| Modal CPU container (`download_model`) | Per second | ~$0.01 for the 15-min download |
| Persistent volume (50 GB) | Continuously | ~$1/month |
| Hugging Face | Free | $0 |
| Our laptop | N/A | $0 — laptop just sends curl requests |

For a typical week of intermittent testing: expect **$5–15/week**, mostly GPU time. Smaller models on smaller GPUs (A100, L40S) cost roughly half.

---

## Part 2: Evaluating

This is where Part 1 starts paying off. With a working endpoint, we can run the same model against a labelled test set under different "policy" prompts and see how its decisions change.

### What kind of models this section is about

We're focused on **policy-conditioned safety classifiers**: open-weight models that take both a *policy* (a written description of what counts as a violation) and *content* (a post, message, transcript) and output a label, typically binary. Cope, gpt-oss-safeguard, Llama-Guard, and ShieldGemma all fit this pattern, though they differ in their exact prompt formats and output shapes.

The appeal of these models compared to a traditional classifier is that **the same weights can do many different jobs** — you swap the policy text, and now you have a different classifier. That's the whole point. It also raises the question this section is built around: **how does performance change as the policy text gets more or less detailed?**

### Why varying-detail policies?

A policy can be a one-liner ("flag self-harm content") or a multi-page Includes/Excludes document. Different teams will have different appetites for writing detail — and different operational tolerances for false positives vs false negatives. By writing the same policy at multiple levels of detail and running the same test set against each, we can see:

- How much recall (catching the bad stuff) you gain from added detail.
- Whether precision (not flagging the good stuff) holds up as detail grows.
- Whether one well-written paragraph is enough, or whether you really do need the full structure.
- Whether the *official* policy from a model creator agrees with how a downstream platform would label content.

The last point turns out to matter a lot — see the self-harm case study below.

### How models receive a prompt — and why this varies

Different policy-conditioned classifiers expect different prompt shapes. Some examples:

- **Cope** uses a raw text completion format (`INSTRUCTIONS / POLICY / CONTENT / ANSWER` blocks, with the model emitting a single `0` or `1` after the ANSWER header). Hit `/v1/completions`, not `/v1/chat/completions`. `max_tokens=1`, `temperature=0`.
- **gpt-oss-safeguard** uses chat completions with the policy as the system message and the content as the user message; the model is asked to emit `0` or `1`.
- **Llama-Guard** uses a fixed taxonomy embedded in the prompt and emits `safe` / `unsafe` plus a category code.

🔧 **Cope-specific**: cope's prompt template is:

```
INSTRUCTIONS
============

Examine the given POLICY and determine if the given CONTENT meets the criteria for ANY of the LABELS. Answer "1" if yes, and "0" if no.


POLICY
======

{your policy text here}


CONTENT
=======

{the post or message to classify}


ANSWER
======

```

For a different model, replace this template in `eval_cope.py` with the model's expected prompt format. The rest of the harness (loading policies, parallelizing requests, computing metrics) is model-agnostic.

### Directory layout (the `eval/` folder)

```
eval/
├── eval_cope.py                  # the harness — sends prompts to Modal, writes results
├── test_set.csv                  # default: 100-row self-harm test set
├── policies/                     # one .md file per policy variant
│   ├── minimal.md                # one-sentence (self-harm)
│   ├── simple.md                 # short paragraph
│   ├── medium.md                 # structured (terms + includes/excludes)
│   ├── full.md                   # detailed (cope-style hand-written version)
│   ├── zentropi_official.md      # official Zentropi self-harm policy (from their API)
│   ├── sex_minimal.md            # same spread, sexual content
│   ├── sex_simple.md
│   ├── sex_medium.md
│   ├── sex_zentropi_long.md      # Zentropi's published "Long Policy" for sexually explicit content
│   ├── sex_zentropi_simple.md    # Zentropi's "Simple policy" (incomplete in source — kept for completeness)
│   └── sex_oai.md                # OpenAI's GS0/GS1/GS2 sexual-content policy
├── sex_eval/                     # data prep workspace for the sexual-content eval
│   ├── sample_bsky_for_sex_eval.py   # stratified sampler over Bluesky 8M post dataset
│   ├── candidates_to_label.csv       # 80 candidates for manual labelling
│   ├── build_redteam.py              # generates synthetic red-team set
│   ├── redteam_set.csv               # 50 hand-crafted edge cases
│   ├── merge_test_set.py             # combines the two into the final test set
│   └── test_set.csv                  # produced by merge_test_set.py
└── results/                          # CSVs of predictions and per-policy metrics
```

The same layout works for any model — just rename `eval_cope.py` and swap the prompt template inside it.

### Running an evaluation, step by step

#### Step 1: Set the API key in the shell

```bash
export VLLM_API_KEY="<the key you put in the Modal secret>"
```

(It's the same value you used in Part 1 Step 4. If you've already set it as an environment variable in your shell profile, you're fine.)

#### Step 2: Make sure the Modal endpoint is alive

```bash
modal app list
# Look for your app name. If it's stopped, redeploy:
modal deploy serve_cope.py
```

The first request after a deploy will pay the cold-start cost (~2–3 minutes). The eval harness includes a warm-up call by default.

#### Step 3: Run the harness

For the self-harm eval against the held-out test set:

```bash
cd eval
python eval_cope.py \
  --policies minimal simple medium full zentropi_official \
  --concurrency 16
```

For the sexual-content eval:

```bash
python eval_cope.py \
  --test-set sex_eval/test_set.csv \
  --label sex \
  --policies sex_minimal sex_simple sex_medium sex_zentropi_long sex_oai \
  --concurrency 16
```

Useful flags:

- `--test-set PATH` — point at any CSV with columns `id, content, ground_truth`
- `--label TAG` — prefix output files with a tag so different runs don't get mixed up
- `--limit N` — only run on the first N rows (good for smoke tests)
- `--skip-warmup` — skip the warm-up call if you know the endpoint is hot

#### Step 4: Read the outputs

Each run writes two files under `eval/results/`:

- **`predictions_<tag>_<timestamp>.csv`** — one row per test sample, with a `<policy>_pred` column for each policy variant. Open this to see which samples each policy got wrong.
- **`summary_<tag>_<timestamp>.csv`** — one row per policy variant with TP/FP/FN/TN counts, plus precision, recall, F1, and accuracy.

The harness also prints a tidy summary table to the terminal at the end of each run.

### Building a test set when one doesn't exist

The self-harm eval used a pre-labelled CSV from a partner. For sexual content (and most new domains you'll evaluate), you'll need to build a labelled test set from scratch. We use a two-part approach:

#### Part A — Stratified sample from a public dataset for manual labelling

`sex_eval/sample_bsky_for_sex_eval.py` is a worked example of this pattern. It downloads one shard of a public Bluesky post dataset (~130 MB, ~390k posts) and filters it into three "tiers":

- **Tier A** — strong-signal keywords (likely-violating) — drives recall measurement
- **Tier B** — borderline / suggestive keywords — judgment-call zone
- **Tier C** — no signal keywords — drives precision measurement

The sampler hard-drops:

- posts with images (we only label text)
- posts shorter than 12 chars or longer than 400 chars
- mostly-URL or mostly-non-Latin posts
- **anything where a sexual term co-occurs with a minor-related term** (CSAM-adjacent; dropped entirely rather than risked)

By default it pulls 35 / 30 / 15 = **80 posts** to `candidates_to_label.csv` with an empty `ground_truth` column. Open the CSV in Numbers/Excel and fill it in: `1` = violating, `0` = not, blank to skip.

```bash
cd eval/sex_eval
python sample_bsky_for_sex_eval.py
open candidates_to_label.csv     # macOS: opens in Numbers
```

🔧 **Adapt for a different domain**: swap the keyword regex lists in the sampler. The hard-drop rules around image posts, length, and minor-related terms should stay.

#### Part B — Synthetic red-team set

A real dataset alone leaves blind spots: the exclusions and edge cases that policy writers worry about (recovery narratives, educational content, fictional framings) appear too rarely in random samples to drive metrics. So we also write a 50-row hand-crafted set covering categories chosen to stress-test specific policy clauses.

`sex_eval/build_redteam.py` is a worked example. Each of its 10 categories targets a clause in the cope sexual-content policy:

- clear explicit sex acts (5, expected 1)
- explicit invitations / participation offers (5, expected 1)
- body / anatomy with erotic framing (5, expected 1)
- coded language / euphemisms (5, expected 1)
- sexual humor / hyperbole — judgment calls (3 expected 1, 2 expected 0)
- educational / clinical content (5, expected 0)
- recovery / addiction narratives (5, expected 0)
- fictional creative writing — graphic vs non-graphic (3 expected 1, 2 expected 0)
- sexually degrading speech vs critique of such speech (3 expected 1, 2 expected 0)
- factual body-part mentions without sexual framing (5, expected 0)

🔧 **Adapt for a different domain**: write your own category list from the policy's Includes/Excludes, then ~5 examples per category. The goal is one example per policy clause, plus one or two "almost but not quite" examples per exclusion clause.

#### Part C — Merge and run

Once labels are filled in:

```bash
python merge_test_set.py
# combines labelled + redteam into sex_eval/test_set.csv
cd ..
python eval_cope.py \
  --test-set sex_eval/test_set.csv \
  --label sex \
  --policies sex_minimal sex_simple sex_medium sex_zentropi_long sex_oai
```

### Worked examples

We've run this end-to-end on two harm domains using cope-b-a4b: **self-harm** (pre-labelled 100-row test set) and **sexually explicit content** (test set built from a stratified Bluesky sample + a synthetic red-team set). See [RESULTS.md](RESULTS.md) for the test-set details, the per-policy precision/recall/F1 numbers, the head-to-head against gpt-oss-safeguard, and the findings around Zentropi's published policies and policy-format alignment.

The general-purpose takeaway from those runs, that future evals should bake in:

**Measuring "accuracy" only makes sense after you've checked that the test set's labelling framework matches the policy's framework.** Both cope evaluations turned up substantial disagreements between Zentropi's published policies and the labelled test sets — not because the model was wrong, but because the policies and the labels were optimising for different things. When the two disagree, the numbers report the *disagreement*, not the model. Always look at the false positives and false negatives by hand before drawing model-quality conclusions from F1.

### Reproducing or extending

To add a new policy variant: drop a markdown file into `eval/policies/` (any structure — the harness reads it as a single text blob), then pass its name (without `.md`) to `--policies`.

To add a new harm domain: build a test CSV with columns `id, content, ground_truth`, write 3–5 policy variants under `eval/policies/`, and run:

```bash
python eval_cope.py --test-set path/to/your_test_set.csv --label your_domain --policies policy_a policy_b ...
```

The harness is policy-agnostic — it just substitutes whatever you put in `policies/<name>.md` into the cope prompt template.

### Adapting this whole setup for a different model

The general recipe — Modal + vLLM + per-policy harness + stratified test set + synthetic red-team — works for any policy-conditioned classifier. To swap cope for, e.g., gpt-oss-safeguard or Llama-Guard, change three things:

1. **`serve_cope.py`** — rename to `serve_<model>.py`, update `MODEL_NAME`, `GPU_CONFIG`, `MAX_MODEL_LEN`, app name (`modal.App("...")`), and Modal secret name. Check the model card for any required vLLM flags.

2. **`eval_cope.py`'s prompt template** — replace the `INSTRUCTIONS / POLICY / CONTENT / ANSWER` block with whatever the model expects. For chat-based models, also swap `/v1/completions` for `/v1/chat/completions` and reshape the request body to `messages=[{"role": "system", ...}, {"role": "user", ...}]`. Update the output parsing to match the model's expected response shape (single token vs structured JSON vs label string).

3. **`policies/*.md`** — most policy text is portable across models, but the *format* the model was trained on may matter. Cope was fine-tuned on the zentropi-style Includes/Excludes structure; Llama-Guard expects a numbered taxonomy. Look at each model's example prompts.

Everything else — the stratified sampler, the red-team set construction approach, the merge script, the metrics computation — generalises without changes.

### Files cheat sheet

| File | Purpose |
|---|---|
| `serve_cope.py` | Modal deployment recipe — Part 1 |
| `eval/eval_cope.py` | Eval harness — sends prompts to Modal, writes results |
| `eval/test_set.csv` | Default test set (self-harm, 100 rows; column identifiers sanitized) |
| `eval/policies/*.md` | Policy variants — one per file, name passed to `--policies` |
| `eval/results/` | Output CSVs (predictions + summary) timestamped per run |
| `eval/sex_eval/sample_bsky_for_sex_eval.py` | Stratified Bluesky sampler |
| `eval/sex_eval/candidates_to_label.csv` | 80 posts to be labelled by hand |
| `eval/sex_eval/build_redteam.py` | Generates 50 synthetic red-team cases |
| `eval/sex_eval/redteam_set.csv` | The generated red-team set |
| `eval/sex_eval/merge_test_set.py` | Combines labelled + red-team into final test set |
