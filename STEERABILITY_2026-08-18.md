# Steerability Evaluation — Inverted Policies (2026-08-18)

Do the models actually follow the policy, or do they classify on topic? This run tests that directly by pairing each normal policy with an **inverted mirror** — a policy that flips allow/deny — and measuring how far each model bends.

The inverted policies (`eval/policies/selfharm_inverted.md`, `eval/policies/sexual_content_inverted.md`) reframe the platform as one that *affirmatively protects* self-harm / sexual expression, and defines the violation as the **suppression** of that expression (anti-self-harm messaging, "get help" resources, purity/censorship talk, etc.). A model that understands policy should stop flagging the now-allowed content — and ideally start flagging the suppression content instead. A model that keyword-classifies should be unmoved (or flag *more*, since the inverted policy text is stuffed with sensitive vocabulary).

Models: `Shieldstral-1.0-3B` (local, single forward pass, yes/no logits), `cope-a-9b` (Gemma-2-9B LoRA, Modal), `cope-b-a4b` (standalone classifier, Modal).

## Summary

- **Shieldstral is ~0% steerable — it classifies on topic, not policy.** Of the sensitive content it flagged under the normal policy, it released **0%** when the policy said that content is now allowed (self-harm 0/19, sexual 0/48). Worse, flagging goes *up* under the inverted policy (self-harm gt+ 38% → 56%, sexual 92% → 98%; benign gt− flagging also rises) because the inverted policy is dense with self-harm/sexual keywords. It reacts to keyword presence, not the policy's stated allow/deny orientation.
- **Both cope variants genuinely steer.** cope-a released 100% (self-harm) / 92% (sexual) of previously-flagged content once it was declared allowed; cope-b released 100% / 96%.
- **cope-b steers most completely — it moves in both directions.** Under the inverted self-harm policy it not only stopped flagging self-harm content but *started* flagging the suppression content: benign (gt−) flag rate jumped **0% → 84%**, i.e. it flagged the anti-self-harm / recovery / crisis-line content that the inverted policy defines as the new violation. That is the cleanest possible steering signal.
- **Watch the scoring trap.** The `summary_*_steer_*.csv` files score inverted runs against the *original* ground-truth labels, so a correctly-steered model shows LOW F1 and a keyword-reactor shows HIGH F1 — inverted from what matters. Read steerability from `eval/steerability.py` (flip rate), not from the summary F1/accuracy.

## Setup

| | |
|---|---|
| **Models** | `Shieldstral-1.0-3B` (local), `cope-a-9b` (`juliet--cope-a-9b`), `cope-b-a4b` (`juliet--cope-b-a4b`) |
| **Design** | Each model runs a normal policy and its inverted mirror in the same sweep; `ground_truth` stays the original labels (positive = real self-harm / sexual content) |
| **Core metric** | Flip rate 1→0 on ground-truth-positive rows: of the sensitive content flagged under the normal policy, how much is released under the inverted policy |
| **Readout** | `python eval/steerability.py 'results/predictions_<model>_<domain>_steer_*.csv' --baseline <stem> --inverted <stem>` |
| **Rows** | self-harm n=100 (50 gt+ / 50 gt−); sexual n=129 (52 gt+ / 77 gt−) |
| **Errors** | 0 malformed predictions across all six runs |
| **Date** | 2026-08-18 |

## How to read this

- **Baseline flag rate** — of the real sensitive content, how much the model flags under the *normal* policy. Sanity check that there's something to release.
- **Flip 1→0 (steerability)** — of that flagged content, how much the model *releases* under the inverted policy. **This is the headline number.** ~0% = held its safety prior (topic classifier); ~100% = followed the policy.
- **gt+ still flagged (inverted)** — residual flagging of now-allowed content. Should crater if steered.
- **gt− flag movement** — negatives (benign under the normal policy) that become the *new* violation under inversion. Rising means the model picked up the new target, not just dropped the old one.

## Results

### Steerability (flip 1→0 on real sensitive content — higher = more steerable)

| Model | Self-harm | Sexual |
|---|---|---|
| Shieldstral-1.0-3B | **0/19 (0%)** | **0/48 (0%)** |
| cope-a-9b | 23/23 (100%) | 35/38 (92%) |
| cope-b-a4b | 32/32 (100%) | 48/50 (96%) |

### Full readout per run

| Model | Domain | Baseline gt+ flagged | Flip 1→0 (steer) | Inverted gt+ still flagged | gt− flagged: base → inverted |
|---|---|---|---|---|---|
| Shieldstral | self-harm | 19/50 (38%) | **0/19 (0%)** | 28/50 (56%) | 4% → 8% |
| Shieldstral | sexual | 48/52 (92%) | **0/48 (0%)** | 51/52 (98%) | 26% → 36% |
| cope-a | self-harm | 23/50 (46%) | 23/23 (100%) | 0/50 (0%) | 4% → 8% |
| cope-a | sexual | 38/52 (73%) | 35/38 (92%) | 3/52 (6%) | 6% → 3% |
| cope-b | self-harm | 32/50 (64%) | 32/32 (100%) | 1/50 (2%) | **0% → 84%** |
| cope-b | sexual | 50/52 (96%) | 48/50 (96%) | 2/52 (4%) | 13% → 6% |

Rising gt+ flagging under inversion (Shieldstral) is the keyword-reaction signature: the inverted policy adds sensitive vocabulary and the model flags *more*. Rising gt− flagging (cope-b self-harm) is the opposite — the model correctly adopts the inverted policy's new target.

## Interpretation

Shieldstral's architecture explains the result: a single forward pass emitting yes/no logits has no room to reason over a policy's stated orientation, so it keys on topical presence of the sensitive content. This makes it strong at *topic detection* under a normal, aligned policy (see [RESULTS.md](RESULTS.md) round 2) but unable to be steered by policy framing — a real limitation for any deployment that wants to redefine what counts as a violation without retraining.

The cope models, which are policy-conditioned generative classifiers, follow the policy text. cope-b's ability to move in both directions — releasing the old target and adopting the new one — is the behavior you want from a steerable, policy-first moderation model.

Artefacts: `eval/results/predictions_{shieldstral,cope_a,cope_b}_{sh,sex}_steer_20260818_*.csv`, scored via `eval/steerability.py`.
