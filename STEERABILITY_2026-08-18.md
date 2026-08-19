# Steerability Evaluation — Inverted & Off-topic Policies (2026-08-18/19)

Do the models actually follow the policy, or do they classify on topic? Two independent probes: an **inverted** policy (flip allow/deny, measure how far each model bends) and an **off-topic** policy (swap in an unrelated harm, measure whether the model wrongly flags anyway). Both point the same way.

## Inverted policies

This run tests policy-following directly by pairing each normal policy with an **inverted mirror** — a policy that flips allow/deny — and measuring how far each model bends.

The inverted policies (`eval/policies/selfharm_inverted.md`, `eval/policies/sexual_content_inverted.md`) reframe the platform as one that *affirmatively protects* self-harm / sexual expression, and defines the violation as the **suppression** of that expression (anti-self-harm messaging, "get help" resources, purity/censorship talk, etc.). A model that understands policy should stop flagging the now-allowed content — and ideally start flagging the suppression content instead. A model that keyword-classifies should be unmoved (or flag *more*, since the inverted policy text is stuffed with sensitive vocabulary).

Models: `Shieldstral-1.0-3B` (local, single forward pass, yes/no logits), `cope-a-9b` (Gemma-2-9B LoRA, Modal), `cope-b-a4b` (standalone classifier, Modal).

## Summary

- **Shieldstral is ~0% steerable — it classifies on topic, not policy.** Of the sensitive content it flagged under the normal policy, it released **0%** when the policy said that content is now allowed (self-harm 0/19, sexual 0/48). Worse, flagging goes *up* under the inverted policy (self-harm gt+ 38% → 56%, sexual 92% → 98%; benign gt− flagging also rises) because the inverted policy is dense with self-harm/sexual keywords. It reacts to keyword presence, not the policy's stated allow/deny orientation.
- **Both cope variants genuinely steer.** cope-a released 100% (self-harm) / 92% (sexual) of previously-flagged content once it was declared allowed; cope-b released 100% / 96%.
- **cope-b steers most completely — it moves in both directions.** Under the inverted self-harm policy it not only stopped flagging self-harm content but *started* flagging the suppression content: benign (gt−) flag rate jumped **0% → 84%**, i.e. it flagged the anti-self-harm / recovery / crisis-line content that the inverted policy defines as the new violation. That is the cleanest possible steering signal.
- **The off-topic null test corroborates it.** Under a violent-extremism policy applied to self-harm / sexual content, both cope models flag **0%** (correctly ignoring content the policy doesn't target) while Shieldstral leaks **12%** on self-harm — reacting to the topic, not the policy. See the off-topic section below.
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

## Off-topic policies (null test)

A second, independent check on the same question. Pair each aligned baseline with a policy about an **unrelated** harm the test sets do not contain (`eval/policies/violent_extremism_offtopic.md` — orthogonal to both self-harm and sexual content, no shared vocabulary, but still a serious safety policy). Because no row is actually about that harm, the correct answer is *not-a-violation* on every row: any flag is topic leakage. Scored with `eval/steerability.py`'s companion `eval/offtopic.py` (false-flag rate); the sanity column confirms each model still flags normally under its aligned policy, so a clean off-topic result reflects policy-reading, not a dead classifier.

| Model | Domain | Aligned baseline flags | Off-topic false-flag (gt+ / all) |
|---|---|---|---|
| Shieldstral | self-harm | 19/50 (38%) | **6/50 (12%)** / 6/100 (6%) |
| Shieldstral | sexual | 48/52 (92%) | 0/52 (0%) / 2/129 (2%) |
| cope-a | self-harm | 23/50 (46%) | 0/50 (0%) / 0/100 (0%) |
| cope-a | sexual | 38/52 (73%) | 0/52 (0%) / 0/129 (0%) |
| cope-b | self-harm | 32/50 (64%) | 0/50 (0%) / 0/100 (0%) |
| cope-b | sexual | 50/52 (96%) | 0/52 (0%) / 0/129 (0%) |

Both cope models flag nothing under a policy that does not target their content, while flagging 46–96% under the aligned policy — clean policy-conditioning. Shieldstral leaks 12% on self-harm and, tellingly, only where the domains overlap semantically: self-harm content trips a *violence* policy (self-directed violence shares surface features), while sexual content does not look like extremism (0%). The leakage clusters exactly where a topic detector would confuse the two.

## Interpretation

Both probes agree. Shieldstral's architecture explains the result: a single forward pass emitting yes/no logits has no room to reason over a policy's stated orientation, so it keys on topical presence of the sensitive content. This makes it strong at *topic detection* under a normal, aligned policy (see [RESULTS.md](RESULTS.md) round 2) but unable to be steered by policy framing — it neither bends to an inverted policy nor stays quiet under an off-topic one. A real limitation for any deployment that wants to redefine what counts as a violation without retraining.

The cope models, which are policy-conditioned generative classifiers, follow the policy text on both probes: they release the old target under inversion (cope-b even adopts the new one) and produce zero false flags under an off-topic policy. That is the behavior you want from a steerable, policy-first moderation model.

Artefacts: `eval/results/predictions_{shieldstral,cope_a,cope_b}_{sh,sex}_{steer,offtopic}_*.csv`, scored via `eval/steerability.py` (inverted) and `eval/offtopic.py` (off-topic). The cope off-topic legs require `VLLM_API_KEY` set for the Modal endpoints.
