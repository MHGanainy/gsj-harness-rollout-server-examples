# ADR-0001 — what the bridge is and isn't

Date: 2026-08-11 (library CP-16). Status: accepted.
Counterpart: library ADR-0018 (why it lives in this repo — the scope law).

## Context

M4's converting condition: the slime bridge runs one OPD loop on
collected traces, consuming masks and logprobs, resolving the library's
A-6. The target shape comes from the vendored Polar
`slime_bridge/adapter.py` (slime v0.3.0 + the router-tokens patch) —
read, not invented. That adapter is demo-purpose in specific, named ways:
it has **no value-level logprob validation** (presence and length only —
a `-9999.0` sentinel or a positive logprob passes `[float(v) for v in
logprobs]` untouched), it trusts the interstitial-0.0 discipline on
comment alone, it consumes Polar's pydantic models (the trainer never has
them), the placeholder hardcodes `tokens=[0, 0]`, missing rewards become
`0.0` in four places, and session status is matched by string literal.
This bridge keeps its good bones (per-trace Samples sharing `group_id`,
the zeroed-mask zero-gradient mechanism, the removable placeholder) and
replaces trust with assertions where our own checkpoints proved trust
fails (CP-02: upstream never value-checks; CP-09′: the measured floor).

## Decision

**What it converts, field by field** (callback-shaped mapping → slime
`Sample`, v0.3.0 surface):

| Trace field | Sample field | Rule |
| --- | --- | --- |
| `prompt_ids + response_ids` | `tokens` | plain concatenation, verbatim ids, never retokenized |
| `len(response_ids)` | `response_length` | — |
| `loss_mask` | `loss_mask` | builder's mask **verbatim** — the bridge never infers trainable positions; absent → all-zero; zeroed wholesale for ABORTED/FAILED/findings |
| `response_logprobs` | `rollout_log_probs` | the behaviour-policy values, asserted (below); `[0.0]*n` only on inert (zero-mask) paths |
| `reward` | `reward = {key: value}` | what the evaluator placed; `null` → `0.0` (FINDINGS F-02) |
| `prompt_messages` | `prompt` | deep copy; `""` when empty |
| `response_messages` | `response` | `[role] content` text — lossy, human-readable only |
| statuses | `status` | `TIMEOUT` → ABORTED; `ERROR`/errors → FAILED; findings → FAILED; `finish_reason == "length"` → TRUNCATED; else COMPLETED |
| session/trajectory/trace metadata | `metadata["polar"]` | the vendored block, plus scheduler keys (`group_id`, `policy_version`, `rollout_step`) when stamped |
| — | `metadata["gsj"]` | trainer-side findings, pins provenance, masked behaviour aggregates |

**What it asserts, and why each is non-negotiable:**

1. *Mask before ratio.* Interstitials carry `0.0` placeholders that read
   as probability 1.0; upstream states the mask is what makes the trainer
   ignore them (`prefix_merging.py`) — so applying it is ours. Enforced,
   not commented: a non-zero logprob at a masked position raises
   `MaskDisciplineError`, and the only behaviour aggregate the bridge
   emits is computed under the mask.
2. *Sentinel rejection.* `-9999.0` is finite and ≤ 0 — the naive
   discipline admits it, and CP-02 established upstream has no
   value-level validation anywhere. `SentinelLogprobError` at ingest,
   against the bridge's **own** `-9000.0` floor, deliberately independent
   of the server's `CheckPolicy` (a loosened policy must not loosen the
   trainer — test-proven).
3. *`checks` runs trainer-side.* The same `validate_session_result` the
   receiver ran, on what actually arrived — law 6 has two legs or it has
   none. Asserted called (recording-wrapper test), not just available.

**Rejection is two-tier:** pipeline poison (sentinel, broken placeholder
discipline, absent/misaligned arrays on a trainable session) **raises** —
the vendored `RolloutLogprobError` precedent, the group drops; episode
badness (gate findings, non-COMPLETED status) **masks** — FAILED, zeroed
mask, findings carried in metadata, group shape preserved. Aborted/failed
episodes contribute zero gradient by the zeroed mask; a session with no
usable trace at all becomes one `remove_sample=True` placeholder.

**No replay.** Replay-style validation needs the serving engine; CP-16 is
fixture-driven by decree. The bridge exports the CP-09′-measured floor
(`H200_REPLAY_FLOOR_MEAN = 0.008`, `H200_REPLAY_FLOOR_PER_POSITION =
0.21`) so CP-17 inherits the measured numbers, never the CP-18 anchor
(mean 0.005 / per-position 0.05), which fails as written on CUDA
continuous-batching capture.

**What it deliberately omits** — no store, no staleness tracking, no
ready grammar, no scheduler, no weight sync, no retries, no reward
computation. If the CP-17 loop needs any of those, that is a finding
about the architecture, not a licence to rebuild the predecessor.

## Consequence

The bridge is ~450 lines of trainer-owned code with zero server imports
beyond `gsj_rollout.checks`. Its tests run anywhere the wheel installs
(FallbackSample stands in for slime off-estate — F-03/F-04 bound what
that proves). CP-17 inputs it does **not** solve, named in the run book:
weight-sync mechanism, policy-version declaration (P3 inert), reward
attach (F-02), collection cadence against the 19-attempt qualification
rate, and on-estate verification of the real Sample surface (A-26).
