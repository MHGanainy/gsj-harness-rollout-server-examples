# ADR-0003 — the verl route: direct, and what settles the uni-agent question

Date: 2026-08-12 (library CP-20). Status: accepted.
Counterparts: ADR-0001 (the slime bridge — the template), library CP-20
report (the milestone: two bridges prove the boundary is real).

## Context

M6a asks for a second bridge: callback-shaped `SessionResult` → verl's
batch type. Two routes were on the table, and a question has been open
since the predecessor (`gsj-envloader`, whose collector pins uni-agent @
`73b0f41`; library `docs/VERDICT.md`: "the predecessor's uni-agent path
already speaks verl, so an adapter consuming our validated callback
bodies … costs less than a second Polar-side bridge — decide after M4"):

- **Direct** — build the `DataProto` ourselves, CP-16's bridge as the
  template for everything except the target class.
- **Via uni-agent** — reuse its trainer-side plumbing into verl and feed
  it our traces, with the hard discipline that uni-agent's capture layer
  (codec, gateway, Framework, sandbox providers) stays out.

Both source trees were read at their pins before any code was written:
uni-agent @ `73b0f41` (the predecessor's pin), verl @
`1ae945592754cbeb1350cbe092fe6117070fd4c7` — **uni-agent's own submodule
pin** (verl 0.9.0.dev), so the verl surface examined is the one the
uni-agent path actually speaks.

## What the investigation found in uni-agent

Its trainer-side entry point is `OpenAICompatibleAgentFramework
.generate_sequences(prompts: TensorDict) -> None` (`uni_agent/framework/
framework.py:384`), reached from verl via the agent-loop extension point
(`rollout.agent.agent_loop_manager_class = uni_agent.framework.entry
.AgentFrameworkRolloutAdapter`). Three facts settle the route:

1. **It never hands verl a batch.** `generate_sequences` returns `None`:
   it drives its own gateway sessions and writes per-trajectory fields —
   UNPADDED 1-D tensors (`prompts`, `responses`, `input_ids`,
   `attention_mask=all-ones`, `position_ids`, `response_mask`,
   `loss_mask=response_mask`, optional `rollout_log_probs`, sparse
   `rm_scores` with the reward on the last response token) — into
   **TransferQueue** (`_trajectory_to_tq_field_and_tag`,
   `framework.py:910–1001`). The padded `[B, L]` batch the CP asks for is
   assembled later, inside verl's TQ-mode trainer
   (`transfer_queue.enable=True`), not anywhere in uni-agent. The
   conversion segment we were invited to reuse does not contain the
   batching.
2. **Its input type lives in the capture layer.** The segment's sole
   input is `Trajectory` — imported from `uni_agent.gateway.session`
   (`framework.py:23`), i.e. constructing one imports the gateway
   package; the class itself is instantiated with a `gateway_manager` as
   its first constructor argument. Mechanically a two-attribute stub plus
   a monkeypatched `tq` handle can drive the private method (uni-agent's
   own CPU test does exactly that with a fake gateway manager) — but
   that is standing up the Framework with its capture-layer imports and
   faking its delivery infrastructure, which is the route's own
   discipline violated twice.
3. **What it would buy is ~90 lines of key-naming** whose target
   conventions verl's own agent-loop worker states independently
   (`verl/experimental/agent_loop/agent_loop.py::_agent_loop_postprocess`
   / `_postprocess`). uni-agent and verl agree on the contract because
   uni-agent implements verl's; reading both gives the direct route two
   in-tree confirmations of every key it writes.

**The standing question, answered plainly**: uni-agent's trainer-side
path cannot be fed externally-produced trajectories as-is. There is no
offline/replay seam; the entry point generates, it does not ingest; the
trajectory type is the gateway's; and the output side presumes a live
TransferQueue plus a verl trainer configured for it. What IS true — and
what the predecessor's intuition was pointing at — is that uni-agent's
per-trajectory field schema is essentially our `Trace` (flat
`prompt_ids` / `response_ids` / `response_mask` / `response_logprobs`
lists): the *contract* transfers; the *plumbing* does not.

## The two trainer generations at this pin (F-11)

Read after the route was chosen, and it sharpens rather than reverses
it: verl @ `1ae9455` is mid-migration. `python -m verl.trainer.main_ppo`
defaults to the **v1 TransferQueue trainer** (`trainer.use_v1: true`,
`ppo_trainer.yaml:222`); the padded `[B, L]` `DataProto` contract this
bridge targets is the **classic** (`main_ppo_v0`) path's — deprecated
upstream, still documented, and the only contract an EXTERNAL process
can hand a batch to. The v1 path cannot ingest externally-built batches
at all: its records are unpadded nested tensors written into
TransferQueue by clients that must run inside the trainer's Ray cluster
after `tq.init()` (`v1/agent_loop_tq.py:52-57`), its uids are minted by
the trainer per prompt row (`v1/trainer_base.py:1315-1372` — a feeder
must echo them, never invent them), and its tag schema is read
unconditionally. So the boundary finding generalizes: **neither
uni-agent nor verl's own default pipeline accepts trajectories from
outside its process boundary** — the consumable seam for an external
rollout server is the classic batch (this bridge) or a custom
`AgentLoop` returning `AgentLoopOutput`, in which case verl drives
generation and does this bridge's padding itself. One concrete
consequence landed in the bridge: the batch carries `loss_mask` equal to
`response_mask`, because the v1 engine's loss normalizer sums
`loss_mask` (`workers/engine/fsdp/transformer_impl.py:676-681`) while
the classic losses read `response_mask` — the same both-keys posture as
verl's own TQ writer and uni-agent.

## Decision

**Direct.** `bridge.py` converts callback bodies to a real
`verl.protocol.DataProto` shaped exactly like the agent-loop worker's
output batch (the conversion table lives in the module docstring and the
CP-20 report):

- fixed-width layout — prompt LEFT-padded, response RIGHT-padded,
  `input_ids = cat`, `attention_mask` from real-token extents,
  `position_ids` via verl's own `compute_position_id_with_mask`;
- the mask key is `response_mask` (verl's RL path has no separate
  `loss_mask`; `ray_trainer.py` only derives the key when absent, so the
  bridge's mask — the builder's `loss_mask`, verbatim — is consumed by
  every loss term);
- **rewards, not advantages** (`rm_scores`, outcome reward at the last
  real response token): advantage estimation is config-owned inside
  verl's fit loop and grouped by the `uid` non-tensor key — producing
  advantages bridge-side would bypass `algorithm.adv_estimator` and
  re-own the numerics class F-08 caught a vendored post-processor
  getting wrong;
- ERROR/TIMEOUT/unknown statuses and gate-failing sessions become
  fully-masked rows — **verl's own zero-gradient mechanism** (all-zero
  `response_mask` contributes exactly 0 in every `agg_loss` mode; verl
  itself neutralizes rejection-sampled and synthetic-padding samples the
  same way), fail-closed where the vendored slime adapter fell through
  to COMPLETED;
- the three CP-16 assertions unchanged (mask-before-ratio, sentinel
  rejection at the bridge's own floor, `checks` trainer-side), each with
  a test that fails when removed, plus a batching-stage invariant: the
  stacking may not manufacture trainable positions or non-zero logprobs
  under padding.

verl is pinned at uni-agent's submodule SHA and is REAL in the tests —
`DataProto`, `compute_grpo_outcome_advantage`,
`compute_rloo_outcome_advantage`, `agg_loss`,
`compute_position_id_with_mask` all execute (contrast F-03: slime needed
a constructor-surface double; verl needs none — its import closure on a
CPU-only box is torch/tensordict/numpy/ray/packaging/omegaconf/
transformers/codetiming/pydantic, requirements.txt).

## Consequence

The bridge is one file with zero server imports beyond
`gsj_rollout.checks`, and the boundary claim survives its second
trainer: nothing in the library repo changed. What the route costs: the
batching stage is ours to get right (the padding tests exist because of
it), and uid-group semantics are ours to surface (F-10: a singleton uid
group gets its raw reward as its advantage — grouping is a named CP-21
input, not a bridge default that can be guessed). What it settles: the
uni-agent shortcut is closed by measurement, not by taste — the segment
it offered does not produce the artifact this bridge exists to produce.
