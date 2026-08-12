# ADR-0004 — the CP-21 loop's shape (one verl step, and everything around it)

## Context

Library CP-21 (M6b) runs the collect → convert → train → sync → collect
loop with verl — the second trainer, on the same unchanged server. CP-20's
run book named seven inputs; ADR-0003 fixed the route (direct bridge,
classic batch). What remained was the loop's own shape: who drives
generation, what process verl runs in, how the batch reaches an optimizer
step, how weights return to the estate's engine, and how the F-10 grouping
trap is closed. One desk finding constrains everything: at the pin the
classic v0 trainer itself is built on the unified model-engine workers
(`workers/engine_workers.py` — `DataParallelPPOActor` no longer exists),
so "feed the classic batch" cannot mean "call the old actor class".

## Decision

1. **Classic batch, generation stays with the estate.** The CP-20
   recommendation accepted: the bridge's padded `DataProto` feeds a
   one-step harness (`cp21_loop/train_one_step.py`); the estate's
   qualified vLLM generates, reached only through `gsj-rollout submit`.
   The v1 TQ residency and AgentLoop options change who drives
   generation — a different experiment, not run.
2. **verl runs standalone: `TrainingWorker` over the FSDP engine,
   WORLD_SIZE=1, no Ray runtime, no rollout worker, no critic, no ref.**
   The harness mirrors the v0 fit loop's own plumbing step for step
   (`ray_trainer.py`: `_compute_old_log_prob` → debug metrics →
   `extract_reward` → `token_level_rewards` →
   `compute_rollout_correction_and_add_to_batch` → `compute_advantage` →
   `_update_actor`), calling verl's functions directly; every numeric op
   is verl's. One deliberate substitution (F-12): verl's
   `left_right_2_no_padding` hard-requires flash-attn, absent on a host
   with no nvcc — reproduced in pure torch with the output contract
   asserted row by row. The engine runs its own flash-free branch
   (`use_remove_padding=False`, `attn_implementation: "sdpa"`) — chosen
   also because the desk-read showed rmpad-under-sdpa silently attends
   across packed sequences.
3. **Decoupled rollout correction, not bypass.** The actor recomputes
   `old_log_probs`; the captured `rollout_log_probs` enter the loss as
   verl's sequence-level truncated IS weights
   (`RolloutCorrectionConfig()` defaults: `rollout_is="sequence"`,
   threshold 2.0) — the CP-17 `--use-tis` analog, and the way the
   captured behaviour-policy values stay load-bearing. Bypass was
   rejected: it aliases the capture into `old_log_probs` wholesale
   (assertion 1's hazard) and would erase the recompute-vs-captured
   measurement that is the loop's replay-style validation (judged
   against the CP-09′ floor 0.008 / 0.21, never the CP-18 anchor).
4. **GRPO, one uid group (F-10 closed).** Every episode of the one
   golden prompt ingests with `uid="cp21-golden"`; `adv_estimator=grpo`
   with `norm_adv_by_std_in_grpo=True` (CP-20 proved the F-08 shape
   structurally immune). The bridge's default uid (= session id) is
   refused by an explicit assert — singleton groups would train on raw
   uncentred rewards while looking like GRPO.
5. **Checkpoint reload for the sync, via verl's own export.**
   `save_checkpoint(save_contents=["hf_model"])` — verl's
   `FSDPCheckpointManager` writes a directly servable HF directory —
   then `staging/serving/serve-updated.sh` (four legs unchanged, same
   served name), then the CP-17 probe with its noise floor measured
   first. NCCL resharding is unavailable by construction: the engine is
   not verl's.
6. **Optimizer/loss knobs mirror CP-17 where they correspond**: AdamW
   lr 1e-5, betas (0.9, 0.98), weight decay 0.1, grad clip 1.0 (verl
   default), clip ratios 0.2/0.28, `loss_agg_mode="token-mean"` (verl
   default, stated), entropy_coeff 0, no KL. `temperature=1.0` for the
   engine's logprob computation: the estate's captured logprobs are raw
   (CP-17's slime recompute agreed with them at 0.0088 without any
   temperature scaling); the serving temperature 0.6 is a sampling
   parameter, not a logprob scale.

## Consequence

The loop's verl leg runs in one process on one GPU with zero
Ray/rollout/critic infrastructure, and the claim "verl trained on these
traces" reduces to auditable parts: verl's worker, verl's loss, verl's
IS correction, verl's estimator, verl's optimizer step, verl's HF
export — wired by ~300 lines of harness that mirror verl's own fit loop
and substitute exactly one flash-attn-bound helper (F-12, asserted).
What this shape deliberately does not test: verl driving generation
(AgentLoop), the v1 TransferQueue pipeline, colocation, NCCL weight
sync, and any multi-step schedule. Those are different experiments, now
cheaper to design because the boundary evidence exists.
