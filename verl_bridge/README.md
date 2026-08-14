# verl_bridge — callback-shaped `SessionResult` → verl `DataProto`

> **Mode note (library CP-31)**: every number in this document — the
> collection-cadence figures included — was measured **thinking-off**,
> the only mode that existed before library CP-30. The shipped default
> in `example_project/` is now ON, which changes wall clock (~2.7×),
> token counts (~2×), and the G6 pins both legs must resolve — current
> per-mode expectations live in `example_project/RUNBOOK.md` §Thinking
> (F-42).

Built at library CP-20 (M6a: the second trainer). The slime bridge
(`../slime_bridge/`, CP-16) is the template; the target class changed,
the three assertions did not. ADR-0003 records the route decision —
direct conversion, after reading uni-agent @ `73b0f41` and finding its
trainer-side path cannot ingest externally-produced trajectories (it
generates into TransferQueue; it does not convert).

**verl surface**: pinned at `1ae945592754cbeb1350cbe092fe6117070fd4c7` —
uni-agent's own submodule pin, verl 0.9.0.dev (`bridge.VERL_SHA`). Real
in the tests, no double (contrast FINDINGS F-03 for slime).

## Run book

```
python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install ../../gsj-harness-rollout-server/dist/gsj_harness_rollout_server-0.1.0-py3-none-any.whl
./.venv/bin/pip install --no-deps "verl @ git+https://github.com/volcengine/verl.git@1ae945592754cbeb1350cbe092fe6117070fd4c7"
./.venv/bin/python -m pytest -q          # 26 passed — no GPU, no estate
```

Usage, exactly as a trainer would:

```python
import bridge
records = bridge.ingest_session_result(body, uid="prompt-group-3")  # checks + 3 assertions
data = bridge.build_batch(records, prompt_length=4096, response_length=32768,
                          pad_token_id=151643)                      # a real DataProto
```

## The conversion (Trace field → DataProto)

| Trace/body field | DataProto key | Rule |
| --- | --- | --- |
| `prompt_ids` | `prompts` [B, P] · left half of `input_ids` | verbatim ids, LEFT-padded to `prompt_length`; never retokenized |
| `response_ids` | `responses` [B, R] · right half of `input_ids` | verbatim ids, RIGHT-padded to `response_length` |
| `loss_mask` | `response_mask` + `loss_mask` [B, R], equal | builder's mask **verbatim** (never inferred); absent → all-zero; zeroed wholesale on non-COMPLETED status / errors / findings. Two keys because verl is mid-migration (F-11): the classic trainer's losses aggregate under `response_mask`, the v1 engine's loss normalizer sums `loss_mask`; verl's own TQ writer and uni-agent both keep them equal |
| `response_logprobs` | `rollout_log_probs` [B, R] f32 | behaviour-policy values, asserted at ingest; `0.0` at pads and on inert paths; consumed by `algorithm.rollout_correction` (bypass mode copies them INTO `old_log_probs`) |
| `reward` | `rm_scores` [B, R] f32 | evaluator's value at the LAST REAL response token (verl's own placement); `null` → `0.0` (F-02) |
| — (derived) | `attention_mask` [B, P+R] | 1 exactly on real tokens, 0 at pads |
| — (derived) | `position_ids` [B, P+R] | verl's `compute_position_id_with_mask` |
| session identity | `uid` (non-tensor) | THE advantage-group key; default = session id; see F-10 before accepting the default |
| statuses / findings / stats | `status`, `masked_reason`, `gsj`, `polar` (non-tensor) | audit trail; fail-closed: only COMPLETED×COMPLETED trains, `finish_reason=="length"` → TRUNCATED (still trainable) |
| — | `meta_info` | widths, pad id, `oversize_dropped` (dropped wholesale, never clipped, never silent) |

What the bridge does **not** produce: `old_log_probs` (the trainer's —
recomputed by the actor, or aliased from `rollout_log_probs` in bypass
mode), `advantages`/`returns` (config-owned estimators in the fit loop),
`token_level_scores` (the reward manager reads `rm_scores`).

## What CP-21 needs that this bridge deliberately does not solve

Named now so cluster time discovers nothing:

1. **The trainer-generation fork (F-11), decided BEFORE bring-up.** At
   this pin `python -m verl.trainer.main_ppo` defaults to the v1
   TransferQueue pipeline (`trainer.use_v1: true`); the padded DataProto
   this bridge builds is the classic (`main_ppo_v0`) contract —
   deprecated upstream but documented, standalone-consumable, and the
   CP-17-style one-step harness's natural input. The v1 path cannot
   ingest an externally-built batch at all: writers join the trainer's
   Ray cluster after `tq.init()` (TransferQueue==0.1.8), uids are
   MINTED BY THE TRAINER per prompt row (a bridge must echo them, never
   invent them), and the tag schema (`status`, `seq_len`,
   `min/max_global_steps` + a prompt-level status record) is read
   unconditionally. If CP-21 wants verl to drive instead, the smallest
   sanctioned seam is a custom `AgentLoop` class that calls
   `gsj-rollout submit` and returns a plain `AgentLoopOutput`
   (prompt_ids/response_ids/response_mask/response_logprobs/
   reward_score) — verl's own worker then does the padding this bridge
   does. Three options, one decision: v0-style batch feed (this bridge,
   deprecated path), v1 TQ residency (infrastructure), or AgentLoop
   (verl drives generation).
2. **verl's environment and co-residency.** verl 0.9.0.dev needs its full
   stack on the estate (Megatron or FSDP backend, Ray). CP-17 answered
   co-residency for slime by separation (vLLM serves GPU 3, training on
   GPU 5); verl defaults to OWNING its rollout engines — running it
   against our external vLLM estate means bypassing its rollout layer and
   feeding batches directly (the batch this bridge builds), which is a
   trainer-loop harness CP-21 must write (the CP-17 `train_one_step`
   analog, not a config flag).
3. **Weight sync.** CP-17 used checkpoint reload (~1 min engine
   downtime). verl's own sync paths (NCCL resharding into its rollout
   workers) are unavailable when the engine is not verl's — expect
   checkpoint reload again, with the same A-13 drain rule; P3 stamping
   goes live the moment collection and training overlap.
4. **Reward attach** (F-02): every real body carries `reward: null`.
   CP-17's `../slime_bridge/reward_cited_pages.py` grades citations
   against the cutoff; reuse it and place the value on the trace before
   ingest, or pass it into `rm_scores` via the record.
5. **uid grouping** (F-10): GRPO with the default uid (= session id)
   makes every group a singleton — mean 0 / std 1 hardcoded — so
   advantages are raw uncentred rewards. Episodes of one prompt must
   share a uid, which means the collection cadence must produce n > 1
   episodes per prompt (CP-09′ qualification budget: ≈19 submissions per
   qualifying episode under H-41, or CP-17's relaxed standard).
6. **Config verl needs that slime didn't**: `algorithm.adv_estimator`
   (default is GAE — needs a critic; GRPO/RLOO must be selected),
   `rollout.calculate_log_probs=true` (default FALSE — without it verl
   has no `rollout_log_probs` to correct against),
   `actor.policy_loss.loss_mode` / `loss_agg_mode` (default token-mean),
   and `algorithm.rollout_correction` mode (bypass vs decoupled) — in
   bypass mode our captured logprobs BECOME `old_log_probs`, which is
   why assertion 1 is enforced at ingest.
7. **Replay-style validation** must use the CP-09′-measured floor
   (`H200_REPLAY_FLOOR_MEAN = 0.008`, `H200_REPLAY_FLOOR_PER_POSITION =
   0.21`), never the CP-18 anchor (0.005/0.05) — it fails as written on
   CUDA continuous-batching capture.
