#!/usr/bin/env python3
"""CP-21: one optimizer step in real verl — the CP-17 `train_one_step` analog.

The classic-batch route (external ADR-0003, run book item 1): collected
callback bodies -> reward attach -> the CP-20 bridge -> one padded
`DataProto` -> verl's own fit-loop plumbing, mirrored step for step from
`trainer/ppo/ray_trainer.py` @ VERL_SHA (line refs below are that file
unless stated) -> exactly one optimizer step -> an HF-format export for
the estate's engine.

WHAT IS VERL'S AND WHAT IS OURS, stated precisely because the claim
"verl trained on these traces" depends on it. verl's own code, called
directly with zero numerical re-implementation:

  - `TrainingWorker` over the FSDP engine (`workers/engine_workers.py` —
    the same worker class the classic v0 trainer's actor path wraps; at
    this pin the old `DataParallelPPOActor` no longer exists),
  - `ppo_loss` (`workers/utils/losses.py`) via `set_loss_fn`, exactly as
    `ActorRolloutRefWorker.init_model` wires it,
  - `extract_reward` (reads the bridge's `rm_scores`),
  - `compute_rollout_correction_and_add_to_batch` (decoupled mode — the
    CP-17 `--use-tis` analog: sequence-level truncated IS weights from
    old_log_probs vs our captured rollout_log_probs, consumed inside
    verl's policy loss),
  - `compute_advantage` -> `compute_grpo_outcome_advantage`, grouped by
    the bridge's `uid` key (F-10: ONE shared uid, never the default),
  - `calculate_debug_metrics` (verl's own rollout-vs-recompute instrument),
  - `no_padding_2_padding` for every model output.

Ours (wiring only, no numerics): loading/grading/ingest, the config
dataclasses, the fit-loop mirroring, and ONE deliberate substitution —
`_to_no_padding()` below reproduces `workers/utils/padding.py::
left_right_2_no_padding` in pure torch, because verl's helper calls
`flash_attn.bert_padding.unpad_input` unconditionally (no torch fallback)
while this host has no nvcc and no flash-attn; the engine itself is run
on its own flash-free branch (`use_remove_padding=False` + sdpa), which
builds its attention mask internally (`transformer_impl.py:1193-1228`).
The substitution is contract-asserted row by row at run time (F-12 in the
external FINDINGS register).

Environment (all required):
    GSJ_COLLECTED_DIR    receiver-accepted SessionResult JSONs (collection 1)
    GSJ_ARTIFACTS_ROOT   pi_harness artifacts_dir (deliverables for grading)
    GSJ_CUTOFF           the submitted timestep (cutoff claim)
    GSJ_PAGE_COUNT       the case's page census (MCP /health)
    GSJ_SNAPSHOT         the pinned Qwen3-0.6B HF snapshot dir
    GSJ_CKPT_DIR         checkpoint output dir (HF export lands under
                         <GSJ_CKPT_DIR>/huggingface/)
    GSJ_SUMMARY          summary JSON path

Usage: train_one_step.py [--dry-run]   (--dry-run stops before the GPU:
ingest -> batch -> nested conversion -> asserts, CPU-only, for the desk.)
"""

from __future__ import annotations

import json
import os
import sys
from functools import partial
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))                     # verl_bridge/  (bridge)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import bridge  # noqa: E402

# CP-17's reward attach, byte-reused — loaded by explicit path because the
# slime_bridge directory also carries a (different) `bridge.py` and must
# never shadow the verl bridge on sys.path.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "reward_cited_pages",
    _HERE.parent.parent / "slime_bridge" / "reward_cited_pages.py")
_reward_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reward_mod)
grade_session = _reward_mod.grade_session

# F-10: the grouping decision, stated where it executes. One prompt (the
# golden triple) => ONE GRPO group. The bridge's default uid (= session id)
# would make every group a singleton whose "advantage" is its raw reward.
GROUP_UID = "cp21-golden"
PAD_TOKEN_ID = 151643      # Qwen3 pad; attention-masked, never reaches compute
TEMPERATURE = 1.0          # raw-logprob convention: the estate's captured values
                           # are raw (CP-17: slime's raw recompute agreed at
                           # 0.0088); the engine divides logits by this value,
                           # so 1.0 recomputes in the captured convention.
MAX_TOKEN_LEN = 32768      # = the serving window; no trace can exceed it


def log(msg: str) -> None:
    print(f"[cp21] {msg}", flush=True)


# ---------------------------------------------------------------- collection
def load_and_grade() -> tuple[list, list, dict]:
    collected = sorted(Path(os.environ["GSJ_COLLECTED_DIR"]).glob("*.json"))
    assert collected, "GSJ_COLLECTED_DIR holds no SessionResult bodies"
    artifacts_root = os.environ["GSJ_ARTIFACTS_ROOT"]
    cutoff = int(os.environ["GSJ_CUTOFF"])
    page_count = int(os.environ["GSJ_PAGE_COUNT"])

    records, grades = [], []
    for path in collected:
        body = json.loads(path.read_text())
        grade = grade_session(body, artifacts_root,
                              cutoff=cutoff, page_count=page_count)
        grades.append(grade)
        # The three assertions + trainer-side `checks` live inside ingest.
        records.extend(bridge.ingest_session_result(body, uid=GROUP_UID))

    rewards = [g["reward"] for g in grades]
    reward_dist = {
        "n": len(rewards), "min": min(rewards), "max": max(rewards),
        "mean": sum(rewards) / len(rewards),
        "nonzero": sum(1 for r in rewards if r > 0),
    }
    log(f"graded {len(collected)} sessions -> {len(records)} records; reward "
        f"min={reward_dist['min']:.4f} mean={reward_dist['mean']:.4f} "
        f"max={reward_dist['max']:.4f} "
        f"nonzero={reward_dist['nonzero']}/{reward_dist['n']}")
    return records, grades, reward_dist


# ------------------------------------------------- padded -> NO_PADDING form
def _to_no_padding(td):
    """`left_right_2_no_padding`'s exact output contract, pure torch (F-12).

    Input: the bridge batch as a TensorDict — padded `input_ids`
    `attention_mask` `response_mask` `position_ids`. Output mutations,
    matching `workers/utils/padding.py:22-92` field for field:
      - `input_ids`  -> 2-D jagged nested tensor of the REAL tokens per row
      - `position_ids` -> nested, mask-selected per row
      - `loss_mask`  -> alias of `response_mask` (padded [B, R], as verl's does)
      - non-tensor `max_seq_len`, `max_response_len`
      - padded `attention_mask`/`prompts`/`responses`/`response_mask` survive
        (verl's helper keeps them too; `no_padding_2_padding` needs them)
    """
    from verl.utils import tensordict_utils as tu

    input_ids = td.pop("input_ids")
    attention_mask = td["attention_mask"]
    response_mask = td["response_mask"]
    position_ids = td["position_ids"]

    tu.assign_non_tensor_data(td, "max_seq_len", input_ids.shape[1])
    tu.assign_non_tensor_data(td, "max_response_len", response_mask.shape[1])

    bool_mask = attention_mask.bool()
    lengths = bool_mask.sum(dim=1)
    flat_ids = input_ids[bool_mask]                    # row-major == unpad_input
    offsets = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    offsets[1:] = lengths.cumsum(0).to(torch.int32)
    ids_nested = torch.nested.nested_tensor_from_jagged(flat_ids, offsets=offsets)

    pos_rows = [position_ids[i][bool_mask[i]] for i in range(bool_mask.shape[0])]
    pos_nested = torch.nested.as_nested_tensor(pos_rows, layout=torch.jagged)

    # Contract assertions: the jagged form must be exactly the real tokens.
    assert ids_nested.offsets().diff().tolist() == lengths.tolist()
    for i in range(bool_mask.shape[0]):
        assert torch.equal(ids_nested[i], input_ids[i][bool_mask[i]])
        assert int(pos_rows[i][0]) == 0 and int(pos_rows[i][-1]) == int(lengths[i]) - 1

    td["input_ids"] = ids_nested
    td["position_ids"] = pos_nested
    td["loss_mask"] = td["response_mask"]
    return td


# ------------------------------------------------------------------ the step
def main() -> None:
    dry_run = "--dry-run" in sys.argv

    records, grades, reward_dist = load_and_grade()
    n_masked = sum(1 for r in records if not r.trainable)
    assert n_masked == 0, (
        f"{n_masked} masked rows in a collection that was pre-filtered to "
        "qualifying episodes — inspect before training (their 0.0 rewards "
        "would enter the GRPO group statistics)")

    data = bridge.build_batch(records, pad_token_id=PAD_TOKEN_ID)
    n = len(data)
    assert not data.meta_info["oversize_dropped"]
    uids = set(data.non_tensor_batch["uid"].tolist())
    assert uids == {GROUP_UID}, f"F-10 violated: uids={uids}"
    log(f"batch: {n} rows, prompt_length={data.meta_info['prompt_length']} "
        f"response_length={data.meta_info['response_length']}, one uid group")

    # ray_trainer.py:1528 + _update_actor's meta (1304-1306).
    data.meta_info["global_token_num"] = torch.sum(
        data.batch["attention_mask"], dim=-1).tolist()
    data.meta_info["temperature"] = TEMPERATURE
    data.meta_info["multi_turn"] = False

    if dry_run:
        td = _to_no_padding(data.to_tensordict())
        log(f"dry run: nested input_ids rows={td['input_ids'].size(0)}, "
            f"lengths={td['input_ids'].offsets().diff().tolist()[:5]}... OK")
        return

    # ---- the worker: verl's own, standalone (WORLD_SIZE=1, no Ray runtime)
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(0)

    from verl.protocol import DataProto  # noqa: F401  (bridge already built one)
    from verl.trainer.config.config import CheckpointConfig
    from verl.trainer.config.algorithm import RolloutCorrectionConfig
    from verl.trainer.ppo.core_algos import AdvantageEstimator
    from verl.trainer.ppo.ray_trainer import compute_advantage
    from verl.trainer.ppo.reward import extract_reward
    from verl.trainer.ppo.rollout_corr_helper import (
        compute_rollout_correction_and_add_to_batch)
    from verl.utils import tensordict_utils as tu
    from verl.utils.debug.metrics import calculate_debug_metrics
    from verl.workers.config import (
        ActorConfig, FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig,
        TrainingWorkerConfig)
    from verl.workers.engine_workers import TrainingWorker
    from verl.workers.utils.losses import ppo_loss
    from verl.workers.utils.padding import no_padding_2_padding

    snapshot = os.environ["GSJ_SNAPSHOT"]
    model_config = HFModelConfig(
        path=snapshot,
        use_remove_padding=False,                       # the flash-free branch
        override_config={"attn_implementation": "sdpa"},
    )
    engine_config = FSDPEngineConfig(
        strategy="fsdp",
        use_dynamic_bsz=True,
        max_token_len_per_gpu=MAX_TOKEN_LEN,
        infer_max_token_len_per_gpu=MAX_TOKEN_LEN,
        # A 30k-token row's full-vocab entropy temporaries OOM even an H200
        # (measured: 114 GiB in use, 28.8 GiB more requested). Chunked
        # entropy is verl's own cure for exactly this.
        entropy_from_logits_with_chunking=True,
    )
    optimizer_config = FSDPOptimizerConfig(              # CP-17's optimizer knobs
        lr=1e-5, betas=(0.9, 0.98), weight_decay=0.1,
    )
    worker = TrainingWorker(TrainingWorkerConfig(
        model_type="language_model",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=CheckpointConfig(save_contents=["hf_model"]),
    ))
    worker.reset()                                       # builds model+optimizer

    # ppo_loss config — exactly what ActorRolloutRefWorker.init_model wires.
    actor_config = ActorConfig(
        strategy="fsdp", rollout_n=1,
        use_dynamic_bsz=True,
        ppo_max_token_len_per_gpu=MAX_TOKEN_LEN,
        ppo_infer_max_token_len_per_gpu=MAX_TOKEN_LEN,
        clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.28,  # CP-17's clips
        loss_agg_mode="token-mean",                      # run book item 6, stated
        entropy_coeff=0, use_kl_loss=False,
        ppo_epochs=1, shuffle=False,
    )
    worker.set_loss_fn(partial(ppo_loss, config=actor_config))
    log("verl TrainingWorker up: FSDP(world_size=1), sdpa, dynamic bsz "
        f"@ {MAX_TOKEN_LEN} tokens/gpu")

    # ---- old_log_probs: decoupled recompute (_compute_old_log_prob, 1265-1300).
    # calculate_entropy=False, deviating from the fit loop's True: the
    # actor/entropy metric is decorative here, and at this pin the
    # non-rmpad branch IGNORES entropy_from_logits_with_chunking
    # (transformer_impl.py:1418 calls the unchunked verl_F.entropy_from_logits
    # while :1310 chunks) — full-vocab fp32 entropy temporaries on a 30k-token
    # padded micro-batch OOM an H200 (measured, twice: 114 GiB / 106 GiB in
    # use at the raise). F-13 in the external register.
    infer_td = _to_no_padding(data.to_tensordict())
    tu.assign_non_tensor(infer_td, calculate_entropy=False,
                         calculate_sum_pi_squared=False, compute_loss=False)
    infer_out = worker.infer_batch(infer_td)
    old_log_probs = no_padding_2_padding(
        tu.get(infer_out, "log_probs"), infer_td).float()
    data.batch["old_log_probs"] = old_log_probs
    log("old_log_probs recomputed by the actor (decoupled); entropy metric "
        "skipped (F-13: chunked entropy not honoured on the non-rmpad branch)")

    # ---- replay-style validation (run book item 7): the MEASURED floor.
    # verl's own instrument diffs in probability space; the floor is in
    # logprob space (nats), so both are reported.
    debug_metrics = calculate_debug_metrics(data)
    mask = data.batch["response_mask"].bool()
    lp_diff = (data.batch["old_log_probs"] - data.batch["rollout_log_probs"]).abs()[mask]
    lp_mean, lp_max = lp_diff.mean().item(), lp_diff.max().item()
    over_floor = int((lp_diff > bridge.H200_REPLAY_FLOOR_PER_POSITION).sum())
    log(f"recompute-vs-captured (logprob space): mean|Δ|={lp_mean:.6f} "
        f"max|Δ|={lp_max:.6f} positions>{bridge.H200_REPLAY_FLOOR_PER_POSITION}"
        f"={over_floor}/{int(mask.sum())} "
        f"[floor: mean {bridge.H200_REPLAY_FLOOR_MEAN}]")
    log(f"verl's own instrument (probability space): "
        f"{ {k: round(v, 6) for k, v in debug_metrics.items()} }")

    # ---- rewards -> scores -> rewards (fit loop 1536-1620, use_kl_in_reward=False)
    reward_tensor, _ = extract_reward(data)
    data.batch["token_level_scores"] = reward_tensor
    data.batch["token_level_rewards"] = data.batch["token_level_scores"]

    # ---- rollout correction, decoupled (1622-1635): sequence-level truncated
    # IS weights from our captured behaviour-policy logprobs, into the loss.
    data, is_metrics = compute_rollout_correction_and_add_to_batch(
        data, RolloutCorrectionConfig())
    log(f"rollout_corr: { {k: round(float(v), 6) for k, v in sorted(is_metrics.items())} }")

    # ---- advantages: verl's estimator registry, grouped by uid (1637-1650)
    data = compute_advantage(data, adv_estimator=AdvantageEstimator.GRPO,
                             gamma=1.0, lam=1.0, num_repeat=1,
                             norm_adv_by_std_in_grpo=True, config=None)
    row_adv = []
    for i in range(n):
        m = data.batch["response_mask"][i].bool()
        row_adv.append(data.batch["advantages"][i][m][0].item() if m.any() else 0.0)
    adv_dist = {"n": n, "min": min(row_adv), "max": max(row_adv),
                "mean": sum(row_adv) / n,
                "nonzero": sum(1 for a in row_adv if a != 0.0)}
    log(f"advantages (GRPO, one group of {n}): min={adv_dist['min']:.4f} "
        f"mean={adv_dist['mean']:.4f} max={adv_dist['max']:.4f} "
        f"nonzero={adv_dist['nonzero']}/{n}")

    # ---- the optimizer step (_update_actor, 1302-1351): one mini-batch =
    # the whole collection, one epoch => exactly one step.
    train_td = _to_no_padding(data.to_tensordict())
    tu.assign_non_tensor(
        train_td,
        calculate_entropy=False,               # entropy_coeff == 0 (1311-1313)
        global_batch_size=n, mini_batch_size=n, epochs=1, seed=42,
        dataloader_kwargs={"shuffle": False}, compute_loss=True)
    train_out = worker.train_mini_batch(train_td)
    metrics = tu.get(train_out, "metrics")
    flat = {}
    for key, value in metrics.items():
        if hasattr(value, "value"):
            value = value.value
        if isinstance(value, list):
            value = [v.value if hasattr(v, "value") else v for v in value]
            value = sum(float(x) for x in value)
        if isinstance(value, torch.Tensor):
            value = value.item()
        try:
            flat[key] = float(value)
        except (TypeError, ValueError):
            flat[key] = str(value)
    log(f"train step metrics: { {k: round(v, 8) if isinstance(v, float) else v for k, v in sorted(flat.items())} }")
    grad_norm = flat.get("grad_norm")
    if os.environ.get("GSJ_SMOKE") == "1":
        # Smoke mode: a single unrewarded fixture gives all-zero advantages,
        # so a zero gradient is the CORRECT outcome — the path, not the step,
        # is what a smoke run proves.
        log(f"SMOKE: grad_norm={grad_norm} (zero expected on a rewardless singleton)")
    else:
        assert grad_norm is not None and grad_norm > 0.0, f"grad_norm={grad_norm}"

    # ---- save: verl's own checkpoint manager writes a servable HF dir.
    ckpt_dir = os.environ["GSJ_CKPT_DIR"]
    worker.save_checkpoint(ckpt_dir, global_step=1, max_ckpt_to_keep=1)
    hf_dir = os.path.join(ckpt_dir, "huggingface")
    assert os.path.exists(os.path.join(hf_dir, "config.json")), hf_dir
    log(f"HF export (verl FSDPCheckpointManager, save_contents=['hf_model']): {hf_dir}")

    summary = {
        "n_sessions": len(grades), "n_records": n,
        "uid": GROUP_UID, "grades": grades,
        "reward_distribution": reward_dist,
        "advantage_distribution": adv_dist,
        "row_advantages": row_adv,
        "logprob_recompute_vs_captured": {
            "mean_abs": lp_mean, "max_abs": lp_max,
            "positions_over_floor": over_floor,
            "floor_mean": bridge.H200_REPLAY_FLOOR_MEAN,
            "floor_per_position": bridge.H200_REPLAY_FLOOR_PER_POSITION,
        },
        "verl_debug_metrics": {k: float(v) for k, v in debug_metrics.items()},
        "rollout_corr_metrics": {k: float(v) for k, v in is_metrics.items()},
        "train_metrics": flat,
        "hf_export": hf_dir,
        "statuses": data.non_tensor_batch["status"].tolist(),
        "trainable_positions": [int(x) for x in
                                data.batch["response_mask"].sum(dim=1)],
    }
    Path(os.environ["GSJ_SUMMARY"]).write_text(json.dumps(summary, indent=2))
    log(f"summary -> {os.environ['GSJ_SUMMARY']}")


if __name__ == "__main__":
    main()
