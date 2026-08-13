"""loop.py — the verl fit-loop machinery behind `example_project/train.py`.

Extracted at CP-25 from `cp21_loop/train_one_step.py` (the CP-21 evidence
artifact, kept verbatim) so a consumer's script can read as collect →
convert → train → sync while the verl plumbing sits here, documented.
Every numeric operation is verl's own, called directly at
`bridge.VERL_SHA`; this module wires, it never re-implements — with ONE
deliberate substitution, `to_no_padding` (F-12, below).

What deliberately does NOT live here, because a consumer must see it in
their script rather than inherit it silently (CP-25's rule):
  - uid grouping (F-10): the caller chooses uids at ingest; singleton
    groups get raw-reward "advantages" under verl's GRPO.
  - reward attach (F-02): callback bodies carry `reward: null`; the caller
    grades BEFORE ingest or trains on zeros.
  - the three bridge assertions: they run inside
    `bridge.ingest_session_result`, at the caller's call site.
  - entropy/KL control: `make_worker` defaults both OFF (the measured
    one-step shape); a multi-step run must turn them on — CP-21 watched
    the post-sync distribution visibly narrow without them.

Heavy imports (torch, verl) happen inside functions so the desk half of a
caller (`--dry-run`, collection) runs on a CPU-only box.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

MAX_TOKEN_LEN = 32768  # = the serving window; no collected trace exceeds it
TEMPERATURE = 1.0      # raw-logprob convention: captured values are raw
                       # (CP-17 measured slime's raw recompute at 0.0088);
                       # the engine divides logits by this, so 1.0
                       # recomputes in the captured convention


def load_reward_module(path: str | Path):
    """Load a reward module by EXPLICIT file path, never by import name.

    Both bridge directories name their module `bridge.py`, and the vendored
    Polar tree carries a package named `slime_bridge` — sys.path import
    would shadow-load the wrong file silently (the CP-21 lesson). The
    shipped grader is `slime_bridge/reward_cited_pages.py` (F-02's answer):
    `grade_session(body, artifacts_root, *, cutoff, page_count)` sets
    `trace["reward"]` in memory on every trace of the session.
    """
    path = Path(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- F-12 port
def to_no_padding(td):
    """`left_right_2_no_padding`'s exact output contract, pure torch (F-12).

    verl's own helper (`workers/utils/padding.py`) hard-requires flash-attn
    with no torch fallback; a host without it cannot run the classic fit
    loop as written. This reproduces the helper field for field and asserts
    the contract row by row. The engine must then run flash-free too:
    `use_remove_padding=False` + sdpa (rmpad-under-sdpa silently attends
    ACROSS packed sequences — wrong logprobs, no crash).
    """
    import torch
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


# ------------------------------------------------------------------- worker
def make_worker(snapshot: str, *, max_token_len: int = MAX_TOKEN_LEN,
                lr: float = 1e-5, betas=(0.9, 0.98), weight_decay: float = 0.1,
                clip_ratio_low: float = 0.2, clip_ratio_high: float = 0.28,
                entropy_coeff: float = 0.0, use_kl_loss: bool = False):
    """verl's own `TrainingWorker`, standalone (WORLD_SIZE=1, no Ray), wired
    exactly as `ActorRolloutRefWorker.init_model` wires the classic path.
    Defaults are the CP-17/CP-21 measured knobs. entropy_coeff/use_kl_loss
    default OFF — right for one audited step, WRONG for a multi-step run
    (the caller's script must say so, not this signature).
    """
    import torch
    from functools import partial
    from verl.trainer.config.config import CheckpointConfig
    from verl.workers.config import (
        ActorConfig, FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig,
        TrainingWorkerConfig)
    from verl.workers.engine_workers import TrainingWorker
    from verl.workers.utils.losses import ppo_loss

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.cuda.set_device(0)

    worker = TrainingWorker(TrainingWorkerConfig(
        model_type="language_model",
        model_config=HFModelConfig(
            path=snapshot,
            use_remove_padding=False,                   # the flash-free branch (F-12)
            override_config={"attn_implementation": "sdpa"},
        ),
        engine_config=FSDPEngineConfig(
            strategy="fsdp", use_dynamic_bsz=True,
            max_token_len_per_gpu=max_token_len,
            infer_max_token_len_per_gpu=max_token_len,
            entropy_from_logits_with_chunking=True,     # dead on this branch (F-13)
        ),
        optimizer_config=FSDPOptimizerConfig(
            lr=lr, betas=betas, weight_decay=weight_decay),
        checkpoint_config=CheckpointConfig(save_contents=["hf_model"]),
    ))
    worker.reset()                                      # builds model + optimizer
    worker.set_loss_fn(partial(ppo_loss, config=ActorConfig(
        strategy="fsdp", rollout_n=1, use_dynamic_bsz=True,
        ppo_max_token_len_per_gpu=max_token_len,
        ppo_infer_max_token_len_per_gpu=max_token_len,
        clip_ratio=clip_ratio_low, clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
        loss_agg_mode="token-mean",
        entropy_coeff=entropy_coeff, use_kl_loss=use_kl_loss,
        ppo_epochs=1, shuffle=False,
    )))
    return worker


def stamp_meta(data) -> None:
    """The fit loop's own meta_info (ray_trainer.py:1528 + 1304-1306)."""
    import torch
    data.meta_info["global_token_num"] = torch.sum(
        data.batch["attention_mask"], dim=-1).tolist()
    data.meta_info["temperature"] = TEMPERATURE
    data.meta_info["multi_turn"] = False


def recompute_old_log_probs(data, worker, *, floor_mean: float,
                            floor_per_position: float) -> dict:
    """Decoupled recompute of `old_log_probs` by the training engine
    (`_compute_old_log_prob`), then the replay-style validation against the
    MEASURED capture floor — never a paper anchor. `calculate_entropy` is
    False by measurement: the non-rmpad branch ignores the chunking flag
    and a 30k-token row's full-vocab entropy temporaries OOM'd an H200
    twice (F-13). Returns the logprob-space delta stats; the caller decides
    what to do when the mean sits above `floor_mean`.
    """
    from verl.utils import tensordict_utils as tu
    from verl.utils.debug.metrics import calculate_debug_metrics
    from verl.workers.utils.padding import no_padding_2_padding

    infer_td = to_no_padding(data.to_tensordict())
    tu.assign_non_tensor(infer_td, calculate_entropy=False,
                         calculate_sum_pi_squared=False, compute_loss=False)
    infer_out = worker.infer_batch(infer_td)
    data.batch["old_log_probs"] = no_padding_2_padding(
        tu.get(infer_out, "log_probs"), infer_td).float()

    mask = data.batch["response_mask"].bool()
    lp_diff = (data.batch["old_log_probs"] - data.batch["rollout_log_probs"]).abs()[mask]
    return {
        "mean_abs": lp_diff.mean().item(),
        "max_abs": lp_diff.max().item(),
        "positions_over_floor": int((lp_diff > floor_per_position).sum()),
        "positions": int(mask.sum()),
        "floor_mean": floor_mean,
        "floor_per_position": floor_per_position,
        "verl_debug_metrics": {k: float(v) for k, v in
                               calculate_debug_metrics(data).items()},
    }


def rewards_correction_advantages(data, n: int) -> dict:
    """rm_scores → token rewards (fit loop 1536-1620), the DECOUPLED rollout
    correction (sequence-level truncated IS from the captured
    behaviour-policy logprobs — bypass mode is refused: it aliases capture
    into old_log_probs, assertion 1's hazard), then GRPO advantages grouped
    by the bridge's uid key (1637-1650). Returns per-row advantage stats —
    the caller should look at them: all-zero means the reward never fired.
    """
    from verl.trainer.config.algorithm import RolloutCorrectionConfig
    from verl.trainer.ppo.core_algos import AdvantageEstimator
    from verl.trainer.ppo.ray_trainer import compute_advantage
    from verl.trainer.ppo.reward import extract_reward
    from verl.trainer.ppo.rollout_corr_helper import (
        compute_rollout_correction_and_add_to_batch)

    reward_tensor, _ = extract_reward(data)
    data.batch["token_level_scores"] = reward_tensor
    data.batch["token_level_rewards"] = data.batch["token_level_scores"]
    data, is_metrics = compute_rollout_correction_and_add_to_batch(
        data, RolloutCorrectionConfig())
    data = compute_advantage(data, adv_estimator=AdvantageEstimator.GRPO,
                             gamma=1.0, lam=1.0, num_repeat=1,
                             norm_adv_by_std_in_grpo=True, config=None)
    row_adv = []
    for i in range(n):
        m = data.batch["response_mask"][i].bool()
        row_adv.append(data.batch["advantages"][i][m][0].item() if m.any() else 0.0)
    return {
        "rollout_corr_metrics": {k: float(v) for k, v in sorted(is_metrics.items())},
        "row_advantages": row_adv,
        "advantages": {"n": n, "min": min(row_adv), "max": max(row_adv),
                       "mean": sum(row_adv) / n,
                       "nonzero": sum(1 for a in row_adv if a != 0.0)},
    }


def train_one_step(data, worker, n: int) -> dict:
    """Exactly one optimizer step: one mini-batch = the whole collection,
    one epoch (`_update_actor`, 1302-1351). Returns the flattened metrics;
    the caller owns the grad_norm assertion (visible, not inherited).
    """
    import torch
    from verl.utils import tensordict_utils as tu

    train_td = to_no_padding(data.to_tensordict())
    tu.assign_non_tensor(
        train_td,
        calculate_entropy=False,               # entropy_coeff == 0 (F-13)
        global_batch_size=n, mini_batch_size=n, epochs=1, seed=42,
        dataloader_kwargs={"shuffle": False}, compute_loss=True)
    train_out = worker.train_mini_batch(train_td)

    flat = {}
    for key, value in tu.get(train_out, "metrics").items():
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
    return flat


def save_hf_export(worker, ckpt_dir: str) -> str:
    """verl's own `FSDPCheckpointManager` writes a directly servable HF dir
    (`save_contents=["hf_model"]`) — the sync mechanism both measured loops
    used: point the estate's `serve-updated.sh` at the returned path.
    """
    worker.save_checkpoint(ckpt_dir, global_step=1, max_ckpt_to_keep=1)
    hf_dir = os.path.join(ckpt_dir, "huggingface")
    assert os.path.exists(os.path.join(hf_dir, "config.json")), hf_dir
    return hf_dir
