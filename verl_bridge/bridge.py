"""Callback-shaped `SessionResult` → verl `DataProto` — the trainer's code.

Written at CP-20 of `gsj-harness-rollout-server` *for* the evaluation, and
owned by the trainer by the scope law (library ADR-0018 precedent; ADR-0003
here). The target shape is exactly what verl's own agent-loop worker hands
the trainer after a multi-turn rollout — read from verl @ 1ae9455 (the SHA
uni-agent pins as its submodule), not invented:

- ``prompts``            [B, P]   int64, LEFT-padded to a fixed width
- ``responses``          [B, R]   int64, RIGHT-padded
- ``input_ids``          [B, P+R] int64, cat(prompts, responses)
- ``attention_mask``     [B, P+R] int64, 0 at pads, 1 at real tokens
- ``position_ids``       [B, P+R] int64, clip(cumsum(attention)-1, min=0)
- ``response_mask``      [B, R]   int64, 1 ONLY at LLM-generated tokens
- ``rollout_log_probs``  [B, R]   float32, 0.0 at non-generated positions
- ``rm_scores``          [B, R]   float32, outcome reward at the last real
  response token, zeros elsewhere

(`verl/experimental/agent_loop/agent_loop.py`: `_agent_loop_postprocess`
lines 741–878 for the per-sample padding, `_postprocess` lines 1050–1092
for the batch and the rm_scores placement. uni-agent's trainer-side field
schema — `framework.py` `_trajectory_to_tq_field_and_tag` — names the same
keys, a second in-tree confirmation of the contract.)

The batch carries BOTH mask keys, equal — ``response_mask`` and
``loss_mask`` — because verl at this pin is mid-migration between two
trainer generations and each generation load-bears a different one. The
classic trainer (`main_ppo_v0`, the documented padded-batch contract this
bridge targets) aggregates every loss term under ``response_mask``
(`core_algos.py::agg_loss`) and derives the key from `attention_mask`
ONLY when absent (`ray_trainer.py`: `if "response_mask" not in
batch.batch.keys()`) — a bridge-provided mask is consumed verbatim. The
v1 TransferQueue trainer's engine normalizes token-mean loss by summing
``loss_mask`` (`workers/engine/fsdp/transformer_impl.py:676-681`). Both
in-tree writers — verl's own `AgentLoopWorkerTQ` ("TODO: uniform
response_mask and loss_mask") and uni-agent's field builder — write the
two keys equal; this bridge does the same.

The bridge produces **rewards, not advantages** (`rm_scores`): advantage
estimation is config-owned inside verl's fit loop (`compute_advantage`,
grouping rows by the ``uid`` non-tensor key), and producing advantages
here would bypass `algorithm.adv_estimator` and re-own exactly the
numerics F-08 showed a vendored post-processor getting wrong.

Three assertions, each with a test that fails when removed — same three
as the slime bridge (library CP-16), because the threat model did not
change with the trainer:

1. **Mask before ratio** (`assert_mask_before_ratio`). Interstitial
   positions carry ``0.0`` placeholders, which read as probability 1.0.
   In verl this is *more* dangerous than in slime: with
   ``algorithm.rollout_correction`` in bypass mode the trainer copies
   ``rollout_log_probs`` INTO ``old_log_probs`` and computes importance
   ratios from them internally (`ray_trainer.py` fit loop: "Bypass mode:
   Sets old_log_probs = rollout_log_probs") — so the discipline is
   asserted at ingest, before verl ever sees the array.
2. **Sentinel rejection** (`assert_sentinel_free`). ``-9999.0`` is finite
   and <= 0, so a naive discipline admits it; CP-02 established upstream
   has NO value-level logprob validation anywhere, and neither does verl's
   agent-loop path (`AgentLoopOutput.response_logprobs` is consumed
   presence-checked only). Rejected at the bridge's own ``-9000.0`` floor,
   independent of whatever `CheckPolicy` the server ran.
3. **`checks` runs trainer-side.** `ingest_session_result` calls the same
   `gsj_rollout.checks.validate_session_result` the receiver ran, on what
   actually arrived — law 6's trainer leg, from the wheel (ADR-0017).

Rejection is two-tier, exactly the slime bridge's split:

- **Pipeline poison raises** (`BridgeAssertionError` subclasses): a
  sentinel, a broken placeholder discipline, absent/misaligned logprobs
  or mask on a session that claims to be trainable.
- **Episode badness masks**: non-empty `checks` findings, TIMEOUT/ERROR,
  or any unknown status convert to fully-masked rows — kept in the batch,
  auditable, never trainable. Zero gradient is **verl's own mechanism**,
  not an invention: an all-zero ``response_mask`` row contributes exactly
  0 to every `agg_loss` mode (token-mean sums under the mask; seq-mean
  modes exclude fully-masked sequences via ``(mask-sum > 0)``), and the
  advantage broadcast multiplies by ``response_mask``. verl itself
  neutralizes rejection-sampled and synthetic-padding samples the same
  way; there is no sample-drop status key in the agent-loop path to use
  instead.

Status handling is **fail-closed**, a deliberate change from the vendored
slime adapter's fall-through-to-COMPLETED: only ``"COMPLETED"`` on both
session and trajectory is trainable; ``finish_reason == "length"`` stays
trainable (truncated-but-real, exercised live at CP-17); anything else —
including a status string this bridge has never seen — masks.

Deliberately omitted: no store, no staleness tracking, no scheduler, no
weight sync, no submits, no reward computation, **no replay** (replay
needs the serving engine; CP-20 is fixture-driven by decree). Any
replay-style validation CP-21 adds must use the CP-09'-measured floor
exported below, never the CP-18 anchor (mean 0.005 / per-position 0.05),
which fails as written on CUDA continuous-batching capture.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from tensordict import TensorDict

from gsj_rollout import checks
from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

# The bridge's own value-level floor — deliberately NOT read from the
# server's CheckPolicy, so a loosened policy cannot loosen the trainer.
SENTINEL_THRESHOLD = -9000.0

# CP-09' (H200, the governing estate): capture-vs-replay noise floor as
# measured — golden mean|Δ| 0.005246, collected 0.007141, per-position
# tail to 0.2107, replay-vs-replay exactly 0.0. CP-21 inherits these
# numbers, not the CP-18 anchor.
H200_REPLAY_FLOOR_MEAN = 0.008
H200_REPLAY_FLOOR_PER_POSITION = 0.21

# The verl surface this bridge was read against: uni-agent's own submodule
# pin (verl 0.9.0.dev). Recorded so CP-21 installs the surface the bridge
# was written for, not whatever master has become.
VERL_SHA = "1ae945592754cbeb1350cbe092fe6117070fd4c7"


class BridgeAssertionError(ValueError):
    """A trace that must not reach training arrived at the bridge."""


class SentinelLogprobError(BridgeAssertionError):
    """A sentinel logprob sat at a stated-trainable position."""


class MaskDisciplineError(BridgeAssertionError):
    """Mask and logprobs disagree about which positions are data."""


class LogprobAbsentError(BridgeAssertionError):
    """A stated-trainable trace arrived without aligned logprobs."""


def assert_sentinel_free(
    logprobs: list[float], loss_mask: list[int], *,
    session_id: str = "?", trace_index: int = -1,
) -> None:
    """Assertion 2 — reject `-9999.0`-class values at ingest (CP-02)."""
    offenders = [
        index for index, (value, mask) in enumerate(zip(logprobs, loss_mask))
        if mask == 1 and value <= SENTINEL_THRESHOLD
    ]
    if offenders:
        raise SentinelLogprobError(
            f"session {session_id} trace {trace_index}: sentinel logprob "
            f"(<= {SENTINEL_THRESHOLD}) at stated-trainable positions "
            f"{offenders[:5]}... count={len(offenders)} — broken capture "
            "pipeline, refused at ingest"
        )


def assert_mask_before_ratio(
    logprobs: list[float], loss_mask: list[int], *,
    session_id: str = "?", trace_index: int = -1,
) -> None:
    """Assertion 1 — the placeholder discipline must hold before anything
    downstream may treat the array as behaviour-policy data. verl's bypass
    mode turns this array into `old_log_probs` wholesale; an unmasked
    non-zero here becomes a silent importance ratio."""
    offenders = [
        index for index, (value, mask) in enumerate(zip(logprobs, loss_mask))
        if mask == 0 and value != 0.0
    ]
    if offenders:
        raise MaskDisciplineError(
            f"session {session_id} trace {trace_index}: non-zero logprob at "
            f"masked positions {offenders[:5]}... count={len(offenders)} — "
            "mask and logprobs disagree; an unmasked importance ratio would "
            "consume these silently"
        )


def masked_behaviour_stats(
    logprobs: list[float], loss_mask: list[int]
) -> dict[str, Any]:
    """The sanctioned view: behaviour-policy values exist only under the
    mask. Every aggregate this bridge emits comes from here."""
    trainable = [
        value for value, mask in zip(logprobs, loss_mask) if mask == 1
    ]
    if not trainable:
        return {
            "trainable_positions": 0,
            "behaviour_logprob_sum": 0.0,
            "behaviour_logprob_mean": 0.0,
        }
    total = sum(trainable)
    return {
        "trainable_positions": len(trainable),
        "behaviour_logprob_sum": total,
        "behaviour_logprob_mean": total / len(trainable),
    }


@dataclass
class TraceRecord:
    """One validated, unpadded trace — `ingest_session_result`'s output and
    `build_batch`'s input. Plain lists, verbatim from the callback body;
    tensors exist only after batching."""

    session_id: str
    trace_index: int
    uid: str
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    rollout_log_probs: list[float]
    reward: float
    status: str
    masked_reason: str  # "" when trainable
    findings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def trainable(self) -> bool:
        return self.masked_reason == ""


def ingest_session_result(
    body: Mapping[str, Any],
    *,
    uid: str | None = None,
    policy: Any = None,
) -> list[TraceRecord]:
    """One callback-shaped `SessionResult` → `TraceRecord`s, one per trace.

    Assertion 3 first: the same `validate_session_result` the receiver ran,
    on what actually arrived. All records of one session share ``uid`` —
    verl's advantage estimators group rows by the ``uid`` non-tensor key,
    so uid assignment IS the group semantics (see the singleton-group note
    in the run book: a group of one gets mean 0/std 1, i.e. its RAW reward
    as its advantage). Default uid is the session id; the loop overrides
    it when episodes of one prompt should baseline each other.

    A session with no usable trace contributes nothing, loudly — verl has
    no group-shape constraint that would need a placeholder row (uid
    groups shrink harmlessly), so the slime bridge's `remove_sample`
    mechanism and its hardcoded `tokens=[0, 0]` (F-05) have no counterpart
    here.
    """
    findings = checks.validate_session_result(body, policy)
    _warn_if_packaged_pins_mismatch(findings)

    session_id = str(body.get("session_id"))
    group_uid = uid if uid is not None else session_id
    trajectory = body.get("trajectory") if isinstance(body.get("trajectory"), Mapping) else {}
    traces = trajectory.get("traces") if isinstance(trajectory.get("traces"), list) else []

    records: list[TraceRecord] = []
    for trace_index, trace in enumerate(traces):
        if not isinstance(trace, Mapping):
            continue  # ADM5 already among findings; nothing to convert
        record = _build_record(
            body=body, trajectory=trajectory, trace=trace,
            trace_index=trace_index, session_id=session_id,
            uid=group_uid, findings=findings,
        )
        if record is not None:
            records.append(record)

    if not records:
        logger.warning(
            "session %s: no usable trace (traces=%d); session contributes "
            "nothing to the batch", session_id, len(traces),
        )
    return records


def _build_record(
    *, body: Mapping[str, Any], trajectory: Mapping[str, Any],
    trace: Mapping[str, Any], trace_index: int, session_id: str,
    uid: str, findings: list[str],
) -> TraceRecord | None:
    prompt_ids = list(trace.get("prompt_ids") or [])
    response_ids = list(trace.get("response_ids") or [])
    if not prompt_ids or not response_ids:
        logger.warning(
            "dropping trace %d of session %s: missing tokens (prompt=%d, response=%d)",
            trace_index, session_id, len(prompt_ids), len(response_ids),
        )
        return None

    status, masked_reason = _status_and_masked_reason(body, trajectory, trace)
    if masked_reason:
        # Episode badness masks: zero mask, relaxed logprobs — kept in the
        # batch so it is auditable, neutralized by verl's own mechanism.
        response_mask = [0] * len(response_ids)
        logprobs = _inert_logprobs(trace, len(response_ids))
    else:
        # The session claims to be trainable: the three assertions run on
        # the STATED mask, verbatim, BEFORE findings can soften anything
        # to masking — pipeline poison must raise, not hide behind a
        # findings-masked row.
        response_mask = _stated_mask(
            trace, len(response_ids),
            session_id=session_id, trace_index=trace_index,
        )
        logprobs = _extract_rollout_log_probs(
            trace, response_len=len(response_ids), loss_mask=response_mask,
            session_id=session_id, trace_index=trace_index,
        )
        if findings:
            # Gate failures, discipline breaches: auditable, group-shape-
            # preserving, never trainable.
            status, masked_reason = "FAILED", f"findings:{len(findings)}"
            response_mask = [0] * len(response_ids)

    stats = masked_behaviour_stats(logprobs, response_mask)
    return TraceRecord(
        session_id=session_id,
        trace_index=trace_index,
        uid=uid,
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        rollout_log_probs=logprobs,
        reward=_reward_value(trace),
        status=status,
        masked_reason=masked_reason,
        findings=list(findings),
        metadata=_record_metadata(
            body=body, trajectory=trajectory, trace=trace,
            trace_index=trace_index, findings=findings, stats=stats,
        ),
    )


def _status_and_masked_reason(
    body: Mapping[str, Any], trajectory: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> tuple[str, str]:
    """Fail-closed status resolution. The vendored slime adapter fell
    through to COMPLETED on anything unrecognized — an unknown status
    became trainable. Here the only trainable statuses are COMPLETED on
    BOTH session and trajectory; `finish_reason == "length"` merely labels
    the row TRUNCATED (still trainable — CP-17 trained one live). Findings
    are handled by the caller AFTER the assertions have run."""
    statuses = (body.get("status"), trajectory.get("status"))
    if body.get("error") or trajectory.get("error"):
        return "FAILED", f"error:{body.get('error') or trajectory.get('error')}"
    if any(status != "COMPLETED" for status in statuses):
        return "ABORTED" if "TIMEOUT" in statuses else "FAILED", \
            f"status:{statuses[0]}/{statuses[1]}"
    if trace.get("finish_reason") == "length":
        return "TRUNCATED", ""
    return "COMPLETED", ""


def _stated_mask(
    trace: Mapping[str, Any], response_len: int, *,
    session_id: str, trace_index: int,
) -> list[int]:
    """The builder-assigned mask, verbatim — the bridge never infers
    trainable positions. Absent mask = no stated-trainable positions."""
    mask = list(trace.get("loss_mask") or [])
    if not mask:
        return [0] * response_len
    if len(mask) != response_len:
        raise MaskDisciplineError(
            f"session {session_id} trace {trace_index}: loss_mask length "
            f"{len(mask)} != response length {response_len}"
        )
    off_domain = [
        value for value in mask
        if isinstance(value, bool) or value not in (0, 1)
    ]
    if off_domain:
        raise MaskDisciplineError(
            f"session {session_id} trace {trace_index}: loss_mask values "
            f"outside {{0, 1}}: {off_domain[:5]}"
        )
    return mask


def _extract_rollout_log_probs(
    trace: Mapping[str, Any], *, response_len: int, loss_mask: list[int],
    session_id: str, trace_index: int,
) -> list[float]:
    """Behaviour-policy values for a stated-trainable trace. Assertions 1
    and 2 live HERE, on every trainable ingest path — not in a comment."""
    logprobs = trace.get("response_logprobs")
    if not logprobs:
        if any(loss_mask):
            raise LogprobAbsentError(
                f"session {session_id} trace {trace_index}: response_logprobs "
                "absent for stated-trainable tokens"
            )
        return [0.0] * response_len
    if len(logprobs) != response_len:
        raise LogprobAbsentError(
            f"session {session_id} trace {trace_index}: response_logprobs "
            f"length {len(logprobs)} != response length {response_len}"
        )
    values = [float(value) for value in logprobs]
    assert_sentinel_free(
        values, loss_mask, session_id=session_id, trace_index=trace_index)
    assert_mask_before_ratio(
        values, loss_mask, session_id=session_id, trace_index=trace_index)
    return values


def _inert_logprobs(trace: Mapping[str, Any], response_len: int) -> list[float]:
    """Logprobs for a zero-masked row: kept when aligned (audit value),
    zeroed otherwise — they can contribute nothing either way."""
    logprobs = trace.get("response_logprobs")
    if isinstance(logprobs, list) and len(logprobs) == response_len:
        try:
            return [float(value) for value in logprobs]
        except (TypeError, ValueError):
            pass
    return [0.0] * response_len


def _reward_value(trace: Mapping[str, Any]) -> float:
    """Read what the evaluator placed on the trace; assignment is the
    evaluator's job. Every real body to date carries `null` → 0.0 — the
    reward attach step is the loop's (FINDINGS F-02, closed for slime at
    CP-17 by `reward_cited_pages.py`; CP-21 reuses it)."""
    reward = trace.get("reward")
    return float(reward) if isinstance(reward, (int, float)) and not isinstance(reward, bool) else 0.0


def _record_metadata(
    *, body: Mapping[str, Any], trajectory: Mapping[str, Any],
    trace: Mapping[str, Any], trace_index: int,
    findings: list[str], stats: dict[str, Any],
) -> dict[str, Any]:
    trajectory_metadata = trajectory.get("metadata")
    polar: dict[str, Any] = {
        "node_id": body.get("node_id"),
        "result_metadata": deepcopy(dict(body.get("metadata") or {})),
        "result_error": body.get("error"),
        "session_id": body.get("session_id"),
        "session_status": body.get("status"),
        "task_id": body.get("task_id"),
        "timing": deepcopy(dict(body.get("timing") or {})),
        "trace_index": trace_index,
        "trajectory_error": trajectory.get("error"),
        "trajectory_metadata": deepcopy(
            dict(trajectory_metadata) if isinstance(trajectory_metadata, Mapping) else {}
        ),
        "trajectory_status": trajectory.get("status"),
        "trace_metadata": deepcopy(dict(trace.get("metadata") or {})),
        "finish_reason": trace.get("finish_reason"),
    }
    polar.update(_scheduler_metadata(body, trajectory, trace))
    return {
        "polar": polar,
        "gsj": {
            "findings": list(findings),
            "pins_path": str(checks.PINS_PATH),
            "packaged_pins": _using_packaged_pins(),
            **stats,
        },
    }


def _scheduler_metadata(
    body: Mapping[str, Any], trajectory: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    """`group_id` / `policy_version` / `rollout_step`, wherever stamped.
    Nothing declares `policy_version` today (P3 inert) — the moment
    collection and training overlap, CP-21 needs it live (A-13)."""
    keys = ("group_id", "policy_version", "rollout_step")
    merged: dict[str, Any] = {}
    for source in (body.get("metadata"), trajectory.get("metadata"), trace.get("metadata")):
        if isinstance(source, Mapping):
            for key in keys:
                if key in source:
                    merged[key] = source[key]
    return merged


def _using_packaged_pins() -> bool:
    packaged = getattr(checks, "PACKAGED_PINS", None)
    return packaged is not None and checks.PINS_PATH == packaged


def _warn_if_packaged_pins_mismatch(findings: list[str]) -> None:
    """ADR-0017's loudness, consumer half: hash gates failing against the
    PACKAGED pins almost certainly means this estate is not the shipping
    estate — say so, next to the findings."""
    if findings and _using_packaged_pins() and any(
        "not_approved" in finding for finding in findings
    ):
        logger.error(
            "hash gates failed against the packaged pins (%s) — those are "
            "the shipping estate's approved sets; set GSJ_PINS_PATH to this "
            "estate's pins before the first import of gsj_rollout.checks. "
            "findings: %s", checks.PINS_PATH, findings,
        )


def build_batch(
    records: list[TraceRecord],
    *,
    prompt_length: int | None = None,
    response_length: int | None = None,
    pad_token_id: int = 0,
) -> DataProto:
    """`TraceRecord`s → one verl `DataProto`, padded per verl's own
    agent-loop conventions (module docstring). This is the stage slime
    never needed: slime takes per-sample objects; verl wants `[B, L]`.

    Widths mirror verl's `rollout.prompt_length` / `rollout.response_length`
    (fixed config widths); ``None`` derives each from the batch maximum. A
    trace longer than a stated width is DROPPED loudly and recorded in
    ``meta_info["oversize_dropped"]`` — never clipped: a clipped mask/
    logprob array would still align mechanically, but the row would claim
    a behaviour-policy measurement over tokens the policy never emitted
    in that shape. (The slime bridge dropped oversize traces too; the
    difference is the batch now carries the audit trail.)

    ``pad_token_id`` defaults to 0 exactly like verl's own fallback when
    the tokenizer declares none (`agent_loop.py::_pad_token_ids`); unlike
    slime's F-05 placeholder, every pad position here is attention-masked,
    so the id never reaches compute.
    """
    if not records:
        raise ValueError("build_batch needs at least one record")

    max_prompt = max(len(record.prompt_ids) for record in records)
    max_response = max(len(record.response_ids) for record in records)
    prompt_width = prompt_length if prompt_length is not None else max_prompt
    response_width = response_length if response_length is not None else max_response

    kept: list[TraceRecord] = []
    oversize_dropped: list[dict[str, Any]] = []
    for record in records:
        if len(record.prompt_ids) > prompt_width or len(record.response_ids) > response_width:
            logger.warning(
                "dropping trace %d of session %s: prompt=%d/response=%d exceeds "
                "widths %d/%d — dropped wholesale, never clipped",
                record.trace_index, record.session_id,
                len(record.prompt_ids), len(record.response_ids),
                prompt_width, response_width,
            )
            oversize_dropped.append({
                "session_id": record.session_id, "trace_index": record.trace_index,
                "prompt_len": len(record.prompt_ids), "response_len": len(record.response_ids),
            })
            continue
        kept.append(record)
    if not kept:
        raise ValueError(
            f"build_batch: every record exceeded widths {prompt_width}/{response_width}"
        )
    if not any(any(record.response_mask) for record in kept):
        # Legal (an all-audit batch), but a training step on it is a
        # hazard, not a no-op: token-mean's live path divides by the
        # GLOBAL valid-token count, and seq-mean modes set
        # global_batch_size = seq_mask.sum() == 0 when every sequence is
        # fully masked (core_algos.agg_loss) — division by zero.
        logger.warning(
            "build_batch: every row is fully masked — nothing here is "
            "trainable, and an all-masked batch divides by zero in verl's "
            "seq-mean agg modes"
        )

    rows_prompts, rows_responses, rows_prompt_attn, rows_response_attn = [], [], [], []
    rows_response_mask, rows_logprobs, rows_rm_scores = [], [], []
    for record in kept:
        prompt_pad = prompt_width - len(record.prompt_ids)
        response_pad = response_width - len(record.response_ids)

        # Prompt LEFT-padded, response RIGHT-padded — the agent-loop layout.
        rows_prompts.append([pad_token_id] * prompt_pad + record.prompt_ids)
        rows_prompt_attn.append([0] * prompt_pad + [1] * len(record.prompt_ids))
        rows_responses.append(record.response_ids + [pad_token_id] * response_pad)
        rows_response_attn.append([1] * len(record.response_ids) + [0] * response_pad)
        rows_response_mask.append(record.response_mask + [0] * response_pad)
        rows_logprobs.append(record.rollout_log_probs + [0.0] * response_pad)

        # Outcome reward at the LAST REAL response token (`_postprocess`
        # lines 1086–1092 place it at attention_mask[:, P:].sum(-1) - 1,
        # which is exactly len(response_ids) - 1 under right-padding).
        rm_scores = [0.0] * response_width
        rm_scores[len(record.response_ids) - 1] = record.reward
        rows_rm_scores.append(rm_scores)

    prompts = torch.tensor(rows_prompts, dtype=torch.int64)
    responses = torch.tensor(rows_responses, dtype=torch.int64)
    prompt_attention = torch.tensor(rows_prompt_attn, dtype=torch.int64)
    response_attention = torch.tensor(rows_response_attn, dtype=torch.int64)
    response_mask = torch.tensor(rows_response_mask, dtype=torch.int64)
    rollout_log_probs = torch.tensor(rows_logprobs, dtype=torch.float32)
    rm_scores = torch.tensor(rows_rm_scores, dtype=torch.float32)

    input_ids = torch.cat([prompts, responses], dim=1)
    attention_mask = torch.cat([prompt_attention, response_attention], dim=1)
    # verl's own helper, not a re-derivation: clip(cumsum(mask)-1, min=0).
    position_ids = compute_position_id_with_mask(attention_mask)

    _assert_padded_alignment(
        kept, response_mask=response_mask,
        response_attention=response_attention,
        rollout_log_probs=rollout_log_probs,
    )

    batch = TensorDict(
        {
            "prompts": prompts,                      # [B, P]
            "responses": responses,                  # [B, R]
            "response_mask": response_mask,          # [B, R] — v0 loss key
            "loss_mask": response_mask.clone(),      # [B, R] — v1 normalizer key, kept equal
            "input_ids": input_ids,                  # [B, P+R]
            "attention_mask": attention_mask,        # [B, P+R]
            "position_ids": position_ids,            # [B, P+R]
            "rollout_log_probs": rollout_log_probs,  # [B, R]
            "rm_scores": rm_scores,                  # [B, R]
        },
        batch_size=len(kept),
    )
    non_tensor_batch = {
        "uid": np.array([record.uid for record in kept], dtype=object),
        "session_id": np.array([record.session_id for record in kept], dtype=object),
        "trace_index": np.array([record.trace_index for record in kept], dtype=np.int64),
        "status": np.array([record.status for record in kept], dtype=object),
        "masked_reason": np.array([record.masked_reason for record in kept], dtype=object),
        "gsj": np.array([record.metadata["gsj"] for record in kept], dtype=object),
        "polar": np.array([record.metadata["polar"] for record in kept], dtype=object),
    }
    data = DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        meta_info={
            "prompt_length": prompt_width,
            "response_length": response_width,
            "pad_token_id": pad_token_id,
            "oversize_dropped": oversize_dropped,
        },
    )
    data.check_consistency()
    return data


def _assert_padded_alignment(
    records: list[TraceRecord], *, response_mask: torch.Tensor,
    response_attention: torch.Tensor, rollout_log_probs: torch.Tensor,
) -> None:
    """Batching invariants, cheap and construction-time: padding must not
    manufacture trainable positions or behaviour-policy values. (The three
    ingest assertions guard the arrays; this guards the stacking.)"""
    if bool((response_mask * (1 - response_attention)).any()):
        raise MaskDisciplineError(
            "padding produced a trainable position outside the real response"
        )
    if bool((rollout_log_probs * (1 - response_attention).float()).any()):
        raise MaskDisciplineError(
            "padding produced a non-zero logprob at a padded position"
        )
    for row, record in enumerate(records):
        length = len(record.response_ids)
        if response_mask[row, :length].tolist() != record.response_mask:
            raise MaskDisciplineError(
                f"row {row}: response_mask shifted during padding"
            )


def ingest_and_build(
    bodies: list[Mapping[str, Any]],
    *,
    uids: list[str] | None = None,
    policy: Any = None,
    prompt_length: int | None = None,
    response_length: int | None = None,
    pad_token_id: int = 0,
) -> DataProto:
    """Convenience: many callback bodies → one verl batch. ``uids`` pairs
    with ``bodies`` BY KEY on each record (every record carries its uid
    from ingest onward) — group↔result pairing is explicit, never
    positional-after-the-fact."""
    if uids is not None and len(uids) != len(bodies):
        raise ValueError(f"uids ({len(uids)}) and bodies ({len(bodies)}) must pair 1:1")
    records: list[TraceRecord] = []
    for index, body in enumerate(bodies):
        records.extend(ingest_session_result(
            body, uid=uids[index] if uids is not None else None, policy=policy,
        ))
    return build_batch(
        records, prompt_length=prompt_length,
        response_length=response_length, pad_token_id=pad_token_id,
    )
