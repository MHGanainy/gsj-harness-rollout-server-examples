"""Callback-shaped `SessionResult` → slime `Sample` — the trainer's code.

Written at CP-16 of `gsj-harness-rollout-server` *for* the evaluation, and
owned by the trainer by the scope law ("if it ... trains — it's out";
ADR-0018 there, ADR-0001 here). The target shape is exactly what the
vendored Polar `slime_bridge/adapter.py` constructs against slime v0.3.0 —
read, not invented. Unlike that adapter this bridge consumes the
callback-shaped mapping (what a `RolloutClient.collect` actually returns),
not Polar's pydantic models: the trainer never imports `polar`.

Three assertions, each with a test that fails when removed:

1. **Mask before ratio** (`assert_mask_before_ratio`). Interstitial
   positions carry ``0.0`` placeholders, which read as probability 1.0;
   upstream says the mask is what makes the trainer ignore them
   (`polar/trajectory/builder/prefix_merging.py`: "Interstitial slots
   (tool results, chat glue) get 0.0; loss_mask=0 makes the trainer
   ignore them") — applying it is the consumer's job. The bridge verifies
   the placeholder discipline (a non-zero logprob at a masked position is
   a mask/logprob disagreement no unmasked consumer would notice) and
   computes its only behaviour-policy aggregate under the mask.
2. **Sentinel rejection** (`assert_sentinel_free`). ``-9999.0`` is vLLM's
   missing-logprob default AND its clamp floor; it is finite and <= 0, so
   a naive finite-and-nonpositive discipline admits it, and CP-02
   established upstream has NO value-level logprob validation anywhere.
   A sentinel at a stated-trainable position is a broken capture
   pipeline, not a bad episode: reject at ingest, independently of
   whatever `CheckPolicy` the server ran.
3. **`checks` runs trainer-side.** `ingest_session_result` calls the same
   `gsj_rollout.checks.validate_session_result` the receiver ran, on what
   actually arrived — law 6's trainer leg, now functional from the wheel
   (ADR-0017 there).

Rejection is two-tier, deliberately:

- **Pipeline poison raises** (`BridgeAssertionError` subclasses): a
  sentinel, a broken placeholder discipline, absent/misaligned logprobs
  or mask on a session that claims to be trainable. The vendored
  adapter's `RolloutLogprobError` precedent — the worker drops the group.
- **Episode badness masks**: non-empty `checks` findings (gate failures,
  zero-rate breaches, non-COMPLETED status) convert to fully-masked
  FAILED samples that keep group shape and carry the findings in
  `metadata["gsj"]["findings"]` — auditable, never trainable, never
  silently dropped.

Deliberately omitted — findings about the architecture if CP-17 needs
them, not licences to rebuild the predecessor: no store, no staleness
tracking, no ready grammar, no scheduler, no weight sync, **no replay**
(replay needs the serving engine; CP-16 is fixture-driven). CP-17
replay-style validation must use the CP-09'-measured floor exported
below, not the CP-18 anchor.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from gsj_rollout import checks

logger = logging.getLogger(__name__)

# The bridge's own value-level floor — deliberately NOT read from the
# server's CheckPolicy, so a loosened policy cannot loosen the trainer.
SENTINEL_THRESHOLD = -9000.0

# CP-09' (H200, the governing estate): capture-vs-replay noise floor as
# measured — golden mean|Δ| 0.005246, collected 0.007141, per-position
# tail to 0.2107, replay-vs-replay exactly 0.0. The CP-18 anchor
# (mean 0.005 / per-position 0.05) fails as written on CUDA
# continuous-batching capture; CP-17 inherits these numbers instead.
H200_REPLAY_FLOOR_MEAN = 0.008
H200_REPLAY_FLOOR_PER_POSITION = 0.21


class BridgeAssertionError(ValueError):
    """A trace that must not reach training arrived at the bridge."""


class SentinelLogprobError(BridgeAssertionError):
    """A sentinel logprob sat at a stated-trainable position."""


class MaskDisciplineError(BridgeAssertionError):
    """Mask and logprobs disagree about which positions are data."""


class LogprobAbsentError(BridgeAssertionError):
    """A stated-trainable trace arrived without aligned logprobs."""


class _Status(Enum):
    """Mirror of `slime.utils.types.Sample.Status` (v0.3.0)."""

    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class FallbackSample:
    """Test double for `slime.utils.types.Sample` (v0.3.0): exactly the
    constructor surface the vendored adapter uses, nothing more. It exists
    so the three assertions are testable off-estate without the training
    stack; the CP-17 loop runs with real slime (FINDINGS F-03/F-04)."""

    Status = _Status

    group_index: int | None = None
    index: int | None = None
    prompt: Any = ""
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    group_id: int | None = None
    reward: Any = None
    loss_mask: list[int] | None = None
    rollout_log_probs: list[float] | None = None
    status: _Status = _Status.PENDING
    remove_sample: bool = False
    session_id: str | None = None
    metadata: dict = field(default_factory=dict)


def load_sample_type() -> Any:
    """slime's `Sample` when installed; the local double otherwise."""
    try:
        from slime.utils.types import Sample  # type: ignore
    except ImportError:
        return FallbackSample
    return Sample


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
    downstream may treat the array as behaviour-policy data."""
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
    mask. The raw per-position array is slime's transport; every aggregate
    this bridge emits comes from here."""
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


def ingest_session_result(
    body: Mapping[str, Any],
    *,
    group_index: int,
    trajectory_index: int,
    reward_key: str = "score",
    max_tokens: int | None = None,
    policy: Any = None,
) -> list[Any]:
    """One callback-shaped `SessionResult` → slime `Sample`s, one per trace.

    Assertion 3 first: the same `validate_session_result` the receiver ran,
    on what actually arrived. All samples of one session share `group_id`
    (= `trajectory_index`) so slime 0.3.0's loss reducer counts the
    trajectory once. If every trace is unusable a single fully-masked
    `remove_sample=True` placeholder keeps the group's shape (the vendored
    adapter's mechanism, kept verbatim).
    """
    Sample = load_sample_type()
    findings = checks.validate_session_result(body, policy)
    _warn_if_packaged_pins_mismatch(findings)

    trajectory = body.get("trajectory") if isinstance(body.get("trajectory"), Mapping) else {}
    traces = trajectory.get("traces") if isinstance(trajectory.get("traces"), list) else []
    samples: list[Any] = []
    for trace_index, trace in enumerate(traces):
        if not isinstance(trace, Mapping):
            continue  # ADM5 already among findings; nothing to convert
        sample = _build_sample(
            Sample=Sample, body=body, trajectory=trajectory, trace=trace,
            trace_index=trace_index, group_index=group_index,
            index=trajectory_index, reward_key=reward_key,
            max_tokens=max_tokens, findings=findings,
        )
        if sample is not None:
            samples.append(sample)

    if samples:
        return samples

    logger.warning(
        "session %s: no usable trace (traces=%d); emitting removable placeholder",
        body.get("session_id"), len(traces),
    )
    return [_build_placeholder(
        Sample=Sample, body=body, trajectory=trajectory,
        group_index=group_index, index=trajectory_index,
        reward_key=reward_key, findings=findings,
    )]


def _build_sample(
    *, Sample: Any, body: Mapping[str, Any], trajectory: Mapping[str, Any],
    trace: Mapping[str, Any], trace_index: int, group_index: int, index: int,
    reward_key: str, max_tokens: int | None, findings: list[str],
) -> Any | None:
    prompt_ids = list(trace.get("prompt_ids") or [])
    response_ids = list(trace.get("response_ids") or [])
    session_id = str(body.get("session_id"))

    if not prompt_ids or not response_ids:
        logger.warning(
            "dropping trace %d of session %s: missing tokens (prompt=%d, response=%d)",
            trace_index, session_id, len(prompt_ids), len(response_ids),
        )
        return None
    if max_tokens is not None and len(prompt_ids) + len(response_ids) > max_tokens:
        logger.warning(
            "dropping trace %d of session %s: %d tokens > max_tokens=%d",
            trace_index, session_id,
            len(prompt_ids) + len(response_ids), max_tokens,
        )
        return None

    status = _session_status(Sample, body, trajectory, trace)
    if status in (Sample.Status.ABORTED, Sample.Status.FAILED):
        # Aborted/failed sessions are inert: zero mask, relaxed logprobs —
        # the vendored adapter's zero-gradient mechanism, kept verbatim.
        loss_mask = [0] * len(response_ids)
        logprobs = _inert_logprobs(trace, len(response_ids))
    else:
        # The session claims to be trainable: the three assertions run on
        # the STATED mask, before findings can soften anything to masking.
        loss_mask = _stated_mask(
            trace, len(response_ids),
            session_id=session_id, trace_index=trace_index,
        )
        logprobs = _extract_rollout_log_probs(
            trace, response_len=len(response_ids), loss_mask=loss_mask,
            session_id=session_id, trace_index=trace_index,
        )
        if findings:
            # Episode badness (gate failures, discipline breaches) masks:
            # auditable, group-shape-preserving, never trainable.
            status = Sample.Status.FAILED
            loss_mask = [0] * len(response_ids)

    stats = masked_behaviour_stats(logprobs, loss_mask)
    metadata = _sample_metadata(
        body=body, trajectory=trajectory, trace=trace,
        trace_index=trace_index, findings=findings, stats=stats,
    )
    prompt_messages = deepcopy(list(trace.get("prompt_messages") or []))
    response_messages = deepcopy(list(trace.get("response_messages") or []))

    return Sample(
        group_index=group_index,
        index=index,
        prompt=prompt_messages if prompt_messages else "",
        tokens=prompt_ids + response_ids,
        response=_messages_to_text(response_messages),
        response_length=len(response_ids),
        group_id=index,
        reward={reward_key: _reward_value(trace)},
        loss_mask=loss_mask,
        rollout_log_probs=logprobs,
        status=status,
        session_id=body.get("session_id"),
        metadata=metadata,
    )


def _build_placeholder(
    *, Sample: Any, body: Mapping[str, Any], trajectory: Mapping[str, Any],
    group_index: int, index: int, reward_key: str, findings: list[str],
) -> Any:
    """Fully-masked `remove_sample=True` stand-in for a session with no
    usable trace — no policy, TIS, or KL contribution; the group still
    trains. Inherits the vendored adapter's hardcoded `tokens=[0, 0]`
    (assumes token id 0 is batcher-safe — FINDINGS F-05)."""
    metadata = _sample_metadata(
        body=body, trajectory=trajectory, trace=None, trace_index=-1,
        findings=findings, stats=masked_behaviour_stats([0.0], [0]),
    )
    metadata["polar"]["placeholder"] = True
    return Sample(
        group_index=group_index,
        index=index,
        prompt="",
        tokens=[0, 0],
        response="",
        response_length=1,
        group_id=index,
        reward={reward_key: 0.0},
        loss_mask=[0],
        rollout_log_probs=[0.0],
        status=Sample.Status.ABORTED,
        remove_sample=True,
        session_id=body.get("session_id"),
        metadata=metadata,
    )


def _session_status(
    Sample: Any, body: Mapping[str, Any], trajectory: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> Any:
    """The vendored `_sample_status`, on the callback shape: session-level
    status paints every trace; `length` marks truncation, still trainable."""
    statuses = (trajectory.get("status"), body.get("status"))
    if "TIMEOUT" in statuses:
        return Sample.Status.ABORTED
    if "ERROR" in statuses or body.get("error") or trajectory.get("error"):
        return Sample.Status.FAILED
    if trace.get("finish_reason") == "length":
        return Sample.Status.TRUNCATED
    return Sample.Status.COMPLETED


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
    """Logprobs for a zero-masked sample: kept when aligned (audit value),
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
    reward attach step is a named CP-17 input (FINDINGS F-02)."""
    reward = trace.get("reward")
    return float(reward) if isinstance(reward, (int, float)) and not isinstance(reward, bool) else 0.0


def _sample_metadata(
    *, body: Mapping[str, Any], trajectory: Mapping[str, Any],
    trace: Mapping[str, Any] | None, trace_index: int,
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
    }
    if trace is not None:
        polar["trace_metadata"] = deepcopy(dict(trace.get("metadata") or {}))
        polar["trace_debug"] = {
            "finish_reason": trace.get("finish_reason"),
            "response_messages": deepcopy(list(trace.get("response_messages") or [])),
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
    trace: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """`group_id` / `policy_version` / `rollout_step`, wherever stamped.
    Nothing declares `policy_version` today (P3 inert) — CP-17's problem,
    named in the run book."""
    keys = ("group_id", "policy_version", "rollout_step")
    merged: dict[str, Any] = {}
    for source in (
        body.get("metadata"), trajectory.get("metadata"),
        trace.get("metadata") if trace is not None else None,
    ):
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
    estate — say so, next to the findings, instead of leaving a wall of
    `*_not_approved` to decode."""
    if findings and _using_packaged_pins() and any(
        "not_approved" in finding for finding in findings
    ):
        logger.error(
            "hash gates failed against the packaged pins (%s) — those are "
            "the shipping estate's approved sets; set GSJ_PINS_PATH to this "
            "estate's pins before the first import of gsj_rollout.checks. "
            "findings: %s", checks.PINS_PATH, findings,
        )


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    """`[role] content` rendering of the response messages — the vendored
    `_messages.messages_to_text`, re-stated here because that module lives
    in Polar's repo, not on any published surface. Known limitation kept:
    drops `tool_calls` structure; training reads tokens + logprobs, this
    is only the human-readable `Sample.response`."""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "assistant"))
        content = _flatten_content(message.get("content"))
        if content:
            parts.append(f"[{role}] {content}")
    return "\n\n".join(parts)


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(item.get("text", "")) for item in content
            if isinstance(item, dict) and ("text" in item or item.get("type") == "text")
        ]
        return "".join(parts).strip()
    if content is None:
        return ""
    return str(content)
