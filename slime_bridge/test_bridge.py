"""Fixture-driven bridge tests — no GPU, no estate, no doubles for data.

The fixtures are the real CP-09' (H200, the qualifying attempt-19 episode,
validates clean) and CP-07 (pi-corpus, pre-CP-13 shape, carries three
missing-evidence findings) callback bodies — exactly what the CP-17 loop
will see. Every "fails" test doctors one copy of one of them, so an
assertion that fires for the wrong reason is visible.

The three assertions each have a test that fails when the assertion is
removed: neutralize `assert_sentinel_free` / `assert_mask_before_ratio`
(or their call sites), or delete the `validate_session_result` call, and
the named test goes red.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import bridge
from gsj_rollout import checks

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def cp09prime_body() -> dict:
    return json.loads(
        (FIXTURES / "cp09prime.callback_session_result.json").read_text())


@pytest.fixture(scope="session")
def cp07_body() -> dict:
    return json.loads(
        (FIXTURES / "cp07.callback_session_result.json").read_text())


def _ingest(body, **overrides):
    kwargs = {"group_index": 0, "trajectory_index": 0}
    kwargs.update(overrides)
    return bridge.ingest_session_result(body, **kwargs)


def _doctored(body, mutate):
    doctored = copy.deepcopy(body)
    mutate(doctored)
    return doctored


def _trace(body):
    return body["trajectory"]["traces"][0]


# --- the conversion: a real trace → a well-formed Sample -----------------


def test_cp09prime_converts_to_a_wellformed_sample(cp09prime_body):
    samples = _ingest(cp09prime_body)
    assert len(samples) == 1
    sample = samples[0]
    trace = _trace(cp09prime_body)

    assert sample.tokens == trace["prompt_ids"] + trace["response_ids"]
    assert sample.response_length == len(trace["response_ids"]) == 3990
    assert len(sample.loss_mask) == sample.response_length
    assert len(sample.rollout_log_probs) == sample.response_length
    assert sample.loss_mask == trace["loss_mask"]  # verbatim, never inferred
    assert sum(sample.loss_mask) == 510
    assert sample.status is bridge.FallbackSample.Status.COMPLETED
    assert sample.group_id == sample.index == 0
    assert sample.session_id == cp09prime_body["session_id"]
    assert sample.remove_sample is False
    assert sample.prompt == trace["prompt_messages"]
    assert sample.metadata["gsj"]["findings"] == []
    assert sample.metadata["polar"]["session_status"] == "COMPLETED"


def test_reward_null_becomes_zero_under_the_reward_key(cp09prime_body):
    # Every real body to date carries reward: null — the attach step is
    # CP-17's (FINDINGS F-02). The bridge consumes, never invents.
    assert _trace(cp09prime_body)["reward"] is None
    (sample,) = _ingest(cp09prime_body, reward_key="score")
    assert sample.reward == {"score": 0.0}


def test_multi_trace_session_shares_group_id(cp09prime_body):
    doubled = _doctored(
        cp09prime_body,
        lambda b: b["trajectory"]["traces"].append(
            copy.deepcopy(b["trajectory"]["traces"][0])),
    )
    samples = _ingest(doubled, trajectory_index=7)
    assert len(samples) == 2
    # Slime 0.3.0's loss reducer counts the trajectory once: one group_id.
    assert {sample.group_id for sample in samples} == {7}


# --- assertion 3: `checks` runs trainer-side (CP-08 recording wrapper) ---


def test_checks_runs_trainer_side_on_what_arrived(cp09prime_body, monkeypatch):
    calls: list[str] = []
    real = checks.validate_session_result
    monkeypatch.setattr(
        checks, "validate_session_result",
        lambda body, policy=None: calls.append(body["session_id"]) or real(body, policy),
    )
    _ingest(cp09prime_body)
    # Law 6's teeth, trainer leg: called, not just available.
    assert calls == [cp09prime_body["session_id"]]


def test_unstamped_cp07_body_is_never_trainable(cp07_body):
    """The pre-CP-13 body really does fail the gates — and the bridge's
    response is masking, not dropping: FAILED, zero mask, findings carried."""
    samples = _ingest(cp07_body)
    assert len(samples) == 1
    sample = samples[0]
    findings = sample.metadata["gsj"]["findings"]
    assert set(findings) == {
        "G5:missing_evidence:workspace",
        "G1:missing_evidence:prompt_source",
        "G7:missing_evidence:settings",
    }
    assert sample.status is bridge.FallbackSample.Status.FAILED
    assert sample.loss_mask == [0] * sample.response_length
    assert sample.metadata["gsj"]["trainable_positions"] == 0


# --- assertion 2: sentinel rejection at ingest ---------------------------


def test_sentinel_at_stated_trainable_position_is_rejected(cp09prime_body):
    mask = _trace(cp09prime_body)["loss_mask"]
    position = mask.index(1)
    doctored = _doctored(
        cp09prime_body,
        lambda b: _trace(b)["response_logprobs"].__setitem__(position, -9999.0),
    )
    with pytest.raises(bridge.SentinelLogprobError, match=str(position)):
        _ingest(doctored)


def test_sentinel_is_rejected_even_when_the_policy_was_loosened(cp09prime_body):
    """The bridge's floor is its own — CP-02: upstream has no value-level
    validation anywhere, so a loosened server policy must not loosen the
    trainer. With LP3 disarmed (threshold sunk), checks returns clean and
    only the bridge's own assertion stands between -9999.0 and training."""
    mask = _trace(cp09prime_body)["loss_mask"]
    position = mask.index(1)
    doctored = _doctored(
        cp09prime_body,
        lambda b: _trace(b)["response_logprobs"].__setitem__(position, -9999.0),
    )
    loosened = checks.CheckPolicy(sentinel_threshold=-1e12)
    assert checks.validate_session_result(doctored, loosened) == []
    with pytest.raises(bridge.SentinelLogprobError):
        _ingest(doctored, policy=loosened)


# --- assertion 1: mask before ratio --------------------------------------


def test_nonzero_logprob_at_masked_position_is_rejected(cp09prime_body):
    """A real-looking value at mask==0 means mask and logprobs disagree —
    exactly what an unmasked importance ratio would consume silently."""
    mask = _trace(cp09prime_body)["loss_mask"]
    position = mask.index(0)
    doctored = _doctored(
        cp09prime_body,
        lambda b: _trace(b)["response_logprobs"].__setitem__(position, -0.25),
    )
    with pytest.raises(bridge.MaskDisciplineError, match=str(position)):
        _ingest(doctored)


def test_behaviour_aggregates_are_computed_under_the_mask(cp09prime_body):
    """The only aggregate the bridge emits is the masked one. Computed
    independently here from the fixture: if the mask stopped being applied
    (mean over 3990 instead of 510), both numbers move and this fails."""
    trace = _trace(cp09prime_body)
    trainable = [
        logprob for logprob, mask in
        zip(trace["response_logprobs"], trace["loss_mask"]) if mask == 1
    ]
    assert len(trainable) == 510  # not 3990 — the mask is applied

    (sample,) = _ingest(cp09prime_body)
    stats = sample.metadata["gsj"]
    assert stats["trainable_positions"] == 510
    assert stats["behaviour_logprob_sum"] == pytest.approx(sum(trainable))
    assert stats["behaviour_logprob_mean"] == pytest.approx(
        sum(trainable) / len(trainable))
    unmasked_mean = sum(trace["response_logprobs"]) / len(trace["response_logprobs"])
    assert stats["behaviour_logprob_mean"] != pytest.approx(unmasked_mean)


# --- absent logprobs, zero gradient, placeholders ------------------------


def test_absent_logprobs_on_a_trainable_trace_are_rejected(cp09prime_body):
    doctored = _doctored(
        cp09prime_body,
        lambda b: _trace(b).__setitem__("response_logprobs", None),
    )
    with pytest.raises(bridge.LogprobAbsentError, match="absent"):
        _ingest(doctored)


def test_misaligned_logprobs_are_rejected(cp09prime_body):
    doctored = _doctored(
        cp09prime_body,
        lambda b: _trace(b)["response_logprobs"].pop(),
    )
    with pytest.raises(bridge.LogprobAbsentError, match="length"):
        _ingest(doctored)


def test_timeout_session_contributes_zero_gradient(cp09prime_body):
    """The mechanism, named: a zeroed loss_mask (slime's loss reducer sees
    no trainable position; the reward post-processor additionally excludes
    ABORTED/FAILED trajectories from every baseline). The sample stays in
    the batch so the group keeps its shape."""
    doctored = _doctored(
        cp09prime_body,
        lambda b: (b.__setitem__("status", "TIMEOUT"),
                   b["trajectory"].__setitem__("status", "TIMEOUT")),
    )
    (sample,) = _ingest(doctored)
    assert sample.status is bridge.FallbackSample.Status.ABORTED
    assert sample.loss_mask == [0] * sample.response_length
    assert sample.remove_sample is False
    assert sample.metadata["gsj"]["trainable_positions"] == 0
    # ADM1 fired trainer-side too — the findings ride along for audit.
    assert any("ADM1" in finding for finding in sample.metadata["gsj"]["findings"])


def test_all_traces_unusable_emits_a_removable_placeholder(cp09prime_body):
    doctored = _doctored(
        cp09prime_body,
        lambda b: _trace(b).update(prompt_ids=[], response_ids=[]),
    )
    (placeholder,) = _ingest(doctored)
    assert placeholder.remove_sample is True
    assert placeholder.status is bridge.FallbackSample.Status.ABORTED
    assert placeholder.loss_mask == [0]
    assert placeholder.rollout_log_probs == [0.0]
    assert placeholder.metadata["polar"]["placeholder"] is True


def test_replay_floor_is_the_measured_one_not_the_cp18_anchor():
    """The bridge does no replay (no serving engine here); what it does do
    is export the CP-09'-measured floor so CP-17 inherits numbers, not the
    CP-18 anchor (mean 0.005 / per-position 0.05) that fails as written on
    CUDA continuous-batching capture."""
    assert bridge.H200_REPLAY_FLOOR_MEAN == 0.008
    assert bridge.H200_REPLAY_FLOOR_PER_POSITION == 0.21
    assert bridge.H200_REPLAY_FLOOR_MEAN > 0.005  # the anchor is too tight
    assert bridge.H200_REPLAY_FLOOR_PER_POSITION > 0.05
