"""Fixture-driven verl-bridge tests — no GPU, no estate, no doubles for data
AND no double for the target class: unlike slime (FINDINGS F-03), verl @ the
pinned SHA imports off-estate, so every batch here is a real
`verl.protocol.DataProto` and every mechanism test calls verl's own code
(`compute_grpo_outcome_advantage`, `compute_rloo_outcome_advantage`,
`agg_loss`, `compute_position_id_with_mask`).

The fixtures are the real CP-09' (H200, the qualifying attempt-19 episode,
validates clean) and CP-07 (pi-corpus, pre-CP-13 shape, carries three
missing-evidence findings) callback bodies — byte-identical to the slime
bridge's. Every "fails" test doctors one copy, so an assertion that fires
for the wrong reason is visible.

The three assertions each have a test that fails when the assertion is
removed: neutralize `assert_sentinel_free` / `assert_mask_before_ratio`
(or their call sites), or delete the `validate_session_result` call, and
the named tests go red.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import bridge
from gsj_rollout import checks
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_grpo_outcome_advantage,
    compute_rloo_outcome_advantage,
)
from verl.utils.model import compute_position_id_with_mask

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
    return bridge.ingest_session_result(body, **overrides)


def _doctored(body, mutate):
    doctored = copy.deepcopy(body)
    mutate(doctored)
    return doctored


def _trace(body):
    return body["trajectory"]["traces"][0]


def _shortened(body, keep: int):
    """A consistent short variant: response arrays cut to `keep` positions.
    CP-09's mask opens with a [0, 270) run of 1s, so the head stays
    trainable and aligned."""
    return _doctored(body, lambda b: _trace(b).update(
        response_ids=_trace(b)["response_ids"][:keep],
        loss_mask=_trace(b)["loss_mask"][:keep],
        response_logprobs=_trace(b)["response_logprobs"][:keep],
    ))


# --- the conversion: a real trace → a well-formed DataProto ---------------


def test_cp09prime_converts_to_a_wellformed_batch(cp09prime_body):
    records = _ingest(cp09prime_body)
    assert len(records) == 1
    data = bridge.build_batch(records)
    trace = _trace(cp09prime_body)
    prompt_len, response_len = len(trace["prompt_ids"]), len(trace["response_ids"])

    assert isinstance(data, DataProto)
    assert len(data) == 1
    assert data.batch["prompts"].shape == (1, prompt_len)
    assert data.batch["responses"].shape == (1, response_len)
    assert data.batch["response_mask"].shape == (1, response_len)
    assert data.batch["rollout_log_probs"].shape == (1, response_len)
    assert data.batch["rm_scores"].shape == (1, response_len)
    assert data.batch["input_ids"].shape == (1, prompt_len + response_len)
    assert data.batch["attention_mask"].shape == (1, prompt_len + response_len)
    assert data.batch["position_ids"].shape == (1, prompt_len + response_len)

    # ids verbatim, never retokenized
    assert data.batch["prompts"][0].tolist() == trace["prompt_ids"]
    assert data.batch["responses"][0].tolist() == trace["response_ids"]
    assert data.batch["input_ids"][0].tolist() == trace["prompt_ids"] + trace["response_ids"]
    # the builder's mask verbatim, never inferred — 510 trainable positions
    assert data.batch["response_mask"][0].tolist() == trace["loss_mask"]
    assert int(data.batch["response_mask"].sum()) == 510
    # behaviour-policy values verbatim
    assert data.batch["rollout_log_probs"][0].tolist() == pytest.approx(
        trace["response_logprobs"])
    # dtypes verl's fit loop expects
    assert data.batch["response_mask"].dtype == torch.int64
    assert data.batch["rollout_log_probs"].dtype == torch.float32
    # both mask keys, equal — v0 losses read response_mask, the v1 engine
    # normalizer sums loss_mask; both in-tree writers keep them identical
    assert torch.equal(data.batch["loss_mask"], data.batch["response_mask"])
    # audit trail
    assert data.non_tensor_batch["uid"][0] == cp09prime_body["session_id"]
    assert data.non_tensor_batch["status"][0] == "COMPLETED"
    assert data.non_tensor_batch["masked_reason"][0] == ""
    assert data.non_tensor_batch["gsj"][0]["findings"] == []
    assert data.meta_info["oversize_dropped"] == []


def test_prompt_left_pad_response_right_pad_layout(cp09prime_body):
    """The agent-loop layout: prompt occupies the RIGHTMOST prompt columns,
    response the LEFTMOST response columns, and the prompt/response split
    is the fixed offset `prompt_length` for every row."""
    trace = _trace(cp09prime_body)
    prompt_len, response_len = len(trace["prompt_ids"]), len(trace["response_ids"])
    P, R = prompt_len + 35, response_len + 10  # force real padding both sides
    data = bridge.build_batch(_ingest(cp09prime_body), prompt_length=P,
                              response_length=R, pad_token_id=0)

    prompts = data.batch["prompts"][0].tolist()
    assert prompts[:35] == [0] * 35                      # left pad
    assert prompts[35:] == trace["prompt_ids"]           # ids flush right
    responses = data.batch["responses"][0].tolist()
    assert responses[:response_len] == trace["response_ids"]  # ids flush left
    assert responses[response_len:] == [0] * 10               # right pad

    attention = data.batch["attention_mask"][0]
    assert attention[:35].tolist() == [0] * 35
    assert attention[35:P + response_len].tolist() == [1] * (prompt_len + response_len)
    assert attention[P + response_len:].tolist() == [0] * 10

    # the split invariant verl's compute_response_mask relies on:
    # attention_mask[:, -R:] is exactly the response half
    assert torch.equal(
        data.batch["attention_mask"][:, -R:] * data.batch["response_mask"],
        data.batch["response_mask"],
    )
    # response tokens really do start at column P of input_ids
    assert data.batch["input_ids"][0, P:P + response_len].tolist() == trace["response_ids"]


def test_position_ids_follow_verls_own_formula(cp09prime_body):
    """position_ids must equal verl's compute_position_id_with_mask —
    left pads sit at 0, the first real token is position 0, and positions
    run contiguously across the prompt/response boundary."""
    data = bridge.build_batch(
        _ingest(cp09prime_body),
        prompt_length=len(_trace(cp09prime_body)["prompt_ids"]) + 7,
    )
    attention = data.batch["attention_mask"]
    assert torch.equal(data.batch["position_ids"],
                       compute_position_id_with_mask(attention))
    row = data.batch["position_ids"][0]
    assert row[:7].tolist() == [0] * 7          # left pads clipped to 0
    assert row[7].item() == 0                   # first real token: position 0
    real = int(attention[0].sum())
    assert row[7:7 + real].tolist() == list(range(real))  # contiguous through the split


def test_reward_null_becomes_zero_and_rm_scores_stay_zeros(cp09prime_body):
    # Every real body to date carries reward: null — the attach step is
    # the loop's (FINDINGS F-02). The bridge consumes, never invents.
    assert _trace(cp09prime_body)["reward"] is None
    data = bridge.build_batch(_ingest(cp09prime_body))
    assert torch.equal(data.batch["rm_scores"],
                       torch.zeros_like(data.batch["rm_scores"]))


def test_reward_is_placed_at_the_last_real_response_token(cp09prime_body):
    """verl's own placement (`_postprocess`: index = attention_mask[:, P:]
    .sum(-1) - 1) — NOT the last padded column."""
    doctored = _doctored(cp09prime_body,
                         lambda b: _trace(b).__setitem__("reward", 1.0))
    response_len = len(_trace(cp09prime_body)["response_ids"])
    data = bridge.build_batch(_ingest(doctored),
                              response_length=response_len + 100)
    rm_scores = data.batch["rm_scores"][0]
    assert rm_scores[response_len - 1].item() == 1.0
    assert float(rm_scores.sum()) == 1.0
    assert rm_scores[-1].item() == 0.0  # never at the padded tail


def test_multi_trace_session_shares_uid(cp09prime_body):
    doubled = _doctored(
        cp09prime_body,
        lambda b: b["trajectory"]["traces"].append(
            copy.deepcopy(b["trajectory"]["traces"][0])),
    )
    records = _ingest(doubled, uid="episode-7")
    assert len(records) == 2
    data = bridge.build_batch(records)
    # verl's estimators group by the uid non-tensor key: one session, one group.
    assert list(data.non_tensor_batch["uid"]) == ["episode-7", "episode-7"]


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
    response is masking, not dropping: zeroed response_mask, findings
    carried, row kept auditable in the batch."""
    records = _ingest(cp07_body)
    assert len(records) == 1
    record = records[0]
    assert set(record.findings) == {
        "G5:missing_evidence:workspace",
        "G1:missing_evidence:prompt_source",
        "G7:missing_evidence:settings",
    }
    assert not record.trainable
    assert record.masked_reason.startswith("findings:")
    data = bridge.build_batch(records)
    assert int(data.batch["response_mask"].sum()) == 0
    assert data.non_tensor_batch["gsj"][0]["trainable_positions"] == 0


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
    validation anywhere (and neither does verl's agent-loop ingest), so a
    loosened server policy must not loosen the trainer. With LP3 disarmed,
    checks returns clean and only the bridge's assertion stands between
    -9999.0 and verl's importance ratio."""
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
    in bypass mode verl copies this array into old_log_probs wholesale,
    so an unmasked non-zero would become a silent importance ratio."""
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

    (record,) = _ingest(cp09prime_body)
    stats = record.metadata["gsj"]
    assert stats["trainable_positions"] == 510
    assert stats["behaviour_logprob_sum"] == pytest.approx(sum(trainable))
    assert stats["behaviour_logprob_mean"] == pytest.approx(
        sum(trainable) / len(trainable))
    unmasked_mean = sum(trace["response_logprobs"]) / len(trace["response_logprobs"])
    assert stats["behaviour_logprob_mean"] != pytest.approx(unmasked_mean)


# --- absent/misaligned arrays, fail-closed statuses ----------------------


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


def test_misaligned_mask_is_rejected(cp09prime_body):
    doctored = _doctored(
        cp09prime_body,
        lambda b: _trace(b)["loss_mask"].pop(),
    )
    with pytest.raises(bridge.MaskDisciplineError, match="length"):
        _ingest(doctored)


def test_off_domain_mask_values_are_rejected(cp09prime_body):
    doctored = _doctored(
        cp09prime_body,
        lambda b: _trace(b)["loss_mask"].__setitem__(0, 2),
    )
    with pytest.raises(bridge.MaskDisciplineError, match="outside"):
        _ingest(doctored)


def test_unknown_status_fails_closed(cp09prime_body):
    """The vendored slime adapter fell through to COMPLETED on anything it
    did not recognize — an unknown status trained. This bridge masks it."""
    doctored = _doctored(
        cp09prime_body,
        lambda b: (b.__setitem__("status", "RUNNING"),
                   b["trajectory"].__setitem__("status", "RUNNING")),
    )
    (record,) = _ingest(doctored)
    assert not record.trainable
    assert record.masked_reason == "status:RUNNING/RUNNING"
    data = bridge.build_batch([record])
    assert int(data.batch["response_mask"].sum()) == 0


def test_timeout_session_is_masked_not_dropped(cp09prime_body):
    doctored = _doctored(
        cp09prime_body,
        lambda b: (b.__setitem__("status", "TIMEOUT"),
                   b["trajectory"].__setitem__("status", "TIMEOUT")),
    )
    (record,) = _ingest(doctored)
    assert record.status == "ABORTED"
    assert not record.trainable
    # ADM1 fired trainer-side too — the findings ride along for audit.
    assert any("ADM1" in finding for finding in record.findings)


# --- zero gradient by verl's mechanism, called and proven ----------------


def test_error_session_contributes_zero_gradient_by_verls_mechanism(
        cp09prime_body):
    """The mechanism, named and then EXECUTED: an all-zero response_mask
    row (a) receives advantage 0 from verl's estimator (the broadcast
    multiplies by response_mask) and (b) contributes exactly 0 to verl's
    agg_loss in every mode — verl's own neutralization path for rejected
    samples, not an invention of this bridge."""
    errored = _doctored(
        cp09prime_body,
        lambda b: (b.__setitem__("status", "ERROR"),
                   b["trajectory"].__setitem__("status", "ERROR"),
                   b.__setitem__("error", "sandbox exploded")),
    )
    good = _ingest(cp09prime_body, uid="g1")
    bad = _ingest(errored, uid="g2")
    assert not bad[0].trainable and bad[0].status == "FAILED"
    data = bridge.build_batch(
        good + bad, response_length=len(_trace(cp09prime_body)["response_ids"]))
    response_mask = data.batch["response_mask"]
    assert int(response_mask[1].sum()) == 0  # the ERROR row

    # (a) verl's advantage estimator zeroes the masked row
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=data.batch["rm_scores"],
        response_mask=response_mask.float(),
        index=data.non_tensor_batch["uid"],
    )
    assert torch.equal(advantages[1], torch.zeros_like(advantages[1]))

    # (b) verl's loss aggregation: the masked row adds nothing, any mode
    per_token_loss = torch.ones_like(response_mask, dtype=torch.float32)
    for mode in ("token-mean", "seq-mean-token-sum", "seq-mean-token-mean"):
        both = agg_loss(loss_mat=per_token_loss,
                        loss_mask=response_mask.float(), loss_agg_mode=mode)
        alone = agg_loss(loss_mat=per_token_loss[:1],
                         loss_mask=response_mask[:1].float(), loss_agg_mode=mode)
        assert torch.isfinite(both)
        assert both.item() == pytest.approx(alone.item())


# --- padding: where a batching bridge breaks -----------------------------


def test_unequal_length_batch_pads_without_corrupting_alignment(
        cp09prime_body, cp07_body):
    """Three real-shaped rows of very different lengths (3990 / 7196 / 100
    response tokens) in one batch: every row's mask and logprobs must sit
    at ITS OWN offsets, verbatim, with zeros exactly where padding is —
    this is the failure mode slime's per-sample shape never had."""
    short = _shortened(cp09prime_body, keep=100)
    records = (_ingest(cp09prime_body, uid="a")
               + _ingest(cp07_body, uid="b")
               + _ingest(short, uid="c"))
    assert [len(r.response_ids) for r in records] == [3990, 7196, 100]
    data = bridge.build_batch(records)
    R = data.meta_info["response_length"]
    P = data.meta_info["prompt_length"]
    assert R == 7196 and len(data) == 3

    for row, record in enumerate(records):
        n_response = len(record.response_ids)
        n_prompt = len(record.prompt_ids)
        # response block: ids, mask, logprobs verbatim at [0, n) — then pad
        assert data.batch["responses"][row, :n_response].tolist() == record.response_ids
        assert data.batch["response_mask"][row, :n_response].tolist() == record.response_mask
        assert data.batch["rollout_log_probs"][row, :n_response].tolist() == pytest.approx(
            record.rollout_log_probs)
        assert int(data.batch["response_mask"][row, n_response:].sum()) == 0
        assert float(data.batch["rollout_log_probs"][row, n_response:].abs().sum()) == 0.0
        # prompt block: left-padded, ids flush right
        assert data.batch["prompts"][row, P - n_prompt:].tolist() == record.prompt_ids
        assert int(data.batch["attention_mask"][row, :P - n_prompt].sum()) == 0
        # the fixed split: this row's response really starts at column P
        assert data.batch["input_ids"][row, P:P + n_response].tolist() == record.response_ids
        # attention is 1 exactly on real tokens
        assert int(data.batch["attention_mask"][row].sum()) == n_prompt + n_response

    # per-row trainable totals survived the stacking
    assert data.batch["response_mask"].sum(dim=1).tolist() == [
        sum(record.response_mask) for record in records]
    # and the CP-07 row is the masked one (its gates fail), not a shifted one
    assert data.non_tensor_batch["masked_reason"][1].startswith("findings:")
    assert int(data.batch["response_mask"][1].sum()) == 0


def test_oversize_trace_is_dropped_loudly_never_clipped(
        cp09prime_body, cp07_body):
    records = _ingest(cp09prime_body, uid="a") + _ingest(cp07_body, uid="b")
    data = bridge.build_batch(records, response_length=4000)  # cp07 is 7196
    assert len(data) == 1
    assert data.non_tensor_batch["session_id"][0] == cp09prime_body["session_id"]
    (dropped,) = data.meta_info["oversize_dropped"]
    assert dropped["session_id"] == cp07_body["session_id"]
    assert dropped["response_len"] == 7196
    with pytest.raises(ValueError, match="every record exceeded"):
        bridge.build_batch(records, response_length=50)


# --- F-08, asked of verl's own estimators --------------------------------


def test_f08_shape_verl_grpo_is_structurally_immune():
    """slime's vendored LOO post-processor divides by stdev(OTHERS)+1e-6;
    with 1 rewarded episode in 27 the others' stdev is exactly 0 and the
    rewarded advantage explodes to 1e6 (measured live at CP-17, F-08).
    verl's GRPO has the same epsilon-floored division TEXT — but the std
    is over the WHOLE group, self included, so std==0 implies every
    deviation is 0: the 1/epsilon factor only ever multiplies zero."""
    # the CP-17 distribution: 1 rewarded in 27, one uid group
    rewards = torch.zeros(27, 5)
    rewards[0, -1] = 1.0
    mask = torch.ones(27, 5)
    uid = np.array(["g"] * 27, dtype=object)
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards, response_mask=mask, index=uid)
    assert float(advantages.abs().max()) < 10.0        # slime's gave 1e6
    assert float(advantages.abs().max()) == pytest.approx(5.0, abs=0.2)

    # the degenerate case F-08 exploded on: identical rewards, std == 0
    flat, _ = compute_grpo_outcome_advantage(
        token_level_rewards=torch.zeros(27, 5), response_mask=mask, index=uid)
    assert torch.equal(flat, torch.zeros_like(flat))   # 0/epsilon, never 1/epsilon


def test_verl_rloo_has_no_variance_division():
    """verl's actual leave-one-out estimator centres by the leave-one-out
    mean and never divides by a variance — the F-08 mechanism (LOO stdev
    of identical remainders) has no counterpart to fire."""
    rewards = torch.zeros(27, 5)
    rewards[0, -1] = 1.0
    advantages, _ = compute_rloo_outcome_advantage(
        token_level_rewards=rewards, response_mask=torch.ones(27, 5),
        index=np.array(["g"] * 27, dtype=object))
    assert float(advantages.abs().max()) == pytest.approx(1.0, abs=0.05)


def test_singleton_uid_group_gets_raw_reward_advantage():
    """The hazard the bridge's default uid (= session_id) inherits: verl
    hardcodes mean 0 / std 1 for a group of ONE, so a singleton's advantage
    is its RAW UNCENTRED reward. Grouping is therefore a named CP-21 input:
    episodes of one prompt must share a uid or GRPO has no baseline."""
    rewards = torch.zeros(1, 5)
    rewards[0, -1] = 1.0
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards, response_mask=torch.ones(1, 5),
        index=np.array(["solo"], dtype=object))
    assert float(advantages[0, 0]) == pytest.approx(1.0, abs=1e-4)


def test_all_masked_batch_warns_it_trains_nothing(cp07_body, caplog):
    """An all-audit batch is legal but a training step on it divides by
    zero in verl's seq-mean agg modes (global_batch_size = seq_mask.sum()
    == 0) — the bridge says so instead of leaving it to the stack trace."""
    records = _ingest(cp07_body)  # gate failures → fully masked
    with caplog.at_level("WARNING", logger="bridge"):
        data = bridge.build_batch(records)
    assert int(data.batch["response_mask"].sum()) == 0
    assert any("fully masked" in message for message in caplog.messages)


# --- the exported floor ---------------------------------------------------


def test_replay_floor_is_the_measured_one_not_the_cp18_anchor():
    """The bridge does no replay (no serving engine here); it exports the
    CP-09'-measured floor so CP-21 inherits numbers, not the CP-18 anchor
    (mean 0.005 / per-position 0.05) that fails as written on CUDA
    continuous-batching capture."""
    assert bridge.H200_REPLAY_FLOOR_MEAN == 0.008
    assert bridge.H200_REPLAY_FLOOR_PER_POSITION == 0.21
    assert bridge.H200_REPLAY_FLOOR_MEAN > 0.005   # the anchor is too tight
    assert bridge.H200_REPLAY_FLOOR_PER_POSITION > 0.05
