#!/usr/bin/env python3
"""CP-17 pre-flight inside the slime container — A-26's answer, first.

Run before burning a training job on a surface mistake:

  1. real slime imports, and `bridge.load_sample_type()` returns
     `slime.utils.types.Sample` (not the local double);
  2. the bridge constructs a REAL Sample from a REAL collected body —
     library A-26 verified or refuted, loudly, in one line;
  3. the three assertions are live on that path (checks findings, the
     masked aggregate, sentinel/mask discipline);
  4. torch sees the GPU;
  5. the vendored LOO post-processor path-loads.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import bridge
from reward_cited_pages import grade_session

print("== 1. slime import + Sample type ==")
Sample = bridge.load_sample_type()
print(f"   load_sample_type() -> {Sample.__module__}.{Sample.__name__}")
assert Sample.__module__.startswith("slime."), "real slime not importable"
import slime  # noqa: E402

print(f"   slime at {Path(slime.__file__).parent}")

print("== 2. real Sample from a real collected body (library A-26) ==")
collected = sorted(Path(os.environ["GSJ_COLLECTED_DIR"]).glob("*.json"))
body = json.loads(collected[0].read_text())
grade = grade_session(body, os.environ["GSJ_ARTIFACTS_ROOT"],
                      cutoff=int(os.environ["GSJ_CUTOFF"]),
                      page_count=int(os.environ["GSJ_PAGE_COUNT"]))
samples = bridge.ingest_session_result(
    body, group_index=0, trajectory_index=0, reward_key="score",
    max_tokens=int(os.environ["GSJ_MAX_TOKENS"]))
sample = samples[0]
assert type(sample).__module__.startswith("slime."), type(sample)
print(f"   A-26 HOLDS — constructed {type(sample).__module__}."
      f"{type(sample).__name__} from {body['session_id']}")
print(f"   tokens={len(sample.tokens)} response_length={sample.response_length} "
      f"mask1={sum(sample.loss_mask)} logprobs={len(sample.rollout_log_probs)} "
      f"status={sample.status} group_id={sample.group_id} "
      f"session_id={sample.session_id!r} reward={sample.reward}")

print("== 3. the three assertions, live on this path ==")
gsj = sample.metadata["gsj"]
print(f"   checks trainer-side: findings={gsj['findings']} "
      f"(pins={'packaged' if gsj['packaged_pins'] else 'checkout'})")
print(f"   masked aggregate: trainable_positions={gsj['trainable_positions']} "
      f"behaviour_logprob_mean={gsj['behaviour_logprob_mean']:.6f}")
doctored = json.loads(collected[0].read_text())
doctored["trajectory"]["traces"][0]["response_logprobs"][
    doctored["trajectory"]["traces"][0]["loss_mask"].index(1)] = -9999.0
try:
    bridge.ingest_session_result(doctored, group_index=0, trajectory_index=0,
                                 reward_key="score", max_tokens=32768)
    raise SystemExit("   FAIL: sentinel was accepted")
except bridge.SentinelLogprobError as exc:
    print(f"   sentinel rejection live: {str(exc)[:90]}...")

print("== 4. torch / GPU ==")
import torch  # noqa: E402

print(f"   torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")

print("== 5. vendored LOO post-processor ==")
from cp17_loop.polar_pp import post_process_rewards  # noqa: E402

print(f"   loaded {post_process_rewards.__module__}.{post_process_rewards.__name__} "
      f"from {os.environ['POLAR_REWARD_PP_FILE']}")
print(f"   grade of {grade['session_id']}: reward={grade['reward']} "
      f"(cited={grade['n_cited']} valid={grade['n_valid']})")
print("SMOKE OK")
sys.exit(0)
