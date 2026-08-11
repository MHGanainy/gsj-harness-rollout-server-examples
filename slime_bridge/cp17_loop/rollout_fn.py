"""CP-17: the slime rollout function — collected bodies in, real Samples out.

Registered via slime's ``--rollout-function-path``; runs INSIDE the slime
training process (``--debug-train-only`` — no SGLang engines; generation is
the estate's vLLM, reached only through `gsj-rollout submit`, never from
here). This is where library A-26 meets reality: with real slime importable,
`bridge.load_sample_type()` returns `slime.utils.types.Sample` and every
conversion calls the real constructor — a surface mismatch TypeErrors
loudly on the first episode.

Imports are flat (`import bridge`), the repo's test convention: the
`slime_bridge/` directory rides PYTHONPATH, never installed as a package.

Environment (all required):
    GSJ_COLLECTED_DIR    directory of receiver-accepted SessionResult JSONs
    GSJ_ARTIFACTS_ROOT   pi_harness artifacts_dir (deliverables + transcripts)
    GSJ_CUTOFF           the submitted timestep (cutoff claim)
    GSJ_PAGE_COUNT       the case's page census (MCP /health)
    GSJ_MAX_TOKENS       per-GPU token bound; the bridge drops beyond it

Returns list[list[Sample]] — one inner list per trajectory, all samples of
a trajectory sharing ``group_id`` (slime's own flatten + group contract).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from bridge import ingest_session_result
from reward_cited_pages import grade_session

logger = logging.getLogger(__name__)


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    assert not evaluation, "CP-17 runs no eval pipeline"
    collected = sorted(Path(os.environ["GSJ_COLLECTED_DIR"]).glob("*.json"))
    assert collected, "GSJ_COLLECTED_DIR holds no SessionResult bodies"
    artifacts_root = os.environ["GSJ_ARTIFACTS_ROOT"]
    cutoff = int(os.environ["GSJ_CUTOFF"])
    page_count = int(os.environ["GSJ_PAGE_COUNT"])
    max_tokens = int(os.environ["GSJ_MAX_TOKENS"])

    groups, grades = [], []
    for trajectory_index, path in enumerate(collected):
        body = json.loads(path.read_text())
        grade = grade_session(body, artifacts_root,
                              cutoff=cutoff, page_count=page_count)
        grades.append(grade)
        samples = ingest_session_result(
            body,
            group_index=0,  # one prompt (the golden triple) => one LOO group
            trajectory_index=trajectory_index,
            reward_key=args.reward_key,
            max_tokens=max_tokens,
        )
        for sample in samples:
            # A-26, stated in code: these are REAL slime Samples or nothing.
            assert type(sample).__module__.startswith("slime."), (
                f"FallbackSample leaked into training: {type(sample)!r}")
        groups.append(samples)
        logger.info(
            "cp17 rollout %s: %s reward=%.3f (cited=%d valid=%d artifact=%s) "
            "samples=%d trainable_positions=%s",
            rollout_id, grade["session_id"], grade["reward"], grade["n_cited"],
            grade["n_valid"], bool(grade["artifact"]), len(samples),
            [s.metadata["gsj"]["trainable_positions"] for s in samples],
        )

    rewards = [g["reward"] for g in grades]
    logger.info(
        "cp17 rollout %s: %d sessions -> %d sample groups; reward "
        "distribution min=%.3f mean=%.3f max=%.3f nonzero=%d/%d",
        rollout_id, len(collected), len(groups), min(rewards),
        sum(rewards) / len(rewards), max(rewards),
        sum(1 for r in rewards if r > 0), len(rewards),
    )
    summary_path = Path(os.environ.get("GSJ_LOOP_SUMMARY",
                                       "/tmp/cp17_rollout_summary.json"))
    summary_path.write_text(json.dumps({"rollout_id": rollout_id,
                                        "grades": grades}, indent=2))
    return groups
