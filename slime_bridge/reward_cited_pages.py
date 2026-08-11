"""The CP-17 reward attach — the predecessor's RLVR shape, trainer-owned.

`gsj-envloader-examples/rlvr/grade.py` is the prior art (R2): the episode's
deliverable cites pages as ``page:N``; the grade is verifiable from the
artifact bytes and the case census alone:

    reward = citations within cutoff / max(total citations, 1)

No artifact => reward 0.0, graded, never skipped — absence of work is a
verifiable outcome and the LOO baseline needs that mass in distribution
(the predecessor's F-16 warns the 0.6B earns it rarely; report the
distribution honestly either way).

The attach mutates the callback body IN MEMORY only: every trace of the
session gets the episode-level reward (the bridge reads
``trace["reward"]``; assignment is the evaluator's job — F-02 closes here,
outside both the server and the bridge, exactly where the scope law put it).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

CITATION = re.compile(r"page:(\d+)")


def grade_session(body: Mapping[str, Any], artifacts_root: str | Path,
                  *, cutoff: int, page_count: int) -> dict[str, Any]:
    """Grade one callback-shaped SessionResult; attach reward to its traces."""
    session_id = str(body.get("session_id"))
    effective_cutoff = min(int(cutoff), int(page_count))
    artifacts = sorted((Path(artifacts_root) / session_id / "out").glob("*.md"))
    if artifacts:
        cited = [int(n) for n in
                 CITATION.findall(artifacts[0].read_text(encoding="utf-8"))]
    else:
        cited = []
    n_valid = sum(1 for n in cited if 1 <= n <= effective_cutoff)
    reward = n_valid / max(len(cited), 1)

    trajectory = body.get("trajectory") or {}
    for trace in trajectory.get("traces") or []:
        if isinstance(trace, dict):
            trace["reward"] = float(reward)
    return {
        "session_id": session_id,
        "artifact": str(artifacts[0]) if artifacts else None,
        "n_cited": len(cited),
        "n_valid": n_valid,
        "cutoff": effective_cutoff,
        "reward": float(reward),
    }
