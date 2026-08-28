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
# The task text declares its deliverable — the corpus skill cards end with
# "Write the … to `out/<task_id>.md`" and the rendered prompt fills the id in
# (`out/ep-3ba9d4a1498f89fc.md` on the CP-09′ body). This is the ONLY
# statement of which file is the deliverable; a directory listing is not one.
DECLARED = re.compile(r"`?out/([^\s`'\"]+\.md)`?")


class AmbiguousArtifact(ValueError):
    """Several deliverables and none declared — a refusal, not a lottery (F-44)."""


def declared_artifact(body: Mapping[str, Any], out_dir: Path) -> Path | None:
    """The deliverable the TASK declared; never the first of a glob (F-44).

    Reads the task's own text (the prompt_messages' user turns) for the
    `out/<name>.md` it asked for. A task that declared none is graded on its
    one `.md` if there is exactly one; on several, this raises — CP-28
    measured 8/15 thinking-on episodes multi-deliverable, and the CP-32 grader
    silently picked whichever sorted first.
    """
    for trace in (body.get("trajectory") or {}).get("traces") or []:
        for message in (trace.get("prompt_messages") or []) if isinstance(trace, dict) else []:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            text = content if isinstance(content, str) else " ".join(
                part.get("text", "") for part in content if isinstance(part, dict))
            names = DECLARED.findall(text)
            if names:
                return out_dir / names[-1]   # the card's last instruction wins
    candidates = sorted(out_dir.glob("*.md"))
    if len(candidates) > 1:
        raise AmbiguousArtifact(
            f"{out_dir}: {len(candidates)} deliverables and the task declared none "
            f"— refusing to pick one: {', '.join(p.name for p in candidates)}")
    return candidates[0] if candidates else None


def grade_session(body: Mapping[str, Any], artifacts_root: str | Path,
                  *, cutoff: int, page_count: int) -> dict[str, Any]:
    """Grade one callback-shaped SessionResult; attach reward to its traces."""
    session_id = str(body.get("session_id"))
    effective_cutoff = min(int(cutoff), int(page_count))
    artifact = declared_artifact(body, Path(artifacts_root) / session_id / "out")
    if artifact is not None and artifact.is_file():
        cited = [int(n) for n in
                 CITATION.findall(artifact.read_text(encoding="utf-8"))]
    else:
        cited = []          # declared but never written, or no deliverable: reward 0.0, graded
    n_valid = sum(1 for n in cited if 1 <= n <= effective_cutoff)
    reward = n_valid / max(len(cited), 1)

    trajectory = body.get("trajectory") or {}
    for trace in trajectory.get("traces") or []:
        if isinstance(trace, dict):
            trace["reward"] = float(reward)
    return {
        "session_id": session_id,
        "artifact": str(artifact) if artifact is not None and artifact.is_file() else None,
        "n_cited": len(cited),
        "n_valid": n_valid,
        "cutoff": effective_cutoff,
        "reward": float(reward),
    }
