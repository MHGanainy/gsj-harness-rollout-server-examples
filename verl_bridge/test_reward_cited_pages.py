"""F-44 (library CP-51): the grader reads the deliverable the TASK declared.

Before CP-51 `grade_session` graded `sorted(glob("*.md"))[0]` and silently
ignored every other `.md` in the artifact dir — a lottery on multi-deliverable
episodes (CP-28 measured 8/15 thinking-on). Now: the task text's own
`out/<name>.md` wins; with no declaration, one file is graded and several
are refused by name. Fixture-driven, no GPU, no estate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import reward_cited_pages as grader

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _body(declared: str | None) -> dict:
    """A callback-shaped body whose task text declares (or not) its deliverable."""
    text = "# Skill: summarize\n\n5. Write the summary to `out/{}`.\n".format(declared) \
        if declared else "Summarize the case; cite pages as page:N."
    return {
        "session_id": "sk-test",
        "task_id": "t-1",
        "trajectory": {"traces": [{
            "prompt_messages": [{"role": "system", "content": "sys"},
                                {"role": "user", "content": [{"type": "text", "text": text}]}],
            "reward": None,
        }]},
    }


def _out(tmp_path: Path, files: dict[str, str]) -> Path:
    out = tmp_path / "sk-test" / "out"
    out.mkdir(parents=True)
    for name, body in files.items():
        (out / name).write_text(body, encoding="utf-8")
    return tmp_path


def test_the_declared_deliverable_wins_over_the_glob_order(tmp_path):
    root = _out(tmp_path, {"aaa-notes.md": "page:1 page:2 page:3",      # sorts first
                           "ep-deadbeef.md": "page:1 page:9"})           # declared
    grade = grader.grade_session(_body("ep-deadbeef.md"), root, cutoff=3, page_count=10)
    assert grade["artifact"].endswith("ep-deadbeef.md")
    assert (grade["n_cited"], grade["n_valid"]) == (2, 1)
    assert grade["reward"] == pytest.approx(0.5)


def test_several_deliverables_and_no_declaration_is_refused_by_name(tmp_path):
    root = _out(tmp_path, {"a.md": "page:1", "b.md": "page:2"})
    with pytest.raises(grader.AmbiguousArtifact) as excinfo:
        grader.grade_session(_body(None), root, cutoff=3, page_count=10)
    assert "2 deliverables" in str(excinfo.value)
    assert "a.md, b.md" in str(excinfo.value)


def test_one_deliverable_and_no_declaration_still_grades(tmp_path):
    root = _out(tmp_path, {"only.md": "page:1 page:2"})
    grade = grader.grade_session(_body(None), root, cutoff=3, page_count=10)
    assert grade["artifact"].endswith("only.md")
    assert grade["reward"] == pytest.approx(1.0)


def test_declared_but_never_written_is_reward_zero_not_a_guess(tmp_path):
    root = _out(tmp_path, {"stray.md": "page:1"})
    grade = grader.grade_session(_body("ep-missing.md"), root, cutoff=3, page_count=10)
    assert grade["artifact"] is None and grade["reward"] == 0.0


def test_the_real_cp09prime_body_declares_its_deliverable(tmp_path):
    body = json.loads((FIXTURES / "cp09prime.callback_session_result.json").read_text())
    out = tmp_path / body["session_id"] / "out"
    assert grader.declared_artifact(body, out) == out / "ep-3ba9d4a1498f89fc.md"
