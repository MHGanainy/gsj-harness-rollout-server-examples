#!/usr/bin/env python3
"""train.py — collect → convert → train → sync against a gsj rollout estate.

One GRPO step per invocation, driven by the two files beside this script:
`config.yaml` (the one YAML — endpoints and little else) and
`taskbank.parquet` (the CP-24 bank: one row per (case, timestep, prompt),
skill cards resolved at build). The verl machinery is behind imports
(`verl_bridge/loop.py`, documented); what stays HERE is everything a
consumer must see rather than inherit: uid grouping (F-10), reward attach
(F-02), the three bridge assertions at ingest, the replay floor, the
entropy/KL caution — and, since library CP-31, the thinking-mode/pins
agreement (G6 is per-mode pins data; this script refuses to spend the
estate on a mismatch it can see). RUNBOOK.md walks this file end to end.

Stages (composable across hosts — collect on the estate, train on a GPU):
    python train.py --collect-only        # submit + save accepted bodies
    python train.py --dry-run             # grade+ingest+batch, CPU only
    python train.py --snapshot <hf dir>   # the full step, needs the GPU

Pass the SAME --thinking (or none, for the config's value) to every stage
that touches one --out directory: the trainer-side gate re-runs at ingest
under whichever pins this process resolved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# The heavy imports (bridge, loop, gsj_rollout) happen in _heavy_imports(),
# AFTER the thinking/pins preflight — gsj_rollout.checks resolves
# GSJ_PINS_PATH exactly once, at first import (library CP-11b), so the
# mode must pick the pins before anything imports checks. Module-level
# imports here would fix the pins to whatever the environment happened to
# hold. (Bonus: --help works without the verl closure installed.)
bridge = loop = None                         # bound by _heavy_imports()
RolloutClient = partition_session_results = None
load_config = render_task_request = None

_PI_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")


def _packaged_on_pins():
    """The installed wheel's thinking-on pins (library CP-33, wishlist 28 —
    the interim ./pins/ copy retired with it), located WITHOUT importing
    gsj_rollout: resolution happens once at first import (CP-11b), so the
    path must be known before any import runs. None = not packaged (a
    pre-0.1.1 install)."""
    import importlib.util
    spec = importlib.util.find_spec("gsj_rollout")
    if spec is None or not spec.submodule_search_locations:
        return None
    path = (Path(list(spec.submodule_search_locations)[0])
            / "pins" / "thinking-on" / "pins.gsj.json")
    return path if path.exists() else None

# The two reference G6 tails (library ADR-0024; token ids under the served
# Qwen3 tokenizer). Used only to CLASSIFY the resolved pins in words —
# the gate itself is the library's, unchanged.
_ON_TAIL = [[151644, 77091, 198]]                       # bare generation prompt
_OFF_TAIL = [[151644, 77091, 198, 151667, 271, 151668, 271]]  # empty-think block
_THINK_OPEN, _THINK_CLOSE = 151667, 151668              # <think> / </think>


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=_HERE / "config.yaml")
    p.add_argument("--bank", default=_HERE / "taskbank.parquet")
    p.add_argument("--out", default=_HERE / "collected", type=Path,
                   help="accepted SessionResult bodies land/read here — use "
                        "a separate dir per thinking mode (e.g. "
                        "collected-on / collected-off) when comparing")
    p.add_argument("--episodes", type=int, default=8,
                   help="ATTEMPTS per bank row (num_samples) — attempts, not "
                        "collected episodes; see RUNBOOK §what to expect")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--thinking", default=None, metavar="LEVEL",
                   help="override config.yaml's harness.thinking for this run "
                        "(default: the config's value — this project ships "
                        "\"medium\", i.e. ON). pi levels: off|minimal|low|"
                        "medium|high|xhigh|max; every non-off level is "
                        "wire-equivalent (CP-28). COST: thinking-on runs "
                        "~2.7x wall clock and ~2x tokens vs off (CP-28, "
                        "2026-08-14) — and needs the thinking-on pins on both "
                        "legs (train.py handles its own; restart "
                        "`gsj-rollout serve` per RUNBOOK when you flip modes)")
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="stop before the GPU: grade, ingest, batch, assert")
    p.add_argument("--snapshot", help="pinned HF snapshot dir (the served "
                                      "model's weights) — required to train")
    p.add_argument("--ckpt-dir", default=_HERE / "ckpt", type=Path)
    p.add_argument("--gpu", default=None, metavar="N",
                   help="GPU index for the training step — sets "
                        "CUDA_VISIBLE_DEVICES=N so the worker's first visible "
                        "device IS your chosen one (F-34: default is whatever "
                        "the environment exposes first; on a shared box check "
                        "`nvidia-smi` and pick a free ~90 GB device)")
    return p.parse_args()


def _effective_thinking(args) -> tuple[str, str]:
    """The mode this run will submit with, and where it came from — decided
    BEFORE any gsj_rollout import so the pins can follow it. The library's
    validator is still the authority (main() re-validates through it);
    this pre-read only needs off-vs-non-off and mirrors the validator's
    YAML 1.1 mapping: pyyaml parses bare `on`/`off` as booleans."""
    if args.thinking is not None:
        return str(args.thinking), "--thinking"
    try:
        import yaml
        raw = yaml.safe_load(Path(args.config).read_text()) or {}
        value = (raw.get("harness") or {}).get("thinking", "off")
    except Exception:
        # Unreadable/invalid config: load_config will refuse it loudly in
        # main() before anything touches the estate. Assume off here.
        return "off", "config.yaml (unreadable — load_config decides)"
    if isinstance(value, bool):
        value = "off" if value is False else "on"
    return str(value), "config.yaml"


def _classify_pins(path: Path) -> str:
    """'on' | 'off' | 'foreign' | 'unreadable' — by the G6 tail ids."""
    try:
        tail = json.loads(path.read_text())["pins"]["g6_expected_tail_ids"]
    except Exception:
        return "unreadable"
    return "on" if tail == _ON_TAIL else "off" if tail == _OFF_TAIL else "foreign"


def _pins_preflight(level: str, source: str) -> None:
    """The CP-31 seam: make GSJ_PINS_PATH agree with the mode, or say in
    WORDS what would otherwise happen — before the estate is spent. Runs
    before the first gsj_rollout import (resolution is once-per-process,
    library CP-11b). G6 is per-mode pins data (ADR-0024): mode and pins
    are two statements of one fact, and a disagreement quarantines every
    episode G6-only — a first run that produces nothing."""
    thinking_on = level not in ("off", "False", "false")
    env = os.environ.get("GSJ_PINS_PATH")

    if env is None and thinking_on:
        on_pins = _packaged_on_pins()
        if on_pins is None:
            sys.exit(
                f"train.py: thinking is {level!r} (ON) but the installed "
                "gsj-harness-rollout-server packages no thinking-on pins — "
                "default resolution means thinking-OFF, so every episode "
                "would be quarantined G6-only. The wheel carries both mode "
                "files since 0.1.1 (library CP-33, wishlist 28): "
                "pip install -U gsj-harness-rollout-server and retry, or "
                "run the off control: --thinking off")
        os.environ["GSJ_PINS_PATH"] = str(on_pins)
        print(f"[train] pins: GSJ_PINS_PATH={on_pins} (the wheel's packaged "
              "thinking-on set, library CP-33; set by train.py for this "
              "process — the RECEIVER is a separate process and needs the "
              "same variable; RUNBOOK §Run)")
        return

    if env is None:
        return  # off + default resolution: the off reference. Correct.

    kind = _classify_pins(Path(env))
    if kind == "unreadable":
        sys.exit(f"train.py: GSJ_PINS_PATH={env} is not a readable pins "
                 "file — fix or unset it (unset = the wheel's packaged "
                 "pins: its thinking-on file when the mode is on, the "
                 "off reference otherwise)")
    if thinking_on and kind == "off":
        sys.exit(
            f"train.py: thinking is {level!r} (ON) but GSJ_PINS_PATH={env} "
            "carries the thinking-OFF G6 tail — every collected episode "
            "would be quarantined G6-only (G6:prompt_suffix_ne_tail_ids on "
            "each) and the estate's time spent for nothing. The mode and "
            "the pins are two statements of one fact (library ADR-0024). "
            "Cure: unset GSJ_PINS_PATH (train.py then selects the wheel's "
            "packaged thinking-on set itself) — and start "
            "`gsj-rollout serve` with the thinking-on pins too, both law-6 "
            "legs. For the off control instead: --thinking off")
    if not thinking_on and kind == "on":
        sys.exit(
            f"train.py: thinking is 'off' but GSJ_PINS_PATH={env} carries "
            "the thinking-ON G6 tail — every off-mode episode would be "
            "quarantined G6-only. Cure: unset GSJ_PINS_PATH for off mode "
            "(default resolution IS the off reference) and restart "
            "`gsj-rollout serve` without it — pins resolve once per "
            "process (CP-11b), so a restart is required, not optional")
    if kind == "foreign":
        print(f"[train] pins: GSJ_PINS_PATH={env} matches neither reference "
              "G6 tail — assuming a re-derived foreign-estate pins file and "
              "proceeding; if this run quarantines everything G6-only, the "
              "pins' mode and harness.thinking disagree (ADR-0024)")
    else:
        print(f"[train] pins: GSJ_PINS_PATH={env} (inherited, "
              f"thinking-{kind})")


def _heavy_imports() -> None:
    """Deferred to keep every gsj_rollout import behind the pins preflight
    (module docstring; CP-11b). Rebinds the module-level names the rest of
    the script uses."""
    global bridge, loop, RolloutClient, partition_session_results
    global load_config, render_task_request
    sys.path.insert(0, str(_HERE.parent / "verl_bridge"))     # bridge, loop
    import bridge as _bridge   # the CP-20 conversion, three assertions inside
    import loop as _loop       # the verl fit-loop machinery (documented)
    from gsj_rollout.client import (RolloutClient as _rc,
                                    partition_session_results as _psr)
    from gsj_rollout.config import (load_config as _lc,
                                    render_task_request as _rtr)
    bridge, loop = _bridge, _loop
    RolloutClient, partition_session_results = _rc, _psr
    load_config, render_task_request = _lc, _rtr


def bank_rows(path: Path) -> list[dict]:
    """The flat ADR-0022 rows, read with pyarrow directly — every column is
    the task triple, a `render_task_request` argument, or `sandbox_image`
    (the row's reproducibility pin, checked against the config in
    collect())."""
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    # Row 32, made visible: `split` is a carried label, not a lock — the
    # TRAINER owns not training on eval. This script enforces it here.
    train_rows = [r for r in rows if r["split"] == "train"]
    print(f"[train] bank: {len(rows)} rows, training on {len(train_rows)} "
          f"(split == 'train'; {len(rows) - len(train_rows)} eval rows untouched)")
    return train_rows


def row_uid(row: dict) -> str:
    """F-10: the GRPO group key. All episodes of ONE bank row share one uid,
    so verl's estimator centres within the group. The bridge's default
    (uid = session id) makes every group a singleton whose "advantage" is
    its raw uncentred reward — that default is refused below."""
    return f"{row['case_id']}:t{row['timestep']}:{row['prompt_id']}"


def collect(cfg, rows: list[dict], out: Path, episodes: int, timeout: float) -> None:
    """Submit every training row, wait, keep what survives `checks` — the
    same validator the receiver ran (law 6: this side re-verifies)."""
    client = RolloutClient(cfg.polar.rollout.base_url)
    tasks = []
    total_ok = total_rejected = 0
    g6_only_rejects = length_terminated = 0
    for row in rows:
        # The row's image pin (ADR-0022): the bank means what it meant only
        # under the image it was built for. A drifted config collects
        # something else while looking fine — refuse instead.
        assert row["sandbox_image"] == cfg.runtime.image, (
            f"bank row {row_uid(row)} pins image {row['sandbox_image']!r} "
            f"but the config runs {cfg.runtime.image!r}")
        request = render_task_request(
            cfg,
            task_id=row_uid(row).replace(":", "-"),
            instruction=row["prompt_text"] or row["skill_card_text"],
            case_id=row["case_id"], timestep=row["timestep"],
            episodes=episodes, timeout_seconds=timeout,
            prompt_source=row["prompt_source"],
            skill_card_text=row["skill_card_text"] or None,  # free rows: unset
            split=row["split"],
        )
        tasks.append((row, client.submit(request)))       # Polar runs them in parallel
    out.mkdir(parents=True, exist_ok=True)
    for row, task_id in tasks:
        results = client.wait(task_id, timeout_s=timeout + 120.0)
        accepted, rejected = partition_session_results(results)
        for result, findings in rejected:
            print(f"[train] rejected {result.get('session_id')}: {findings}")
            if findings and all(str(f).startswith("G6:") for f in findings):
                g6_only_rejects += 1
        for result in accepted:
            body_path = out / f"{result['session_id']}.json"
            body_path.write_text(json.dumps(
                {**result, "gsj_uid": row_uid(row)}))     # remember the group
            if any(t.get("finish_reason") == "length"
                   for t in (result.get("trajectory") or {}).get("traces") or []):
                length_terminated += 1
        print(f"[train] {row_uid(row)}: {len(accepted)}/{episodes} attempts "
              f"qualified -> {out}")
        total_ok += len(accepted)
        total_rejected += len(rejected)
    # F-27: the aggregate the stranger hand-counted (71/72 at CP-26).
    print(f"[train] collect total: {total_ok}/{total_ok + total_rejected} "
          f"terminal attempts qualified across {len(tasks)} rows "
          f"({total_rejected} rejected/quarantined client-side) -> {out}")
    # ADR-0025 (F-47): tail finish_reason=length qualifies BY DESIGN — 7/72
    # at CP-32 entered the batch silently. The count makes the wall visible.
    print(f"[train] length-terminated: {length_terminated}/{total_ok} accepted "
          "episodes ended finish_reason=length — qualified by design; the "
          "bridge labels them TRUNCATED at ingest (library ADR-0025)")
    # F-51 (CP-32): the counts above are THIS process's verdicts only. In
    # the receiver-leg pins mismatch (§Thinking) this line reads healthy
    # while the durable archive receives nothing.
    print("[train] the receiver holds the archive's own verdict — after a "
          "first collect confirm traces_dir gained files; healthy here + "
          "empty archive + G6-only quarantine = the receiver-leg pins "
          "mismatch (RUNBOOK §Thinking)")
    # CP-31: an all-G6 wipeout is the mode/pins disagreement in the one
    # place this process can see it — name it, don't let it read as a
    # broken gate (F-18's failure mode).
    if total_ok == 0 and g6_only_rejects and g6_only_rejects == total_rejected:
        print("[train] every attempt was rejected with G6-only findings: "
              "that is the thinking mode and the resolved pins DISAGREEING "
              "(ADR-0024), not a broken gate — this process's pins passed "
              "preflight, so check the RECEIVER leg too, and re-run with "
              "matching GSJ_PINS_PATH on both (RUNBOOK §Thinking)")


def _think_token_share(records: list) -> tuple[int, int]:
    """(think mask-1 tokens, total mask-1 tokens) across the collection —
    tags included, the CP-28 measurement's own convention. Deterministic
    and tokenizer-free: the <think>/</think> ids segment the mask-1 spans
    (CP-28 §4 — the builder's mask stays binary; a reasoning submask is
    always DERIVED, exactly like this)."""
    think = total = 0
    for r in records:
        inside = False
        for tok, mask in zip(r.response_ids, r.response_mask):
            if mask == 1:
                total += 1
            if tok == _THINK_OPEN:
                inside = True
            if mask == 1 and (inside or tok == _THINK_CLOSE):
                think += 1
            if tok == _THINK_CLOSE:
                inside = False
    return think, total


def grade_and_ingest(cfg, out: Path) -> list:
    """Reward attach (F-02) then conversion. Bodies arrive with
    `reward: null`; grading BEFORE ingest is what makes this GRPO rather
    than an expensive way to train on zeros."""
    grader = loop.load_reward_module(
        _HERE.parent / "slime_bridge" / "reward_cited_pages.py")
    page_counts = cfg.user.get("page_counts", {})         # ours, via `user:`
    records = []
    rewards = []
    n_length = 0
    bodies = sorted(out.glob("*.json"))
    assert bodies, f"{out} holds no collected bodies — run --collect-only first"
    for path in bodies:
        body = json.loads(path.read_text())
        if any(t.get("finish_reason") == "length"
               for t in (body.get("trajectory") or {}).get("traces") or []):
            n_length += 1
        uid = body.pop("gsj_uid")
        case_id = body["trajectory"]["traces"][0]["metadata"]["case_id"]
        cutoff = int(body["trajectory"]["traces"][0]["metadata"]["timestep"])
        grade = grader.grade_session(
            body, str(cfg.harness.artifacts_dir), cutoff=cutoff,
            page_count=int(page_counts.get(case_id, cutoff)))
        # The three assertions + trainer-side `checks` run INSIDE ingest:
        # mask-before-ratio, sentinel rejection, validate_session_result.
        # NOTE (CP-31): that re-validation runs under THIS process's pins —
        # grading a thinking-on collection under off pins (or vice versa)
        # fails here with G6-named findings; pass the same --thinking to
        # every stage that reads this --out directory.
        records.extend(bridge.ingest_session_result(body, uid=uid))
        rewards.append(grade["reward"])
        print(f"[train] {path.name}: reward={grade['reward']:.3f} uid={uid}")

    # F-27: the distribution the run book promises, aggregated — sparse is
    # the measured shape (1/27–1/112 measured OFF; thinking-on measured
    # richer, 3/15 — CP-28), but ALL-zero trains on nothing: see it here,
    # before the GPU does.
    nonzero = sum(1 for r in rewards if r != 0.0)
    print(f"[train] reward distribution: {nonzero}/{len(rewards)} nonzero, "
          f"mean={sum(rewards) / len(rewards):.4f}, max={max(rewards):.3f}")

    # ADR-0025 (F-47): the bridge labels tail-length TRUNCATED (trainable);
    # aggregate it at grade too, so the wall is visible where the batch is
    # assembled — dropping them here is the consumer's one-line policy.
    n_trunc = sum(1 for r in records if r.status == "TRUNCATED")
    print(f"[train] length-terminated: {n_length}/{len(bodies)} bodies ended "
          f"finish_reason=length; {n_trunc}/{len(records)} ingested rows "
          "carry status=TRUNCATED — trainable by design (library ADR-0025)")

    # CP-31, the trainability fact met where the reward is attached: in a
    # thinking-on collection the reward lands on ALL mask-1 tokens, and
    # most of them are reasoning (CP-28 measured median 67%).
    think, total = _think_token_share(records)
    if think:
        print(f"[train] thinking-on collection: {100 * think // max(total, 1)}% "
              f"of trainable (mask-1) tokens are <think> tokens "
              f"({think}/{total}; CP-28 median: 67%) — under RLVR/GRPO that "
              "share of the gradient mass rides reasoning. OPD: that is the "
              "point. SFT on own reasoning at 0.6B: don't — derive a think "
              "submask from ids 151667/151668 or use a teacher (CP-28 §4)")

    # A collection pre-filtered to qualifying episodes must have zero
    # masked rows — a masked row's 0.0 reward would still enter its GRPO
    # group's statistics. Inspect before training, never train through it.
    masked = [r for r in records if not r.trainable]
    if masked:
        # F-49 (row 30): the G6 re-check at ingest masks, it does not name —
        # say the reasons in words before the assert below turns them fatal.
        from collections import Counter
        reasons = Counter(("; ".join(r.findings) or r.masked_reason or r.status)
                          for r in masked)
        print("[train] masked rows by reason: " + "; ".join(
            f"{reason!r} x{count}" for reason, count in reasons.most_common()))
    n_masked = len(masked)
    assert n_masked == 0, (
        f"{n_masked}/{len(records)} masked rows in a qualifying collection"
        + (" — ALL rows masked is the mode/pins disagreement at ingest "
           "(ADR-0024) wearing its trainer-side face: this process graded "
           "a foreign-mode --out directory (the G6 re-check masks, it does "
           "not name — F-49, measured CP-32). Re-run with the --thinking "
           "that collected this directory (RUNBOOK §Thinking)"
           if n_masked == len(records) else ""))

    # F-10, enforced: a group needs >= 2 EPISODES (sessions — one session's
    # traces share a uid and cannot centre each other). Singleton groups
    # are dropped loudly rather than trained on raw uncentred rewards that
    # only look like GRPO advantages.
    sessions_by_uid: dict[str, set] = {}
    for r in records:
        sessions_by_uid.setdefault(r.uid, set()).add(r.session_id)
    keep = [r for r in records if len(sessions_by_uid[r.uid]) >= 2]
    for uid, sessions in sorted(sessions_by_uid.items()):
        if len(sessions) < 2:
            print(f"[train] DROPPED singleton group {uid} (F-10: one episode "
                  "cannot centre; collect more attempts for this row)")
    assert keep, "no uid group has >= 2 qualifying episodes — nothing to train on"
    return keep


def main() -> None:
    # F-46 (CP-32): python block-buffers redirected stdout, so a
    # backgrounded `… > log` showed nothing for an entire collect.
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if args.gpu is not None:
        # F-34: must land before the first CUDA call — under
        # CUDA_VISIBLE_DEVICES the worker's hardcoded cuda:0 maps to the
        # chosen physical device.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # CP-31, in this order and before any gsj_rollout import (CP-11b):
    # decide the mode, make the pins agree with it, then load the library.
    level, source = _effective_thinking(args)
    print(f"[train] thinking: {level} ({source})")
    _pins_preflight(level, source)
    if level not in ("off", "False", "false"):
        print("[train] thinking-on costs ~2.7x wall clock and ~2x tokens vs "
              "off (CP-28, 2026-08-14; median episode 7.7s -> 21.1s) — a "
              "slow collect is the mode working, not a malfunction; "
              "--thinking off collects the control")
    _heavy_imports()

    cfg = load_config(args.config)
    if args.thinking is not None:
        # The library's validator is the authority on levels (it names the
        # silent-clamp trap); re-validate rather than assign, because
        # assignment would bypass it.
        cfg.harness = type(cfg.harness).model_validate(
            {**cfg.harness.model_dump(), "thinking": args.thinking})

    if not args.dry_run and not (args.collect_only or args.snapshot):
        sys.exit("train.py: pass --snapshot <pinned HF dir> to train, or "
                 "--collect-only / --dry-run for the desk halves")

    rows = bank_rows(Path(args.bank))
    # --dry-run never touches the estate: it reads what a prior
    # --collect-only saved (grade_and_ingest says so if nothing is there).
    if args.collect_only or (not args.dry_run and not args.out.exists()):
        collect(cfg, rows, args.out, args.episodes, args.timeout)
        if args.collect_only:
            return

    records = grade_and_ingest(cfg, args.out)
    data = bridge.build_batch(records, pad_token_id=151643)   # Qwen3 pad
    n = len(data)
    assert not data.meta_info["oversize_dropped"]
    print(f"[train] batch: {n} rows, {len(set(r.uid for r in records))} GRPO groups")
    if args.dry_run:
        # Exercise the F-12 nested conversion's contract asserts on the
        # desk (CPU) — the one re-implemented piece should never first run
        # on the GPU box.
        td = loop.to_no_padding(data.to_tensordict())
        print(f"[train] dry run OK — nested conversion contract holds "
              f"({td['input_ids'].size(0)} rows); stopped before the GPU")
        return

    # ---- the step: every numeric op verl's, wired by loop.py -------------
    loop.stamp_meta(data)
    worker = loop.make_worker(args.snapshot)   # entropy/KL OFF: right for ONE
    # audited step; a multi-step run must arm them — CP-21 measured the
    # post-sync distribution visibly narrowing without them.
    replay = loop.recompute_old_log_probs(
        data, worker, floor_mean=bridge.H200_REPLAY_FLOOR_MEAN,
        floor_per_position=bridge.H200_REPLAY_FLOOR_PER_POSITION)
    print(f"[train] recompute vs captured: mean|Δ|={replay['mean_abs']:.6f} "
          f"(measured floor {replay['floor_mean']}), "
          f"{replay['positions_over_floor']}/{replay['positions']} over "
          f"{replay['floor_per_position']}")

    stats = loop.rewards_correction_advantages(data, n)
    adv = stats["advantages"]
    print(f"[train] GRPO advantages: min={adv['min']:.4f} mean={adv['mean']:.4f} "
          f"max={adv['max']:.4f} nonzero={adv['nonzero']}/{n}")

    metrics = loop.train_one_step(data, worker, n)
    grad_norm = metrics.get("grad_norm")
    assert grad_norm is not None and grad_norm > 0.0, f"grad_norm={grad_norm}"
    print(f"[train] one optimizer step: grad_norm={grad_norm:.4f} "
          f"pg_loss={metrics.get('actor/pg_loss')}")

    hf_dir = loop.save_hf_export(worker, str(args.ckpt_dir))
    (args.ckpt_dir / "summary.json").write_text(json.dumps(
        {"n": n, "replay": {k: v for k, v in replay.items()
                            if k != "verl_debug_metrics"},
         "advantages": adv, "train_metrics": metrics, "hf_export": hf_dir},
        indent=2, default=str))
    print(f"[train] HF export: {hf_dir}")
    print("[train] next — the sync (WORKSTATION-side, not the estate box: the\n"
          "        script drives the estate over ssh via GSJ_VLLM_SSH_HOST,\n"
          "        default 'h200-admin' — run it where that alias resolves,\n"
          "        F-29; ~1 min engine downtime):\n"
          f"        staging/serving/serve-updated.sh {hf_dir}\n"
          "        probe before/after: slime_bridge/cp17_loop/probe_sync.py\n"
          "        then collect again; drain in-flight episodes first (A-13)")


if __name__ == "__main__":
    main()
