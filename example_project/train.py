#!/usr/bin/env python3
"""train.py — collect → convert → train → sync against a gsj rollout estate.

One GRPO step per invocation, driven by the two files beside this script:
`config.yaml` (the one YAML — endpoints and little else) and
`taskbank.parquet` (the CP-24 bank: one row per (case, timestep, prompt),
skill cards resolved at build). The verl machinery is behind imports
(`verl_bridge/loop.py`, documented); what stays HERE is everything a
consumer must see rather than inherit: uid grouping (F-10), reward attach
(F-02), the three bridge assertions at ingest, the replay floor, and the
entropy/KL caution. RUNBOOK.md walks this file end to end — including what
collection costs (expect many attempts per qualifying episode; that is the
measured shape, not a malfunction).

Stages (composable across hosts — collect on the estate, train on a GPU):
    python train.py --collect-only        # submit + save accepted bodies
    python train.py --dry-run             # grade+ingest+batch, CPU only
    python train.py --snapshot <hf dir>   # the full step, needs the GPU
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "verl_bridge"))     # bridge, loop

import bridge   # noqa: E402  — the CP-20 conversion, three assertions inside
import loop     # noqa: E402  — the verl fit-loop machinery (documented)

from gsj_rollout.client import RolloutClient, partition_session_results  # noqa: E402
from gsj_rollout.config import load_config, render_task_request  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=_HERE / "config.yaml")
    p.add_argument("--bank", default=_HERE / "taskbank.parquet")
    p.add_argument("--out", default=_HERE / "collected", type=Path,
                   help="accepted SessionResult bodies land/read here")
    p.add_argument("--episodes", type=int, default=8,
                   help="ATTEMPTS per bank row (num_samples) — attempts, not "
                        "collected episodes; see RUNBOOK §what to expect")
    p.add_argument("--timeout", type=float, default=900.0)
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
        for result in accepted:
            body_path = out / f"{result['session_id']}.json"
            body_path.write_text(json.dumps(
                {**result, "gsj_uid": row_uid(row)}))     # remember the group
        print(f"[train] {row_uid(row)}: {len(accepted)}/{episodes} attempts "
              f"qualified -> {out}")
        total_ok += len(accepted)
        total_rejected += len(rejected)
    # F-27: the aggregate the stranger hand-counted (71/72 at CP-26).
    print(f"[train] collect total: {total_ok}/{total_ok + total_rejected} "
          f"terminal attempts qualified across {len(tasks)} rows "
          f"({total_rejected} rejected/quarantined) -> {out}")


def grade_and_ingest(cfg, out: Path) -> list:
    """Reward attach (F-02) then conversion. Bodies arrive with
    `reward: null`; grading BEFORE ingest is what makes this GRPO rather
    than an expensive way to train on zeros."""
    grader = loop.load_reward_module(
        _HERE.parent / "slime_bridge" / "reward_cited_pages.py")
    page_counts = cfg.user.get("page_counts", {})         # ours, via `user:`
    records = []
    rewards = []
    bodies = sorted(out.glob("*.json"))
    assert bodies, f"{out} holds no collected bodies — run --collect-only first"
    for path in bodies:
        body = json.loads(path.read_text())
        uid = body.pop("gsj_uid")
        case_id = body["trajectory"]["traces"][0]["metadata"]["case_id"]
        cutoff = int(body["trajectory"]["traces"][0]["metadata"]["timestep"])
        grade = grader.grade_session(
            body, str(cfg.harness.artifacts_dir), cutoff=cutoff,
            page_count=int(page_counts.get(case_id, cutoff)))
        # The three assertions + trainer-side `checks` run INSIDE ingest:
        # mask-before-ratio, sentinel rejection, validate_session_result.
        records.extend(bridge.ingest_session_result(body, uid=uid))
        rewards.append(grade["reward"])
        print(f"[train] {path.name}: reward={grade['reward']:.3f} uid={uid}")

    # F-27: the distribution the run book promises, aggregated — sparse is
    # the measured shape (1/27–1/112), but ALL-zero trains on nothing:
    # see it here, before the GPU does.
    nonzero = sum(1 for r in rewards if r != 0.0)
    print(f"[train] reward distribution: {nonzero}/{len(rewards)} nonzero, "
          f"mean={sum(rewards) / len(rewards):.4f}, max={max(rewards):.3f}")

    # A collection pre-filtered to qualifying episodes must have zero
    # masked rows — a masked row's 0.0 reward would still enter its GRPO
    # group's statistics. Inspect before training, never train through it.
    n_masked = sum(1 for r in records if not r.trainable)
    assert n_masked == 0, f"{n_masked} masked rows in a qualifying collection"

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
    args = parse_args()
    if args.gpu is not None:
        # F-34: must land before the first CUDA call — under
        # CUDA_VISIBLE_DEVICES the worker's hardcoded cuda:0 maps to the
        # chosen physical device.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg = load_config(args.config)

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
