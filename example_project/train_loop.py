#!/usr/bin/env python3
"""train_loop.py — collect → grade → batch → train → SYNC → collect → train.

The training loop CP-69 asked for: N steps end to end against a gsj
rollout estate, with the weight sync scripted — the piece that was three
manual steps at CP-17 and CP-21 — and PROVEN each time by the CP-17 probe
(teacher-forced logprobs on a fixed token stream, before and after). A
sync that reports success without measuring that the weights moved is the
failure mode this project exists to catch; this loop aborts on it.

Built on the audited stages of `train.py` (imported, not duplicated:
collect with the image-pin assert and F-27/F-51 counters, grade+ingest
with the three bridge assertions and the F-10 singleton drop) and
`verl_bridge/loop.py` (verl's own numerics). One worker persists across
steps, so step 2 continues step 1's optimizer state — a loop, not N
restarts.

The sync is RESTART-BASED, deliberately (CP-69 Step 2, FINDINGS F-79):
verl @ the pinned SHA cannot push weights into an engine it did not spawn
— every in-place path terminates in a Ray collective_rpc against a
verl-launched AsyncLLM carrying verl's worker extension class, reached
over node-local CUDA-IPC/ZMQ. Our engine is a standalone `vllm serve`
behind Polar's gateway, so in-place sync is not buildable here without
surrendering the serving to verl (F-79 records the middle paths). What IS
buildable is what CP-17 and CP-21 both did by hand: export HF weights,
stop the engine, restart it on the checkpoint under the SAME served name
(`estate/serving/serve-updated.sh` is the workstation-side recipe;
`sync_engine_local.sh` beside this script is the run-where-you-serve
form), wait for /v1/models. ~1 min downtime, measured here every sync.

STALENESS (A-13): this loop is SERIALIZED — `collect` returns only when
every episode is terminal, the sync starts only after that, and the next
step's collection starts only after /v1/models confirms the new
checkpoint. No collection can span a sync, so every batch is
single-policy BY CONSTRUCTION — stated here rather than holding by
accident. (An overlapped collect/train needs Polar's P3 first — CP-21's
list.)

Usage (two steps, the CP-69 proof shape):
    python train_loop.py --row 0 --steps 2 --episodes 28 \
        --snapshot <pinned HF dir> \
        --sync-cmd 'bash sync_engine_local.sh {ckpt}'

Use a fresh absent or empty --run-dir for each invocation; resume is
unsupported. The pins preflight is train.py's, reused (RUNBOOK §Thinking).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                     # train.py — audited stages
sys.path.insert(0, str(_HERE.parent / "verl_bridge"))
import train  # noqa: E402  (no heavy import at its module top — CP-31 order)
import probe_sync  # noqa: E402  (stdlib-only: the CP-17 instrument)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="The estate-specific values all arrive here — the bank row, "
               "the snapshot, the endpoint, the sync command, the step "
               "count. Nothing is baked in (CP-21's frozen artifact had "
               "the H200's topology hardcoded; this script must not).")
    p.add_argument("--config", default=_HERE / "config.yaml")
    p.add_argument("--bank", default=_HERE / "taskbank.parquet")
    p.add_argument("--row", type=int, default=0,
                   help="bank row index, 0-based over the train-split rows — "
                        "ONE row per run: its episodes form the GRPO group "
                        "(F-10), one group per step")
    p.add_argument("--steps", type=int, default=2,
                   help="training steps; each = collect + one optimizer step "
                        "+ sync")
    p.add_argument("--episodes", type=int, default=28,
                   help="collection ATTEMPTS per step (num_samples); the "
                        "measured reward density is sparse — 1/27 to 1/112 "
                        "off-mode (CP-17/CP-21) — so small N may collect "
                        "zero reward; the loop says so before training")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument("--snapshot", required=True,
                   help="pinned HF snapshot dir of the SERVED model — the "
                        "worker trains from it at step 1")
    p.add_argument("--run-dir", type=Path, default=None,
                   help="everything lands here (default: runs/loop-<row>): "
                        "step<k>/collected|ckpt|probes + summary.json")
    p.add_argument("--engine-url", default=None,
                   help="engine base URL for the sync wait and the probes "
                        "(default: the config's estate.serving_base_url — "
                        "the ENGINE, not the gateway)")
    p.add_argument("--sync-cmd", default=None,
                   help="shell command that restarts the engine on a new "
                        "checkpoint; standalone unquoted {ckpt} is replaced "
                        "with the shell-quoted HF export dir. The estate recipes: sync_engine_local.sh "
                        "(run on the serving host) or "
                        "estate/serving/serve-updated.sh (workstation-side, "
                        "F-29). REQUIRED when --steps > 1")
    p.add_argument("--sync-timeout", type=float, default=600.0,
                   help="seconds to wait for /v1/models after --sync-cmd")
    p.add_argument("--gpu", default=None, metavar="N",
                   help="training GPU — sets CUDA_VISIBLE_DEVICES=N before "
                        "the first CUDA call (F-34); discover free GPUs at "
                        "run time, never assume the last run's map")
    p.add_argument("--thinking", default=None, metavar="LEVEL",
                   help="override config.yaml's harness.thinking (train.py's "
                        "preflight runs; RUNBOOK §Thinking)")
    p.add_argument("--pad-token-id", type=int, default=None,
                   help="batch pad id (attention-masked, never reaches "
                        "compute); default: derived from the snapshot's "
                        "config/tokenizer — no hardcoded pad token")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--entropy-coeff", type=float, default=0.0,
                   help="unsupported: nonzero values are refused before collection")
    p.add_argument("--use-kl-loss", action="store_true",
                   help="unsupported: refused before collection; no reference-policy leg")
    p.add_argument("--grpo-std-normalization", action="store_true",
                   help="opt back into verl's std-normalized GRPO. Default "
                        "OFF — the F-08 guard: see the comment at the "
                        "advantage call")
    p.add_argument("--allow-zero-advantage", action="store_true",
                   help="train even when no episode earned reward (the step "
                        "is then weight-decay-only); default: refuse loudly")
    args = p.parse_args()
    for name in ("steps", "episodes", "timeout", "sync_timeout"):
        value = getattr(args, name)
        if not 0 < value < float("inf"):
            flag = name.replace("_", "-")
            p.error(f"found --{flag} {value}; expected a positive finite value; "
                    f"use --{flag} with a value greater than zero")
    if args.entropy_coeff != 0.0 or args.use_kl_loss:
        p.error(f"found entropy_coeff={args.entropy_coeff}, use_kl_loss={args.use_kl_loss}; "
                "expected entropy_coeff=0 and KL disabled: both controls are unsupported. "
                "Use --entropy-coeff 0 without --use-kl-loss; phase 5 must fund a safe "
                "entropy path and a reference policy before enabling them.")
    if args.sync_cmd:
        sync_command(args.sync_cmd, "checkpoint")  # validate before collection
    if args.steps > 1 and not args.sync_cmd:
        p.error("--steps > 1 needs --sync-cmd: without a sync, step 2 would "
                "collect from the PRE-step-1 policy while the loop claims "
                "otherwise — the exact silent staleness A-13 forbids. "
                "Recipes: 'bash sync_engine_local.sh {ckpt}' on the serving "
                "host, or 'estate/serving/serve-updated.sh {ckpt}' from the "
                "workstation (F-29: where GSJ_VLLM_SSH_HOST resolves).")
    return args


def resolve_pad_token_id(snapshot: Path, flag: int | None) -> tuple[int, str]:
    """No hardcoded pad token (CP-69): the id comes from the flag or the
    snapshot itself. Pads are attention-masked and never reach compute
    (bridge.build_batch's contract), so correctness needs only SOME id —
    but it must be stated, not assumed."""
    if flag is not None:
        return flag, "--pad-token-id"
    for name in ("config.json", "generation_config.json"):
        try:
            value = json.loads((snapshot / name).read_text()).get("pad_token_id")
        except (OSError, ValueError):
            continue
        if value is not None:
            return int(value), name
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(snapshot))
        if tok.pad_token_id is not None:
            return int(tok.pad_token_id), "tokenizer"
    except Exception as exc:  # noqa: BLE001 — every miss funnels to the exit
        print(f"[loop] tokenizer probe failed: {exc}")
    sys.exit("train_loop.py: cannot derive pad_token_id from the snapshot "
             f"({snapshot}) — pass --pad-token-id explicitly (the served "
             "Qwen3-0.6B's is 151643)")


def engine_models(engine_url: str, timeout_s: float = 3.0) -> list[str]:
    with urllib.request.urlopen(f"{engine_url}/v1/models",
                                timeout=timeout_s) as r:
        return [m["id"] for m in json.load(r).get("data", [])]


def wait_for_engine(engine_url: str, model: str, timeout_s: float) -> float:
    """Poll /v1/models until the served name is back; return the wait."""
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        try:
            if model in engine_models(engine_url):
                return time.monotonic() - start
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(5.0)
    sys.exit(f"train_loop.py: engine did not serve {model!r} at "
             f"{engine_url}/v1/models within {timeout_s:.0f}s of the sync "
             "command — the engine is DOWN or serving something else. "
             "Check the sync command's own output/log above, then restore "
             "by hand (estate.sh serve <family>, or serve-updated for the "
             "checkpoint) BEFORE rerunning; do not collect against an "
             "engine in an unknown state.")


def sync_command(template: str, hf_dir: str) -> str:
    """Each {ckpt} must be a standalone, unquoted shell word.

    Shell quoting is supplied here, exactly once. Other shell syntax is
    operator-owned. The placeholder is optional for checkpoint-free commands.
    """
    for match in re.finditer(re.escape("{ckpt}"), template):
        # shlex alone loses the distinction between quoted/unquoted words.
        prefix = template[:match.start()]
        lexer = shlex.shlex(prefix, posix=True)
        try:
            list(lexer)
        except ValueError:
            sys.exit("found quoted {ckpt}; expected a standalone unquoted placeholder; "
                     "use --sync-cmd 'bash sync_engine_local.sh {ckpt}'")
        if ((match.start() and not template[match.start()-1].isspace())
                or (match.end() < len(template) and not template[match.end()].isspace())):
            sys.exit("found embedded {ckpt}; expected a standalone unquoted placeholder; "
                     "use --sync-cmd 'bash sync_engine_local.sh {ckpt}'")
    return template.replace("{ckpt}", shlex.quote(hf_dir))


def run_sync(sync_cmd: str, hf_dir: str, engine_url: str, model: str,
             sync_timeout: float) -> float:
    """The restart sync, scripted (CP-69 Step 3): what CP-17 and CP-21 did
    by hand. Returns the measured downtime (sync start → /v1/models OK)."""
    cmd = sync_command(sync_cmd, hf_dir)
    print(f"[loop] sync: {cmd}")
    started = time.monotonic()
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        sys.exit(f"train_loop.py: the sync command exited "
                 f"{result.returncode} after {time.monotonic() - started:.3f}s from command start "
                 "— the engine's state is UNKNOWN "
                 "(possibly stopped with no replacement). Read its output "
                 "above; restore the engine by hand before rerunning. A "
                 "loop that shrugged here would collect from a stale or "
                 "dead engine — the failure mode this stage exists to stop.")
    try:
        wait_for_engine(engine_url, model, sync_timeout)
    except SystemExit:
        print(f"[loop] sync failed after {time.monotonic() - started:.3f}s from command start",
              file=sys.stderr)
        raise
    downtime = time.monotonic() - started
    print(f"[loop] sync: {model!r} back at {engine_url} — "
          f"{downtime:.0f}s from sync start to /v1/models")
    return downtime


def probe_paths(step_dir: Path) -> tuple[Path, Path]:
    return step_dir / "probe_before.json", step_dir / "probe_after.json"


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)    # F-46
    args = parse_args()
    run_dir = args.run_dir or (_HERE / "runs" / f"loop-row{args.row}")
    train.require_fresh_directory(run_dir)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)   # F-34

    # CP-31 order, train.py's own preflight reused: mode, then pins, then
    # the first gsj_rollout import.
    level, source = train._effective_thinking(args)
    print(f"[loop] thinking: {level} ({source})")
    train._pins_preflight(level, source)
    train._heavy_imports()
    bridge, loop = train.bridge, train.loop

    cfg = train.load_config(args.config)
    if args.thinking is not None:
        cfg.harness = type(cfg.harness).model_validate(
            {**cfg.harness.model_dump(), "thinking": args.thinking})

    rows = train.bank_rows(Path(args.bank))
    if not 0 <= args.row < len(rows):
        sys.exit(f"train_loop.py: --row {args.row} out of range — the bank "
                 f"holds train rows 0..{len(rows) - 1}")
    row = rows[args.row]
    run_dir.mkdir(parents=True, exist_ok=True)
    engine_url = args.engine_url or cfg.estate.serving_base_url
    model = cfg.estate.model
    snapshot = Path(args.snapshot)
    pad_id, pad_src = resolve_pad_token_id(snapshot, args.pad_token_id)
    print(f"[loop] row {args.row} ({train.row_uid(row)}), {args.steps} steps x "
          f"{args.episodes} attempts; engine {engine_url} serving {model}; "
          f"pad_token_id {pad_id} (from {pad_src}); run dir {run_dir}")

    if args.steps > 1 and args.entropy_coeff == 0.0 and not args.use_kl_loss:
        print("[loop] WARNING: multi-step run with entropy AND KL control "
              "off — CP-21 measured the post-sync distribution visibly "
              "narrowing in exactly this configuration and warned that one "
              "more step deepens the collapse. This run proceeds as "
              "configured (measured, not tuned). Both controls are unsupported; "
              "phase 5 must decide their memory and compute budget.")

    worker = None
    probe_stream: Path | None = None               # fixed for the whole run
    summary: dict = {"row": train.row_uid(row), "steps": []}

    for k in range(1, args.steps + 1):
        step_dir = run_dir / f"step{k}"
        collected = step_dir / "collected"
        print(f"\n[loop] ===== step {k}/{args.steps} =====")

        # -- collect: train.py's stage (image-pin assert, gsj_uid stamp,
        # F-27/F-51 counters). Serialized: returns only when every episode
        # is terminal, so nothing is in flight at the sync boundary.
        train.collect(cfg, [row], collected, args.episodes, args.timeout,
                      task_suffix=f"-step{k}")
        bodies = sorted(collected.glob("*.json"))
        if len(bodies) < 2:
            sys.exit(f"train_loop.py: step {k} collected {len(bodies)} "
                     "episodes — a GRPO group needs >= 2 (F-10). Extend "
                     "--episodes, check the receiver/quarantine, or pick a "
                     "healthier --row.")
        if probe_stream is None:
            probe_stream = bodies[0]               # the run's fixed stream
            print(f"[loop] probe stream fixed for the run: "
                  f"{probe_stream.name} (prompt+response ids, CP-17's "
                  "instrument)")

        # -- grade + ingest: train.py's stage (reward attach before ingest,
        # three assertions inside, masked-row naming, singleton drop).
        records = train.grade_and_ingest(cfg, collected)
        if (not args.allow_zero_advantage
                and all(r.reward == 0.0 for r in records)):
            # Refuse HERE, before the GPU spends minutes on worker init +
            # recompute for a step that is provably decay-only: all-zero
            # rewards give all-zero GRPO advantages. The advantage-level
            # check below still stands — it also catches the all-EQUAL
            # nonzero case (centering zeroes those too, CP-17's post-sync
            # 8/8 shape) that this cheap check cannot see.
            sys.exit(f"train_loop.py: step {k}'s reward never fired (0/"
                     f"{len(bodies)} sessions rewarded) — the step would "
                     "be weight decay only. The measured density is "
                     "1/27–1/112 (CP-17/CP-21): extend --episodes, pick "
                     "another --row, or pass --allow-zero-advantage to "
                     "take the decay-only step anyway (measured, not "
                     "hidden).")

        data = bridge.build_batch(records, pad_token_id=pad_id)
        n = len(data)
        assert not data.meta_info["oversize_dropped"]
        print(f"[loop] batch: {n} rows, "
              f"P={data.meta_info['prompt_length']} "
              f"R={data.meta_info['response_length']}")

        if worker is None:
            worker = loop.make_worker(str(snapshot), lr=args.lr,
                                      entropy_coeff=args.entropy_coeff,
                                      use_kl_loss=args.use_kl_loss)
            print(f"[loop] worker up from {snapshot} (lr={args.lr}, "
                  f"entropy_coeff={args.entropy_coeff}, "
                  f"use_kl_loss={args.use_kl_loss}); it PERSISTS across "
                  "steps — optimizer state continues, only the engine is "
                  "restarted at each sync")

        loop.stamp_meta(data)
        replay = loop.recompute_old_log_probs(
            data, worker, floor_mean=bridge.H200_REPLAY_FLOOR_MEAN,
            floor_per_position=bridge.H200_REPLAY_FLOOR_PER_POSITION)
        print(f"[loop] recompute vs captured: mean|Δ|={replay['mean_abs']:.6f}"
              f" (floor {replay['floor_mean']}), "
              f"{replay['positions_over_floor']}/{replay['positions']} over "
              f"{replay['floor_per_position']}")
        if k > 1:
            # Post-sync cross-check: step k's episodes were captured by the
            # engine on step k-1's checkpoint, and this worker HOLDS those
            # weights — agreement at the floor re-verifies the previous
            # sync end to end (a stale engine would diverge loudly here).
            if replay["mean_abs"] <= 3 * replay["floor_mean"]:
                verdict = ("at the floor — the previous sync served the "
                           "trained weights")
            else:
                verdict = ("ABOVE the floor — inspect before trusting the "
                           "previous sync")
            print(f"[loop] sync cross-check: recompute vs step {k}'s "
                  f"captured logprobs is {verdict}")

        # -- advantages. norm_adv_by_std defaults OFF — the F-08 guard,
        # transplanted: slime's vendored LOO post-processor exploded sparse
        # binary reward to 1e6-scale advantages (F-08, measured grad_norm
        # 450053.6 at CP-17; guarded there by --disable-grpo-std-
        # normalization → unclipped 0.4513). verl's whole-group std cannot
        # blow up that way (F-09, test-pinned) but on sparse reward it
        # still amplified the lone rewarded episode to +10.39 and a
        # clipped 2.32 gradient at CP-21. Dr.GRPO (centre, don't divide)
        # is the shape both measured guards converge on; a loop that ran
        # normalization silently on sparse reward is the shape F-08 warns
        # about, so opting back in is explicit: --grpo-std-normalization.
        if args.grpo_std_normalization:
            print("[loop] WARNING: --grpo-std-normalization — std-normalized "
                  "GRPO on sparse reward amplifies the lone rewarded episode "
                  "(CP-21: +10.39 advantage, clipped step); proceeding as "
                  "told.")
        stats = loop.rewards_correction_advantages(
            data, n, norm_adv_by_std=args.grpo_std_normalization)
        adv = stats["advantages"]
        print(f"[loop] advantages (GRPO, norm_by_std="
              f"{args.grpo_std_normalization}): min={adv['min']:.4f} "
              f"mean={adv['mean']:.4f} max={adv['max']:.4f} "
              f"nonzero={adv['nonzero']}/{n}")
        if adv["nonzero"] == 0 and not args.allow_zero_advantage:
            sys.exit(f"train_loop.py: step {k}'s reward never fired — every "
                     "advantage is 0.0, so the policy gradient is zero and "
                     "the 'step' would be weight decay only. The measured "
                     "density is 1/27–1/112 (CP-17/CP-21): extend "
                     "--episodes, pick another --row, or pass "
                     "--allow-zero-advantage to take the decay-only step "
                     "anyway (measured, not hidden).")

        metrics = loop.train_one_step(data, worker, n)
        grad_norm = metrics.get("grad_norm")
        if not args.allow_zero_advantage:
            assert grad_norm is not None and grad_norm > 0.0, \
                f"grad_norm={grad_norm}"
        print(f"[loop] optimizer step {k}: "
              f"pg_loss={metrics.get('actor/pg_loss')} "
              f"grad_norm={grad_norm} "
              f"clipfrac={metrics.get('actor/pg_clipfrac')}")

        hf_dir = loop.save_hf_export(worker, str(step_dir / "ckpt"))
        print(f"[loop] HF export: {hf_dir}")

        step_record = {
            "step": k, "collected": len(bodies), "attempts": args.episodes,
            "batch_rows": n,
            "replay": {key: value for key, value in replay.items()
                       if key != "verl_debug_metrics"},
            "advantages": adv, "train_metrics": metrics, "hf_export": hf_dir,
        }

        # -- the sync (skipped only on a final step with no --sync-cmd;
        # syncing after the last step keeps the ENGINE at the final policy).
        if args.sync_cmd:
            before_path, after_path = probe_paths(step_dir)
            probe_sync.probe(str(probe_stream), str(before_path),
                             engine_url, model)
            if k == 1:
                # The noise floor, once per run: two probes on IDENTICAL
                # weights must differ at exactly 0.0 (CP-09'/CP-17: this
                # engine's replay is bit-deterministic) — otherwise deltas
                # cannot be attributed to the sync and the proof is void.
                floor_path = step_dir / "probe_floor.json"
                probe_sync.probe(str(probe_stream), str(floor_path),
                                 engine_url, model)
                floor = probe_sync.compare(str(before_path), str(floor_path))
                if floor["mean_abs"] != 0.0:
                    sys.exit("train_loop.py: the probe is NOISY on this "
                             f"engine (mean|Δ|={floor['mean_abs']:.6f} on "
                             "identical weights, expected exactly 0.0) — "
                             "the sync proof cannot attribute deltas to "
                             "the weight change. Fix the engine's "
                             "determinism story before looping.")
                print("[loop] probe noise floor: mean|Δ|=0.000000 — "
                      "zero-noise instrument confirmed")
                step_record["probe_floor"] = floor
            print(f"[loop] sync boundary: step {k}'s collection is fully "
                  "terminal and no collection is in flight — the loop is "
                  "serialized, so no batch spans a sync (A-13, by "
                  "construction and now stated)")
            downtime = run_sync(args.sync_cmd, hf_dir, engine_url, model,
                                args.sync_timeout)
            probe_sync.probe(str(probe_stream), str(after_path),
                             engine_url, model)
            moved = probe_sync.compare(str(before_path), str(after_path))
            if moved["mean_abs"] == 0.0:
                sys.exit("train_loop.py: SYNC FAILED THE PROOF — the engine "
                         "answered /v1/models but ZERO probe positions "
                         "moved: it is serving the OLD weights (stop half "
                         "failed, wrong checkpoint path, or a cached "
                         "process). This is the failure mode the probe "
                         "exists to catch; do not trust this sync.")
            print(f"[loop] sync PROVEN: mean|Δ|={moved['mean_abs']:.6f} "
                  f"max|Δ|={moved['max_abs']:.6f} "
                  f"moved={moved['nonzero']}/{moved['positions']} "
                  f"(downtime {downtime:.0f}s)")
            step_record["sync"] = {"downtime_s": downtime, "probe": moved}
        else:
            print("[loop] no --sync-cmd: the engine still serves the "
                  "PRE-step weights — sync by hand before any further "
                  "collection (estate.sh serve-updated "
                  f"{hf_dir})")

        summary["steps"].append(step_record)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str))

    print(f"\n[loop] done: {args.steps} steps; summary -> "
          f"{run_dir / 'summary.json'}")
    rewards = [s["advantages"] for s in summary["steps"]]
    print("[loop] per-step advantages: " + "; ".join(
        f"step {i + 1}: nonzero {a['nonzero']}/{a['n']} max {a['max']:.3f}"
        for i, a in enumerate(rewards)))


if __name__ == "__main__":
    main()
