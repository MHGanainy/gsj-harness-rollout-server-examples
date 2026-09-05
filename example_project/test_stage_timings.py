"""CP-87 orchestration boundaries and interrupted records; no GPU numerics."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_loop


@pytest.fixture
def run_loop(tmp_path, monkeypatch):
    """Run the real collecting loop with controlled external-stage costs.

    Each external call reads the on-disk record before returning. That tests
    visibility while a call stalls, not only a successful final summary.
    """
    clock = SimpleNamespace(now=100.0)
    control = SimpleNamespace(interrupt=False, moved=True, zero_rewards=False,
                              noisy_floor=False)
    calls = []
    worker = SimpleNamespace(optimizer_steps=0)
    run_dir = tmp_path / "run"
    args = SimpleNamespace(
        run_dir=run_dir, gpu=None, config="config", bank="bank", row=0,
        steps=2, episodes=2, timeout=10, snapshot="snapshot",
        thinking=None, engine_url=None, pad_token_id=0, lr=1e-5,
        entropy_coeff=0.0, use_kl_loss=False, grpo_std_normalization=False,
        allow_zero_advantage=False, sync_cmd="sync {ckpt}", sync_timeout=10,
    )
    costs = {"collect": 2., "grade": 3., "batch": 4., "worker_init": 5.,
             "replay": 6., "advantages": 7., "optimizer": 8., "export": 9.,
             "sync": 10., "probe_before": 1., "probe_after": 1.}

    def record():
        return json.loads((run_dir / "summary.json").read_text())

    def enter(name):
        step = record()["steps"][-1]
        assert step["stages"][name]["status"] == "running"
        assert "duration_s" not in step["stages"][name]
        calls.append((step["step"], name))
        clock.now += costs[name]
        return step

    def collect(cfg, rows, out, episodes, timeout, **kwargs):
        step = enter("collect")
        if step["step"] == 2:
            previous = record()["steps"][0]
            assert previous["status"] == "complete"
            assert previous["sync"]["probe"]["nonzero"] == 2
            assert worker.optimizer_steps == 1
        out.mkdir(parents=True)
        for i in range(episodes):
            (out / f"body-{i}.json").write_text("{}")

    def grade(*args):
        enter("grade")
        return [SimpleNamespace(reward=0.),
                SimpleNamespace(reward=0. if control.zero_rewards else 1.)]

    class Batch:
        meta_info = {"oversize_dropped": [], "prompt_length": 100,
                     "response_length": 28005}

        def __len__(self):
            return 2

    def batch(records, **kwargs):
        enter("batch")
        return Batch()

    def init(*args, **kwargs):
        enter("worker_init")
        return worker

    def replay(data, actual_worker, **kwargs):
        enter("replay")
        assert actual_worker is worker
        return {"mean_abs": .014297, "max_abs": .5, "floor_mean": .008,
                "floor_per_position": .21, "positions_over_floor": 831,
                "positions": 90677, "verl_debug_metrics": {"omitted": 1}}

    def advantages(data, n, **kwargs):
        enter("advantages")
        assert not kwargs["norm_adv_by_std"]
        return {"advantages": {"n": n, "min": -.5, "max": .5,
                               "mean": 0., "nonzero": 2}}

    def optimizer(data, actual_worker, n):
        step = enter("optimizer")
        assert step["replay"]["mean_abs"] == .014297
        assert step["advantages"]["nonzero"] == 2
        assert actual_worker is worker and n == 2
        if control.interrupt:
            raise KeyboardInterrupt
        worker.optimizer_steps += 1
        return {"grad_norm": .4, "actor/pg_loss": -.3}

    def export(actual_worker, directory):
        enter("export")
        assert actual_worker is worker
        return str(Path(directory) / "huggingface")

    def sync(*args):
        enter("sync")
        return costs["sync"]

    def probe(stream, destination, *args):
        enter("probe_after" if "after" in destination else "probe_before")

    def compare(before, after):
        moved = control.noisy_floor if "floor" in after else control.moved
        return {"mean_abs": .1 if moved else 0., "max_abs": .2 if moved else 0.,
                "nonzero": 2 if moved else 0, "positions": 2}

    monkeypatch.setattr(train_loop.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(train_loop, "parse_args", lambda: args)
    monkeypatch.setattr(train_loop.sys.stdout, "reconfigure", lambda **_: None,
                        raising=False)
    train = train_loop.train
    monkeypatch.setattr(train, "_effective_thinking", lambda _: ("off", "test"))
    monkeypatch.setattr(train, "_pins_preflight", lambda *_: None)
    monkeypatch.setattr(train, "_heavy_imports", lambda: None)
    monkeypatch.setattr(train, "load_config", lambda _: SimpleNamespace(
        estate=SimpleNamespace(serving_base_url="engine", model="model")))
    monkeypatch.setattr(train, "bank_rows", lambda _: [{}])
    monkeypatch.setattr(train, "row_uid", lambda _: "group")
    monkeypatch.setattr(train, "collect", collect)
    monkeypatch.setattr(train, "grade_and_ingest", grade)
    monkeypatch.setattr(train, "bridge", SimpleNamespace(
        build_batch=batch, H200_REPLAY_FLOOR_MEAN=.008,
        H200_REPLAY_FLOOR_PER_POSITION=.21))
    monkeypatch.setattr(train, "loop", SimpleNamespace(
        make_worker=init, stamp_meta=lambda _: None,
        recompute_old_log_probs=replay, rewards_correction_advantages=advantages,
        train_one_step=optimizer, save_hf_export=export))
    monkeypatch.setattr(train_loop, "run_sync", sync)
    monkeypatch.setattr(train_loop.probe_sync, "probe", probe)
    monkeypatch.setattr(train_loop.probe_sync, "compare", compare)
    return SimpleNamespace(args=args, control=control, calls=calls, costs=costs,
                           worker=worker, record=record)


def test_two_steps_persist_separate_costs_and_reuse_optimizer(run_loop):
    train_loop.main()
    summary = run_loop.record()
    assert summary["timing_clock"] == "time.monotonic"
    assert run_loop.worker.optimizer_steps == 2
    assert [name for _, name in run_loop.calls].count("worker_init") == 1
    for step in summary["steps"]:
        assert step["status"] == "complete"
        assert step["collected"] == step["batch_rows"] == 2
        assert step["rewards"] == {"rows": 2, "nonzero": 1, "mean": .5}
        assert "verl_debug_metrics" not in step["replay"]
        for name in ("collect", "worker_init", "replay", "optimizer", "export", "sync"):
            stage = step["stages"][name]
            if name == "worker_init" and step["step"] == 2:
                assert stage == {"status": "skipped", "duration_s": 0.,
                                 "reason": "persistent worker reused"}
                continue
            assert stage["status"] == "complete"
            assert stage["duration_s"] == run_loop.costs[name]
            assert stage["ended_monotonic_s"] - stage["started_monotonic_s"] == stage["duration_s"]
        assert step["sync"]["downtime_s"] == 10.
        assert step["sync"]["probe"]["nonzero"] == 2
    assert summary["steps"][0]["probe_floor"]["mean_abs"] == 0.


def test_optimizer_interruption_keeps_replay_advantages_and_elapsed(run_loop):
    run_loop.control.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        train_loop.main()
    step, = run_loop.record()["steps"]
    assert step["status"] == "failed"
    assert step["replay"]["mean_abs"] == .014297
    assert step["advantages"]["nonzero"] == 2
    assert step["stages"]["optimizer"]["status"] == "failed"
    assert step["stages"]["optimizer"]["error_type"] == "KeyboardInterrupt"
    assert step["stages"]["optimizer"]["duration_s"] == 8.
    assert "train_metrics" not in step and "export" not in step["stages"]


def test_failed_sync_proof_retains_downtime_and_zero_delta(run_loop):
    run_loop.control.moved = False
    with pytest.raises(SystemExit, match="SYNC FAILED THE PROOF"):
        train_loop.main()
    step, = run_loop.record()["steps"]
    assert step["status"] == "failed"
    assert step["stages"]["sync"]["status"] == "complete"
    assert step["stages"]["probe_after"]["status"] == "failed"
    assert step["sync"]["downtime_s"] == 10.
    assert step["sync"]["probe"]["mean_abs"] == 0.


def test_zero_reward_refusal_is_recorded_before_worker(run_loop):
    run_loop.control.zero_rewards = True
    with pytest.raises(SystemExit, match="reward never fired"):
        train_loop.main()
    step, = run_loop.record()["steps"]
    assert step["status"] == "failed"
    assert step["stages"]["grade"]["status"] == "failed"
    assert step["rewards"]["nonzero"] == 0
    assert "worker_init" not in step["stages"]


def test_noisy_probe_retains_floor_and_never_syncs(run_loop):
    run_loop.control.noisy_floor = True
    with pytest.raises(SystemExit, match="probe is NOISY"):
        train_loop.main()
    step, = run_loop.record()["steps"]
    assert step["status"] == "failed"
    assert step["probe_floor"]["mean_abs"] == .1
    assert step["stages"]["probe_before"]["status"] == "failed"
    assert "sync" not in step["stages"]


def test_unsynced_final_step_marks_sync_unmeasured(run_loop):
    run_loop.args.steps = 1
    run_loop.args.sync_cmd = None
    train_loop.main()
    step, = run_loop.record()["steps"]
    assert step["status"] == "complete"
    assert step["stages"]["sync"] == {"status": "skipped", "reason": "no --sync-cmd"}
    assert "sync" not in step


def test_hard_exit_preserves_running_start_and_completed_stage(tmp_path):
    # A subprocess cannot execute a context manager's finally on os._exit.
    # This is the interruption shape CP-82's end-only summary lost entirely.
    code = """
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from train_loop import step_stage
run_dir = Path(sys.argv[2])
step = {"step": 1, "status": "running"}
summary = {"steps": [step]}
with step_stage(run_dir, summary, step, "replay"):
    step["replay"] = {"mean_abs": 0.014297}
with step_stage(run_dir, summary, step, "optimizer"):
    os._exit(23)
"""
    result = subprocess.run([sys.executable, "-S", "-c", code, str(HERE),
                             str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 23, result.stderr
    step, = json.loads((tmp_path / "summary.json").read_text())["steps"]
    assert step["replay"]["mean_abs"] == .014297
    assert step["stages"]["replay"]["status"] == "complete"
    assert step["stages"]["replay"]["duration_s"] >= 0.
    assert step["stages"]["optimizer"]["status"] == "running"
    assert step["stages"]["optimizer"]["started_monotonic_s"] > 0.
    assert "duration_s" not in step["stages"]["optimizer"]
    assert "ended_monotonic_s" not in step["stages"]["optimizer"]
