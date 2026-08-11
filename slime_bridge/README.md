# slime_bridge

Callback-shaped `SessionResult` → slime `Sample` (v0.3.0), the trainer's
side of `gsj-harness-rollout-server`'s M4. What it is and isn't:
[`decisions/ADR-0001`](decisions/ADR-0001-what-the-bridge-is-and-isnt.md).
**CP-16 builds and proves the bridge; CP-17 runs the loop.**

## Run book

```bash
# 1. Build the library wheel (in the server repo — the CP-16 wheel or
#    later; earlier wheels do not ship the pins and the trainer leg raises):
#      cd ../gsj-harness-rollout-server && python -m build --wheel
# 2. Here:
python3 -m venv .venv
./.venv/bin/pip install ../../gsj-harness-rollout-server/dist/gsj_harness_rollout_server-0.1.0-py3-none-any.whl pytest
./.venv/bin/python -m pytest -q          # 14 tests, fixture-driven, no GPU
```

On an estate with its own approved sets, export `GSJ_PINS_PATH` before
the first import of `gsj_rollout.checks` — the packaged pins are the
shipping estate's values (library ADR-0017; the bridge logs the hint
whenever hash gates fail against packaged pins).

With real slime installed (THUDM v0.3.0 + the router-tokens patch, per
the vendored `slime_bridge/README.md`), `load_sample_type()` returns the
real `Sample` and the same code path feeds training; without it, tests
run on `FallbackSample` (FINDINGS F-03/F-04).

## Fixtures

The real bodies the loop will see — see [`fixtures/README.md`](fixtures/README.md):
CP-09′ (H200 qualifying episode, validates clean) and CP-07 (pre-CP-13
shape, carries three missing-evidence findings; proves the
never-trainable path on real data).

## What CP-17 needs that this directory does not build

Named here so the loop doesn't discover them on cluster time:

1. **Weight sync — whose mechanism?** Adapter push vs checkpoint reload
   vs slime's own `update_weights_interval` against the SGLang router is
   undecided. Library A-13 (trainer drains all in-flight sessions before
   every sync) has never been exercised against a real sync.
2. **Policy-version declaration.** Nothing stamps `policy_version` today
   — the library's P3 storage half is carried and inert. The bridge
   forwards scheduler metadata when present; at ONE sync the loop may
   need nothing more, but that is a decision to take, not a default to
   inherit.
3. **Reward attach.** Every real body carries `reward: null` (F-02) — an
   evaluator/attach step must run before advantages mean anything; the
   scope law keeps it out of both the server and this bridge.
4. **Collection cadence.** CP-09′ qualified on attempt 19 of a 26-cap
   (H-41 refusal class; 7 at CP-04′ and CP-09). Budget ~19 submissions
   per qualifying episode on this estate, or arrive with a deliberate
   H-41 stance.
5. **On-estate Sample-surface verification.** F-04 / library A-26: the
   constructor surface is verified only against the vendored adapter's
   usage until real slime constructs one.
6. **`max_tokens`.** Megatron bounds sample length
   (`max_tokens_per_gpu × context_parallel_size` in the vendored config);
   the bridge accepts the parameter, the loop must set it.
