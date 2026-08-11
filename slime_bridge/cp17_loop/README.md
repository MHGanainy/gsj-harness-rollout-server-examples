# cp17_loop — the loop, run book

One iteration of **collect → convert → train → sync → collect**, run at
library CP-17 on the H200. The bridge (`../bridge.py`) is CP-16's and
unchanged; this directory is only what the *loop* needs around it.

| file | what it is |
| --- | --- |
| `rollout_fn.py` | slime's `--rollout-function-path`: reads receiver-accepted `SessionResult` JSONs, grades them, converts each through the CP-16 bridge into **real** `slime.utils.types.Sample`s (library A-26's live test — the module assert refuses a `FallbackSample`) |
| `../reward_cited_pages.py` | the reward attach (F-02's answer): `page:N` citations within cutoff / total citations; no artifact → 0.0, graded not skipped. The predecessor's `rlvr/grade.py` shape |
| `polar_pp.py` | path-loads Polar's **vendored** trajectory-aware LOO post-processor — no copy, no fork (its package is also named `slime_bridge`, hence by-path) |
| `train_one_step.sh` | inside `slimerl/slime:v0.3.0`: HF → torch_dist convert, one optimizer step, torch_dist → HF export for the estate's engine |
| `probe_sync.py` | the sync proof: teacher-forced logprobs on a fixed token stream, before vs after. CP-09′ measured the engine's replay path bit-deterministic (replay-vs-replay ≡ 0.000000), so any nonzero Δ is the weights |

## Why this shape

- **No SGLang.** `--debug-train-only` keeps slime's engines out: generation
  belongs to the estate's qualified vLLM (the four legs, the pinned
  template), reached only through `gsj-rollout submit`. Swapping engines
  mid-CP would void comparability with CP-09′ and the pins.
- **Checkpoint reload, not NCCL.** slime's native `update_weights` syncs
  into engines slime owns; ours it does not own. So: save → convert to HF →
  restart the engine on the new snapshot under the same served model name.
- **One prompt group.** Every episode is the same golden triple, so all
  trajectories share `group_index=0` and the LOO baseline is across them.

## Run

```bash
# 1. collect (host, estate up) — see the library's staging/README.md
gsj-rollout submit --config staging/rollout.h200.yaml --case case_0001 \
  --timestep 12 --prompt-file instruction.golden.txt --episodes N \
  --out ~/cp17/collected

# 2. convert + train (container; GPU disjoint from the serving GPU)
docker run --rm --gpus '"device=5"' --ipc=host --shm-size=16g \
  -v /home/sysadmin:/host slimerl/slime:v0.3.0 \
  bash /host/cp17-examples/slime_bridge/cp17_loop/train_one_step.sh convert
docker run --rm --gpus '"device=5"' --ipc=host --shm-size=16g \
  -v /home/sysadmin:/host slimerl/slime:v0.3.0 \
  bash /host/cp17-examples/slime_bridge/cp17_loop/train_one_step.sh train

# 3. sync: probe, restart the engine on the exported HF dir, probe again
python3 probe_sync.py probe <collected.json> before.json
#   ... restart vLLM with --model ~/cp17/ckpt/cp17_hf_updated \
#       --served-model-name Qwen/Qwen3-0.6B (four legs unchanged) ...
python3 probe_sync.py probe <collected.json> after.json
python3 probe_sync.py compare before.json after.json

# 4. collect again — same command as step 1
```

`GSJ_PINS_PATH` is not set: this estate IS the shipping estate, so the
wheel's packaged pins are the right approved sets (library ADR-0017).
