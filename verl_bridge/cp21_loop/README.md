# cp21_loop — the verl loop, run book

One iteration of **collect → convert → train → sync → collect**, run at
library CP-21 on the H200. The bridge (`../bridge.py`) is CP-20's and
unchanged; this directory is only what the *loop* needs around it — the
CP-17 (`../../slime_bridge/cp17_loop/`) analog for the second trainer.

| file | what it is |
| --- | --- |
| `train_one_step.py` | the whole trainer side: bodies → grade (CP-17's `reward_cited_pages.py`, byte-reused) → the CP-20 bridge (three assertions live, ONE shared uid — F-10) → verl's own fit-loop plumbing mirrored from `ray_trainer.py` → one optimizer step in a real verl `TrainingWorker` (FSDP, world_size 1) → HF export via verl's own `FSDPCheckpointManager` (`save_contents=["hf_model"]`) |
| `../../slime_bridge/cp17_loop/probe_sync.py` | the sync proof, reused verbatim — noise floor first, then across the sync |

## Why this shape

- **Classic batch (run book item 1), decided at the desk.** verl never
  drives generation; the estate's qualified vLLM does, through
  `gsj-rollout submit`. At the pin the classic v0 trainer itself runs on
  the unified model-engine workers, so "feed the classic batch" =
  mirror the v0 fit loop's own data plumbing around
  `TrainingWorker.infer_batch` / `train_mini_batch` — every numeric op
  (loss, IS correction, GRPO advantage, optimizer, HF export) is verl's
  own code called directly.
- **No Ray runtime, no rollout worker, no critic, no ref** — WORLD_SIZE=1,
  plain `init_process_group`. GRPO needs no critic; no KL term is used.
- **Decoupled rollout correction** (not bypass): the actor recomputes
  `old_log_probs`; our captured `rollout_log_probs` enter the loss as
  verl's sequence-level truncated IS weights (the CP-17 `--use-tis`
  analog) — and the recompute-vs-captured delta is the replay-style
  validation, judged against the CP-09′ measured floor (0.008 / 0.21),
  never the CP-18 anchor.
- **Checkpoint reload, not NCCL** — verl's resharding syncs into engines
  verl owns; ours it does not own. Save → verl's own HF export →
  `staging/serving/serve-updated.sh` (four legs unchanged).
- **One uid group.** Every episode is the same golden triple;
  `uid="cp21-golden"` on every ingest. The bridge default (session id)
  would make every GRPO group a singleton = raw-reward advantages (F-10).
- **Flash-free engine branch** (F-12): the host has no nvcc and no
  flash-attn, so the engine runs `use_remove_padding=False` + sdpa
  (its own supported branch); the one plumbing helper that hard-requires
  flash-attn (`left_right_2_no_padding`) is reproduced in pure torch with
  its contract asserted row by row.

## Run

```bash
# 0. desk check, CPU, fixtures (no GPU, no estate):
GSJ_COLLECTED_DIR=<dir with cp09prime fixture> GSJ_ARTIFACTS_ROOT=/tmp \
GSJ_CUTOFF=12 GSJ_PAGE_COUNT=18 ./.venv/bin/python cp21_loop/train_one_step.py --dry-run

# 1. collect (host, estate up) — see the library's staging/README.md
gsj-rollout submit --config staging/rollout.h200.yaml --case case_0001 \
  --timestep 12 --prompt-file instruction.golden.txt --episodes N \
  --out ~/cp21/collect1

# 2. train: one optimizer step on GPU (venv: torch cu + tensordict 0.10 +
#    transformers 5.10 + verl @ bridge.VERL_SHA --no-deps + the wheel)
CUDA_VISIBLE_DEVICES=7 GSJ_COLLECTED_DIR=~/cp21/collect1 \
GSJ_ARTIFACTS_ROOT=~/cp04prime/artifacts GSJ_CUTOFF=12 GSJ_PAGE_COUNT=18 \
GSJ_SNAPSHOT=<pinned Qwen3-0.6B snapshot> GSJ_CKPT_DIR=~/cp21/ckpt \
GSJ_SUMMARY=~/cp21/summary.json python cp21_loop/train_one_step.py

# 3. sync: probe (noise floor first), restart the engine on
#    ~/cp21/ckpt/huggingface via serve-updated.sh, probe again, compare
python ../slime_bridge/cp17_loop/probe_sync.py probe <collected.json> before.json
# ... serve-updated.sh ~/cp21/ckpt/huggingface ...
python ../slime_bridge/cp17_loop/probe_sync.py probe <collected.json> after.json
python ../slime_bridge/cp17_loop/probe_sync.py compare before.json after.json

# 4. collect again — same command as step 1
```

`GSJ_PINS_PATH` is not set: this estate IS the shipping estate, so the
wheel's packaged pins are the right approved sets (library ADR-0017).
