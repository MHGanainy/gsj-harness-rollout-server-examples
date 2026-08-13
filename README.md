# gsj-harness-rollout-server-examples

Consumer code for [`gsj-harness-rollout-server`], written **exactly as the
trainer would write it** — against the published import surface
(`gsj_rollout.checks`, `gsj_rollout.client`) and the callback-shaped
bodies a collect returns, never against server internals and never
importing `polar`. The repo-shape precedent is `gsj-envloader-examples`
(the predecessor's zero-CLI proof): self-contained project directories,
committed fixtures, a FINDINGS register. That register is part of the
point of this repo as much as the code.

## Projects

- **`example_project/`** — **start here.** The consumer surface, built at
  library CP-25 for Phase D: the commented `config.yaml` a stranger
  copies (six required values, everything else defaulted and documented),
  the committed CP-24 `taskbank.parquet` with its rebuild script, a
  230-line `train.py` (collect → convert → train → sync; the verl
  machinery behind `verl_bridge/loop.py`), `install.sh` — the one install
  command (library ADR-0023) — and `RUNBOOK.md`, the document to read
  first.
- **`slime_bridge/`** — callback-shaped `SessionResult` → slime `Sample`
  (v0.3.0 surface), built at library CP-16 for the M4 evaluation. The
  library's ADR-0018 records why it lives here (the scope law: a bridge
  exists to feed a trainer, so it is the trainer's); `slime_bridge/decisions/ADR-0001`
  records what it is and isn't. CP-17 runs the loop.
- **`verl_bridge/`** — callback-shaped `SessionResult` → verl `DataProto`
  (0.9.0.dev @ uni-agent's submodule pin `1ae9455`), built at library
  CP-20 for M6a: the second trainer, the proof the boundary is real.
  Same three assertions as the slime bridge, plus the batching stage
  slime never needed (`[B, L]` padding); verl is real in the tests, no
  double. `verl_bridge/decisions/ADR-0003` records the route decision
  and the uni-agent answer. **CP-21 ran the loop**:
  `verl_bridge/cp21_loop/` is the one-step harness (collected bodies →
  reward attach → the bridge → one real verl optimizer step →
  verl's own HF export for the estate engine), mirroring the v0 fit
  loop's plumbing around a standalone `TrainingWorker` — ADR-0004
  records the shape; F-12/F-13 record what verl needed worked around.
  At CP-25 the harness's machinery moved behind `verl_bridge/loop.py`
  (documented, consumer-importable); `cp21_loop/` stays verbatim as the
  CP-21 evidence artifact.

## Conventions

- No packaging: each project is a directory with its own
  `requirements.txt`, `.venv`, and run book. Nothing here is installed.
- The library arrives as a **wheel**, not a checkout — that is the point
  (the CP-16 packaging fix is what makes `checks` work from a wheel).
  Honesty rider (CP-26 F-15): until the PyPI name is claimed (library
  wishlist 17) the wheel is BUILT from a library checkout that must sit
  beside this repo — `install.sh` documents the layout — so today a
  consumer holds both trees even though only the wheel is installed. The
  server role additionally runs Polar from that same checkout's
  `vendor/polar/`.
- Commits tracking a library checkpoint are titled `CP-NN (library): …`;
  this repo has no CP numbering of its own.
- Every friction against the library surface gets a row in
  [`FINDINGS.md`](FINDINGS.md).
