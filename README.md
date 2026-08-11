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

- **`slime_bridge/`** — callback-shaped `SessionResult` → slime `Sample`
  (v0.3.0 surface), built at library CP-16 for the M4 evaluation. The
  library's ADR-0018 records why it lives here (the scope law: a bridge
  exists to feed a trainer, so it is the trainer's); `slime_bridge/decisions/ADR-0001`
  records what it is and isn't. CP-17 runs the loop.

## Conventions

- No packaging: each project is a directory with its own
  `requirements.txt`, `.venv`, and run book. Nothing here is installed.
- The library arrives as a **wheel**, not a checkout — that is the point
  (the CP-16 packaging fix is what makes `checks` work from a wheel).
- Commits tracking a library checkpoint are titled `CP-NN (library): …`;
  this repo has no CP numbering of its own.
- Every friction against the library surface gets a row in
  [`FINDINGS.md`](FINDINGS.md).
