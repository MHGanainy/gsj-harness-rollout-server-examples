# Fixtures — real callback bodies, committed provenance

Byte-identical copies of `../slime_bridge/fixtures/` (one file of record
per project — the repo convention is self-contained project directories),
which are themselves byte-identical copies of bodies committed in the
library repo (`docs/polar/`), the provenance record. They are the two
ends of the trainer's world: one clean qualifying episode, one real body
that fails the gates.

| File | Source (library repo) | Provenance |
| --- | --- | --- |
| `cp09prime.callback_session_result.json` | `docs/polar/h200-fidelity/callback_session_result.json` | CP-09′, H200, task `cp09prime-fidelity-a19`, session `sk-polar-44620742-…` — the attempt-19 qualifying episode; validates **clean** against the shipped pins (prompt 2965 ids, response 3990, 510 trainable, zero-rate 37/510 = 7.3%, no sentinels) |
| `cp07.callback_session_result.json` | `docs/polar/pi-corpus/callback_session_result.json` | CP-07, pi 0.83.0 on the real corpus, task `cp07-pi-corpus`, triple `(case_0001, timestep-12, skill:summarize)` — **pre-CP-13 shape**: no `prompt_source`, no settings echo, no workspace echo, so trainer-side checks yield exactly `G1:missing_evidence:prompt_source`, `G7:missing_evidence:settings`, `G5:missing_evidence:workspace` |

Doctored variants are built in-test from deep copies; these files are
never mutated.
