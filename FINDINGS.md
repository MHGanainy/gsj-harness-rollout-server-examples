# FINDINGS — the external register

Frictions hit while consuming `gsj-harness-rollout-server` from the
outside. Severity: BLOCKER (cannot proceed), FRICTION (workaround
needed), DOC (surprising but documented here), COSMETIC. "library-side?"
marks findings the library should absorb. UNVERIFIED rows are reported,
not confirmed — the predecessor repo's rule: unverified is not false.

| # | severity | finding | library-side? | status |
| --- | --- | --- | --- | --- |
| F-01 | BLOCKER | `pins/` did not ship in the wheel: `checks.PINS_PATH` resolved into site-packages and every trainer-side `validate_session_result` raised `PinsConfigurationError` — law 6's trainer leg was non-functional from an installed wheel (first recorded at library CP-11b, dispositioned then as "both legs run from the checkout by design") | yes | **FIXED at library CP-16** (ADR-0017: packaged pins + `GSJ_PINS_PATH` override; proven from a scratch venv against the real CP-09′ body) |
| F-02 | DOC | `reward` is `null` on every real callback body to date (CP-06/07/09/09′). The bridge consumes what the evaluator placed — `null` → `0.0` under the reward key — so a loop run today trains on zero reward everywhere and LOO advantages are identically zero. A reward/evaluator attach step is a named CP-17 input, not a bridge feature (the scope law: scoring is out) | no — corpus/evaluator question | OPEN, CP-17 input |
| F-03 | FRICTION | slime is not importable off-estate (THUDM git + Megatron + torch; the unrelated PyPI `slime` is 0.0.0), and the vendored Polar adapter hard-raises without it. The bridge carries `FallbackSample`, a test double mirroring exactly the constructor surface the vendored adapter uses, so its assertions are testable fixture-driven. The loop must run with real slime | no — deliberate consumer choice, documented | ACCEPTED |
| F-04 | DOC | The vendored adapter passes `session_id=` (and reads `group_id`, `remove_sample`, `Status.FAILED`) on `slime.utils.types.Sample` — a surface we can only verify against real slime v0.3.0 (+ the router-tokens patch) on-estate. If the real constructor rejects any of these, the bridge fails loudly at the first conversion | unverifiable here | UNVERIFIED until CP-17 (library A-26) |
| F-05 | DOC | The vendored placeholder hardcodes `tokens=[0, 0]` / `response_length=1` — assumes token id 0 is batcher-safe for the served vocabulary. Kept verbatim in the bridge (deviating from the template needs on-estate evidence); flagged so CP-17 knows where it lives | no | OPEN, CP-17 watches |
