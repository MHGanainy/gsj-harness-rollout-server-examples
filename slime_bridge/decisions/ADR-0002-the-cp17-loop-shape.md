# ADR-0002 — the CP-17 loop's shape

Date: 2026-08-11 (library CP-17). Status: accepted.
Counterpart: [`ADR-0001`](ADR-0001-what-the-bridge-is-and-isnt.md) (what
the bridge is); library CP-17 report (the run and its numbers).

## Context

CP-16 built the bridge and named six inputs the loop had to decide before
cluster time: weight-sync mechanism, policy-version declaration, reward
attach, collection cadence, on-estate `Sample`-surface verification
(library A-26), and `max_tokens`. This ADR records the four that are
design decisions; the other two (A-26, `max_tokens`) are measurements and
a number, reported in the CP.

The constraint that shapes all of them: **the estate's serving engine is
qualified, and the qualification is what makes a trace comparable.** vLLM
with the four legs (`--generation-config`, `--enable-auto-tool-choice
--tool-call-parser hermes`, `--max-model-len 32768`, the symmetric
`--chat-template`) is what CP-04′/CP-09′ measured the golden pair against,
and what `pins/pins.gsj.json` hashes. slime's native training path assumes
slime owns the inference engines (SGLang behind its router, NCCL weight
sync every step). Ours it does not own.

## Decision

**1. Weight sync: checkpoint reload, not NCCL, not adapters.**
`save_model` → Megatron torch_dist → `tools/convert_torch_dist_to_hf.py` →
an HF-format directory → the engine restarted on it via
`staging/serving/serve-updated.sh`, which is `serve.sh` with exactly two
deltas: the model argument is the local checkpoint dir, and
`--served-model-name Qwen/Qwen3-0.6B` keeps the wire identity constant.
The four legs are byte-identical across the sync — the weights change and
nothing else does.

*Rejected:* slime's `--update-weights-interval` (syncs into slime-managed
SGLang engines — a different engine, a different tokenizer path, and the
pins would no longer describe the estate); adapter push (the predecessor's
LoRA mechanism; Megatron here trains full parameters, there is no adapter).

**2. Policy version: not declared at one sync, and the reason is
ordering.** The loop is strictly serialized — collection 1 completes,
training runs, the sync is proven, collection 2 starts — so every trace's
policy provenance is unambiguous from its collection epoch plus the serve
argv. Stamping would add a claim no consumer here checks. **The second
sync is where this stops being true**: two collections overlapping one
sync need P3's stamping live (library A-13's drain rule, and the storage
half is already carried end-to-end: `TaskRequest.metadata` → callback body
→ the bridge's `_scheduler_metadata` forwarding). That is a trainer-loop
line, not a server change.

**3. Reward: the citation grader, in this repo.**
`reward_cited_pages.py` — `page:N` citations within cutoff over total
citations; no artifact → 0.0, graded not skipped (the LOO baseline needs
that mass). It is the predecessor's `rlvr/grade.py` shape (R2: prior art
found, adopted, cited), ~50 lines, and it lives here because the scope law
keeps scoring out of both the server and the bridge. The bridge still only
*reads* `trace["reward"]`; the grader attaches it in memory before ingest.

**4. Collection cadence: H-41 relaxed, loudly.** The loop accepts
`COMPLETED` + zero findings + `chains_total == 1` + cutoff held, and does
**not** require a successful built-in tool call. That standard is a
collection-time quality bar from the fidelity checkpoints, never a gate in
`checks.py`; the loop does not care whether a `grep` succeeded. Every
episode's built-in outcome is still printed to an H-41 ledger so the
relaxation is visible, never silent, and `CheckPolicy` is untouched.

**Loop plumbing:** slime runs `--debug-train-only` (no SGLang engines —
generation belongs to the estate, reached only through `gsj-rollout
submit`), with `--rollout-function-path cp17_loop.rollout_fn.generate_rollout`
reading receiver-accepted bodies from disk and converting them through the
CP-16 bridge in-process. Advantages come from Polar's **vendored**
trajectory-aware LOO post-processor, path-loaded (`polar_pp.py`) rather
than copied. All episodes share one prompt group, so the LOO baseline is
across trajectories.

## Consequence

The loop closes without slime owning the serving engine, at the cost of an
engine restart per sync (~1 min here) — acceptable at one sync, and the
honest cost to state for a real run, where NCCL-into-owned-engines is the
right mechanism and the estate's qualification story would need re-doing
against SGLang. Nothing in `gsj_rollout/` changed to make the loop run.
