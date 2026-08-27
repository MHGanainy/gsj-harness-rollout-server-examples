# gsj-harness-rollout-server-examples

**Traces in, batches out, for two trainers.** The rollout server
([`gsj-harness-rollout-server`](https://github.com/MHGanainy/gsj-harness-rollout-server),
PyPI 0.1.2) runs a legal-corpus research agent: given
`(case, timestep, prompt)` it runs a pinned agent in an isolated sandbox
with temporally-scoped retrieval — nothing past page `timestep` is
reachable — on an *estate* (the server-side host set: corpus, retrieval
service, git host, serving engine), and POSTs a validated,
training-ready trace. The code here turns that callback body into your
trainer's batch type — slime `Sample`s or a verl `DataProto` — with
three assertions standing on the path. It is written only against the
published pip surface (`gsj_rollout.checks`, `gsj_rollout.client`) and
the callback-shaped bodies a collect returns — never server internals,
never `import polar`.

```
 the rollout server               the bridge (this repo)             your trainer
┌────────────────────┐          ┌──────────────────────────────┐   ┌──────────────────┐
│ task → sandbox →   │ callback │ 1 mask before ratio          │   │ slime `Sample`s  │
│ agent → trace      │──JSON──▶ │ 2 sentinel rejection         │──▶│ verl `DataProto` │
│ (pip: gsj-harness- │  bodies  │ 3 `checks` re-run on what    │   │   — your batch   │
│  rollout-server)   │          │   actually arrived           │   └──────────────────┘
└────────────────────┘          └──────────────────────────────┘
```

The three assertions are the reason a bridge exists at all: a logprob at
a masked position becomes a silent importance ratio inside your trainer
(verl's bypass mode copies the captured array wholesale into
`old_log_probs`); a `-9999.0` sentinel is finite and passes every shape
check; and validation at the server proves nothing about what crossed
the wire, so the same `checks` validators run again trainer-side, on the
arrived bytes, from the wheel's own pins — the estate's approved hashes:
tool roster, system prompt, engine settings, the per-mode G6 tail. Each
assertion has failing-when-removed test evidence (library
`docs/reports/CP-16.md` / `CP-20.md`).

*Notation, used throughout*: `F-nn` = rows in this repo's
[`FINDINGS.md`](FINDINGS.md); `CP-nn` = library checkpoints
(`docs/reports/CP-nn.md` in the library repo); `A-nn` = library charter
§4 assumptions; `P1`–`P3` = the library's carried Polar patches.

## The two bridges

| | `slime_bridge/` | `verl_bridge/` |
| --- | --- | --- |
| trainer | slime v0.3.0 (THUDM) + Megatron | verl 0.9.0.dev @ `1ae9455` |
| batch shape | per-episode `Sample` objects, `group_id`-linked | one padded `[B, L]` `DataProto`, uid-grouped |
| proven by | **CP-17's loop**: 27 episodes → real `Sample`s → one Megatron optimizer step (grad_norm 0.4513, the synced run) → engine sync → 8/8 post-sync episodes | **CP-21's loop**: 110 episodes → one `DataProto` → one real verl step (`pg_loss −0.0944`) → engine sync → 8/8 post-sync episodes |
| trainer real in tests? | no — `FallbackSample` double off-estate (F-03/F-04); the real surface verified on-estate at CP-17 | yes — real verl at the pin, no double (26 tests) |
| stack cost | 24.4 GB image (~55 GB on disk), pulled through a firewall | host python3.12 venv; flash-attn-free hosts need the F-12/F-13 workarounds (carried in `cp21_loop/`) |

Both proof loops ran **thinking-off** — the only mode before library
CP-30; the shipped example default is now ON. The bridge READMEs carry
per-document mode riders; `example_project/RUNBOOK.md` §Thinking carries
the per-mode expectations.

What each still needs from you — the proof loops supplied these once; a
real run must own them:

- **Reward.** Every real callback body arrives `reward: null` (F-02);
  the scope law keeps scoring out of the server AND the bridge. The
  citation grader (`slime_bridge/reward_cited_pages.py`) is the worked
  example, not a solution.
- **Weight sync.** Checkpoint reload with an engine restart is the one
  proven mechanism (~1 min downtime per sync); nothing non-restart
  exists here.
- **Grouping.** verl's GRPO hands a singleton uid group its RAW reward
  as "advantage" (F-10) — episodes of one prompt must share a uid, and
  your collection cadence must produce n > 1 per prompt.

**Both bridges are consumer code, not library code** (library ADR-0018:
a bridge exists to feed a trainer, so it is the trainer's). A bug in
`bridge.py` is yours to find and this repo's register to record —
[`FINDINGS.md`](FINDINGS.md), not the library's tracker. What the
library warrants is the surface the bridges consume: the callback body's
shape and `gsj_rollout.checks`.

## What it costs, measured

From real runs on the reference estate (one H200 box, Qwen3-0.6B serving
and training), each number labelled with its thinking mode and source
checkpoint. Every constant here is that pairing: the harness is
model-general (a second family ran at library CP-38), but a different
model family means re-deriving the pins, two family-bound serving flags
(chat template, tool-call parser), and your own replay floor — a
re-qualification pass, not a config edit. `example_project/RUNBOOK.md`
carries the operational versions and the per-mode expectations
(§Thinking).

- **Collection.** The default 72-attempt collect, pooled over 6 episode
  containers, same day and same engine both legs (CP-32, 2026-08-14):
  **13m35s thinking-ON / 4m36s OFF — 2.95×**. Yield under the relaxed
  standard the loops train on: 72/72 in both modes (CP-26, off: 71/72).
  The oft-quoted **≈19 attempts per qualifying episode** is the STRICT
  CP-09′ standard (thinking-off era) — not what the loops use, but the
  rate you inherit if you demand a successful built-in tool call per
  episode.
- **The qualification standard is a choice, and attrition lives
  elsewhere.** The loops accept: session COMPLETED, `checks` findings
  `[]`, one reconstructed chain, retrieval cutoff held; successful
  built-ins informational only. Under the strict standard, 89 of
  CP-21's 112 attempts (off) would have been discarded (CP-17, off: 24
  of 27). Length-terminated episodes QUALIFY by design (library
  ADR-0025 — surfaced, never screened): at scale, 7/72 thinking-ON
  attempts hit the 32k window and every one entered the batch (CP-32);
  the `length-terminated: K/N` line is printed, and dropping such rows
  is your one-line policy.
- **Reward is sparse at 0.6B, and the rate is not a stable constant.**
  Citation reward measured: 1/27 (CP-17, off) · identically ZERO at 28
  and at 56 attempts, then 1/112 (CP-21, off) · 1/72 ON vs 0/72 OFF the
  same day (CP-32). CP-21's collection ran 4× its stated budget before
  the reward was non-degenerate. Budget for that, or arrive with a
  denser grader — a batch too small to contain a nonzero reward trains
  on zeros while every pipeline light stays green.
- **The replay floor** — how far the engine's captured logprobs sit
  from a trainer-side recompute on identical weights, the number that
  says whether the captured values are trustworthy: mean |Δ| ≈ 0.008,
  per-position tail 0.21 (thinking-off, CP-09′), confirmed from inside
  both trainers (slime recompute 0.008813 at CP-17, verl 0.009442 at
  CP-21); thinking-ON measured mean |Δ| 0.016546 with a 0.35% tail over
  0.21 (CP-32) — ~2× the off constant, same order. Same order as the
  floor: capture noise. An order of magnitude above: a real mismatch
  (wrong snapshot, wrong engine flags).
- **The sync.** Export HF-format weights, restart the serving engine
  (`serve-updated.sh`, workstation-side — F-29), **~1 minute of engine
  downtime per sync**. A-13's drain rule was satisfied by construction
  each time — the loop is serialized, zero in-flight episodes at the
  boundary — so the *situation* is proven, not a drain *mechanism*.
  Proven three times with a zero-noise probe (CP-17/CP-21/CP-32:
  identical weights probe exactly 0.0; a real sync moved >95% of probed
  positions each time).
- **The GPU step.** One-step GRPO at 0.6B is not small: CP-21's 110-row
  batch (off) peaked 71.3 GiB allocated on an H200; CP-32's ON leg (72
  rows, longest 32,645 ids) peaked **94% of the device** (135,093 MiB),
  wall 11m38s — and unchunked entropy at padded width is a known verl
  OOM trap on flash-free hosts (F-13).

## What is not solved

The loops close — collect → convert → train → sync → collect again, on
both trainers, with the boundary unchanged — and each closure produced
its list of what a REAL training run still needs. These are the
trainer's, i.e. yours (CP-17 §"what a real training run would need",
plus CP-21's addition):

1. **A weight sync that does not restart the engine.** NCCL into
   trainer-owned engines means re-qualifying the estate — the pins and
   the golden-pair fidelity evidence are vLLM-shaped. Today: restart,
   ~1 min, serialized.
2. **Concurrency.** The moment collection and training overlap, A-13's
   drain rule needs a real barrier and P3's policy-version stamping
   goes live — both carried, both inert; every measured sync was
   serialized by construction.
3. **Denser reward, a stronger actor — and more tasks.** 1-in-tens at
   0.6B makes a defensible single step and a hopeless training curve;
   and the committed bank is 12 tasks over 4 cases (9 train) — a
   proof-of-loop bank, not a training corpus. Growing it means running
   the library repo's corpus pipeline (F-38).
4. **The F-08 guard.** Polar's vendored slime-path LOO post-processor
   (`reward_post_process.py`) divides by an epsilon-floored stdev;
   degenerate variance — the NORMAL case under sparse reward — exploded
   one advantage to 1e6, measured live (CP-17, off). Worked around
   trainer-side (slime's `--disable-grpo-std-normalization`), not
   fixed. verl is structurally immune (F-09) but leans on clipping
   instead: CP-21's step (off) ran grad_norm 2.32 into a 1.0 clip —
   visible and bounded, but a real run tunes it rather than leaning on
   it.
5. **Entropy/KL control** (CP-21's addition). One clipped sparse-reward
   step with the controls off visibly narrowed the sampling
   distribution: post-sync, 8/8 episodes earned reward by near-verbatim
   format-copying of the one rewarded trajectory — a mode-collapse
   onset, at ONE step. verl has both controls; the harness left them
   off by design.
6. **Throughput.** Pooled collection is measured (above); collection
   overlapped with training is not — parallel submission at cadence is
   exactly where the async callback path and the drain rule start
   interacting for real.
7. **Checkpoint retention.** One step wrote ~2 GB of checkpoints
   (CP-17, off; plus the 24.4 GB slime image, ~55 GB on disk);
   retention, cleanup, and checkpoint identity are entirely unbuilt.

One more cost that is not this repo's to count: the SERVER side needs an
estate. The library README covers the two-role split;
[`gsj-rollout-demo`](https://github.com/MHGanainy/gsj-rollout-demo) is
the bring-your-own-estate walk, measured from nothing (cold `up` ~2.5
min, 6–7 GB of images, ~20–40 s per episode against a host-local 0.6B —
synthetic two-case corpus, you bring the engine; register rows
F-54–F-69). The reference estate's full bring-up spans the library
repo's `estate/` plus the predecessor repo's BRINGUP walk (F-33).

## Projects

- **`example_project/`** — **start here.** The full loop as a consumer
  runs it: the commented `config.yaml` edited in place (six SUPPLY
  values — plus two handovers that travel OUTSIDE the file: the MCP
  secret and the engine's HF revision, on the RUNBOOK's handover list),
  the committed `taskbank.parquet` (12 tasks over 4 cases, 9 train),
  `train.py` (collect → convert → train → sync; verl underneath, via
  `verl_bridge/loop.py`), `install.sh` (CUDA-12.x driver hosts
  additionally need the documented torch `+cu126` re-pin — RUNBOOK
  §Install has the measured traps), and `RUNBOOK.md`, the document to
  read first. **Thinking ships ON** (`thinking: "medium"`, library
  CP-31); the serve leg must export `GSJ_PINS_PATH` to the on-mode pins
  or every episode quarantines server-side while the trainer side looks
  healthy — RUNBOOK §Thinking has the exact command and the failure
  signatures; `--thinking off` collects the control.
- **`slime_bridge/`** — callback body → slime `Sample` (v0.3.0), built
  at library CP-16; CP-17 ran its loop (`cp17_loop/`, kept as evidence
  — the runnable `example_project` loop is the verl path). What it is
  and isn't: `slime_bridge/decisions/ADR-0001`; ownership: library
  ADR-0018.
- **`verl_bridge/`** — callback body → verl `DataProto` (0.9.0.dev @
  `1ae9455`), built at library CP-20; CP-21 ran its loop (`cp21_loop/`,
  kept verbatim as evidence; the reusable machinery is
  `verl_bridge/loop.py`). Route decision:
  `verl_bridge/decisions/ADR-0003`; the loop's shape: `ADR-0004`.

## Conventions

- No packaging: each project is a directory with its own
  `requirements.txt`, `.venv`, and run book. Nothing here is installed.
- The library arrives as a **wheel**, not a checkout — that is the
  point. Since library CP-29 the wheel comes from PyPI
  (`pip install gsj-harness-rollout-server` — 0.1.2 as of library
  CP-34), so the trainer role holds ONE tree: this repo. A library
  checkout sitting beside this repo (with a built wheel in its `dist/`)
  overrides the index install — the developer path; `install.sh`
  documents it. The server role still runs Polar from that checkout's
  `vendor/polar/`, so it holds both trees.
- Commits tracking a library checkpoint are titled `CP-NN (library): …`;
  this repo has no CP numbering of its own.
- Every friction against the library surface gets a row in
  [`FINDINGS.md`](FINDINGS.md) — the register is part of the point of
  this repo as much as the code (the repo-shape precedent is
  `gsj-envloader-examples`, the predecessor's zero-CLI proof).
