# gsj-harness-rollout-server-examples

**One bridge, one loop, one command.** The rollout server
([`gsj-harness-rollout-server`](https://github.com/MHGanainy/gsj-harness-rollout-server),
PyPI 0.1.7, verified at library CP-83) runs a legal-corpus research agent: given
`(case, timestep, prompt)` it runs a pinned agent in an isolated sandbox
with temporally-scoped retrieval — nothing past page `timestep` is
reachable — on an *estate* (the server-side host set: corpus, retrieval
service, git host, serving engine), and POSTs a validated,
training-ready trace. This repo is the trainer side of that wire: the
**verl bridge** (callback body → one padded `DataProto`, three
assertions standing on the path) and the **training loop** that drives
it — collect → grade → batch → train → **sync** → collect → train, N
steps end to end, with every weight sync measured rather than assumed.
It is written only against the published pip surface
(`gsj_rollout.checks`, `gsj_rollout.client`) and the callback-shaped
bodies a collect returns — never server internals, never `import polar`.

```
 the rollout server               the bridge (this repo)             verl
┌────────────────────┐          ┌──────────────────────────────┐   ┌──────────────────┐
│ task → sandbox →   │ callback │ 1 mask before ratio          │   │ one `DataProto`  │
│ agent → trace      │──JSON──▶ │ 2 sentinel rejection         │──▶│ → one optimizer  │
│ (pip: gsj-harness- │  bodies  │ 3 `checks` re-run on what    │   │   step → sync →  │
│  rollout-server)   │          │   actually arrived           │   │   collect again  │
└────────────────────┘          └──────────────────────────────┘   └──────────────────┘
```

The one command (from `example_project/`, against a running estate):

```
python train_loop.py --row 0 --steps 2 --episodes 64 \
    --snapshot <pinned HF dir> \
    --sync-cmd 'bash sync_engine_local.sh {ckpt}'
```

Use a fresh absent or empty `--run-dir` for every invocation. A nonempty
run is refused before collection; resume is unsupported. `{ckpt}` must
appear as a standalone **unquoted** word inside the quoted `--sync-cmd`
template. The loop supplies shell quoting for the exported path, including
spaces and apostrophes. The sync measurement covers command start through
`/v1/models` readiness, including shell execution; failure diagnostics
report elapsed time too.

Per step it collects N episodes (train.py's audited collect stage —
image-pin assert, F-27/F-51 counters), grades them
(`verl_bridge/reward_cited_pages.py` — the F-02 answer), converts
through the bridge (three assertions at ingest), takes one real verl
optimizer step (the worker persists across steps — optimizer state
continues), exports HF-format weights, **restarts the engine on them**
(the sync CP-17 and CP-21 did by hand, scripted at last), and **proves
the sync** with the CP-17 probe: teacher-forced logprobs on a fixed
token stream before and after — a zero-noise instrument (identical
weights probe at exactly 0.000000 on this estate), so every moved
position attributes to the weight change. A sync that "succeeds" while
zero positions move aborts the loop: that engine is serving the old
weights, and training on its collections would be the silent staleness
this project exists to catch.

**Why restart-based sync, and not in-place** — established from verl's
own source at the pinned SHA, not guessed (F-79): verl cannot push
weights into an engine it did not spawn. Every in-place path ends in a
Ray `collective_rpc` against a verl-launched vLLM engine whose workers
carry verl's injected extension class, over node-local CUDA-IPC — a
standalone `vllm serve` behind Polar's gateway has none of that, and it
cannot be retrofitted onto a running server. The in-place middle path
(verl's standalone replica serving a real OpenAI endpoint the gateway
could front) exists but means the engine lives and dies with the verl
Ray job — surrendering the serving. This estate keeps engine ownership;
the loop keeps the ~1-minute restart and measures it every time.

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

*Notation, used throughout*: `F-nn` = rows in
[`FINDINGS.md`](FINDINGS.md) (one register across this repo and the demo
repo); `CP-nn` = library checkpoints (`docs/reports/CP-nn.md` in the
library repo); `A-nn` = library charter §4 assumptions; `P1`–`P3` = the
library's carried Polar patches.

## What left, and where it went

Until CP-69 this repo carried a second bridge. **`slime_bridge/` — the
complete slime v0.3.0 path: the bridge and its 14-test suite, the CP-17
one-step loop (`cp17_loop/`: rollout function, vendored-LOO shim,
one-step trainer, sync probe, smoke test, run book), its two ADRs and
fixtures — is frozen at tag
[`slime-cp17`](https://github.com/MHGanainy/gsj-harness-rollout-server-examples/tree/slime-cp17/slime_bridge)**
(commit `73e63f0e`, tree `5f28d0f5`); restore with
`git checkout slime-cp17 -- slime_bridge`. The CP-17 evidence it holds:
one real Megatron optimizer step on 27 collected episodes (grad_norm
0.4513, the synced run; the unguarded control measured F-08's 1e6
advantage explosion), the sync proof (noise floor 0.000000; across the
sync mean|Δ| 0.041835, 5623/5782 positions moved), 8/8 post-sync
episodes. The trainer-agnostic claim — two trainers, two loops, zero
server changes (CP-17 + CP-21) — is followable through that tag. Two
files moved to main instead of leaving, because the verl path uses them:
the reward grader (`verl_bridge/reward_cited_pages.py`, with its 5
tests) and the sync probe (`verl_bridge/probe_sync.py`).

## The bridge, and what the loop still hands you

| | `verl_bridge/` |
| --- | --- |
| trainer | verl 0.9.0.dev @ `1ae9455` |
| batch shape | one padded `[B, L]` `DataProto`, uid-grouped |
| proven by | **CP-21's loop**: 110 episodes → one `DataProto` → one real verl step (`pg_loss −0.0944`) → engine sync → 8/8 post-sync episodes; **CP-69's two-step loop** (`train_loop.py`, off-mode, 2026-08-31): 192 + 128 attempts → two optimizer steps through ONE persisting worker (grad_norm 0.0332 then 0.4710, both unclipped under the F-08 guard) → both syncs probe-proven (mean\|Δ\| 0.040133 then 0.074026 over a 0.000000 floor; 165 s / 185 s downtime) → the post-sync collection's reward went **1/191 → 78/128** by near-verbatim format-copying of the one rewarded trajectory — CP-17's mode-collapse caution, measured at scale |
| trainer real in tests? | yes — real verl at the pin, no double (26 bridge tests + 5 grader tests) |
| stack cost | host python3.12 venv; flash-attn-free hosts need the F-12/F-13 workarounds (carried in `verl_bridge/loop.py`) |

The proof loops ran **thinking-off**; the shipped example default is ON
(`example_project/RUNBOOK.md` §Thinking carries the per-mode
expectations). What the loop still hands you — supplied once by the
proofs, owned by any real run:

- **Reward.** Every real callback body arrives `reward: null` (F-02);
  the scope law keeps scoring out of the server AND the bridge. The
  citation grader (`verl_bridge/reward_cited_pages.py`) is the worked
  example, not a solution — and it is sparse at 0.6B (see the costs).
- **Grouping.** verl's GRPO hands a singleton uid group its RAW reward
  as "advantage" (F-10) — the loop trains ONE bank row per run so its
  episodes share a uid, and refuses a group of one.
- **The F-08 guard, set by default.** The loop runs Dr.GRPO
  (`norm_adv_by_std` off) — verl is structurally immune to slime's 1e6
  explosion (F-09) but std-normalization still amplified the lone
  rewarded episode to +10.39 at CP-21; opting back in is an explicit
  flag that warns.
- **Entropy/KL control is unsupported.** Nonzero `--entropy-coeff` and
  `--use-kl-loss` refuse before collection, imports or worker allocation.
  `loop.make_worker` also refuses unsupported values. The loop still warns
  on multi-step runs with both off: CP-21 observed distribution narrowing.
  Phase 5 must fund safe entropy and a reference-policy leg; neither
  control is enabled by this checkpoint. See [the cost and refusal contract](example_project/RUNBOOK.md#cp-86-training-contract).

**The bridge is consumer code, not library code** (library ADR-0018: a
bridge exists to feed a trainer, so it is the trainer's). A bug in
`bridge.py` is yours to find and this repo's register to record —
[`FINDINGS.md`](FINDINGS.md), not the library's tracker. What the
library warrants is the surface the bridge consumes: the callback body's
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
  denser grader — the loop REFUSES a zero-reward batch rather than
  training on zeros while every pipeline light stays green
  (`--allow-zero-advantage` overrides, loudly).
- **The replay floor** — how far the engine's captured logprobs sit
  from a trainer-side recompute on identical weights, the number that
  says whether the captured values are trustworthy: mean |Δ| ≈ 0.008,
  per-position tail 0.21 (thinking-off, CP-09′), confirmed from inside
  both trainers (slime recompute 0.008813 at CP-17, verl 0.009442 at
  CP-21); thinking-ON measured mean |Δ| 0.016546 with a 0.35% tail over
  0.21 (CP-32) — ~2× the off constant, same order. Same order as the
  floor: capture noise. An order of magnitude above: a real mismatch
  (wrong snapshot, wrong engine flags). From step 2 on, the loop reads
  this same number as a free end-to-end re-verification of the previous
  sync: the engine that collected and the worker that recomputes hold
  the same post-step weights, so floor-level agreement means the sync
  really served the trained checkpoint.
- **The sync.** Export HF-format weights, restart the serving engine,
  **~1 minute of engine downtime per sync**, measured by the loop each
  time. A-13's drain rule is satisfied by construction — the loop is
  serialized: collection is fully terminal before a sync starts, and
  the next collection starts only after `/v1/models` confirms the new
  checkpoint, so no batch ever spans a sync. Proven with a zero-noise
  probe every run (identical weights probe exactly 0.0; a real sync
  moved >95% of probed positions each measured time).
- **The GPU step.** One-step GRPO at 0.6B is not small: CP-21's 110-row
  batch (off) peaked 71.3 GiB allocated on an H200; CP-32's ON leg (72
  rows, longest 32,645 ids) peaked **94% of the device** (135,093 MiB),
  wall 11m38s — and unchunked entropy at padded width is a known verl
  OOM trap on flash-free hosts (F-13).

## What is not solved

The loop closes — N steps, sync measured each time — and what remains is
the honest list for a REAL training run (CP-17's list, CP-21's
additions, CP-69's answers):

1. **In-place weight sync is not buildable against an engine you own**
   — answered at CP-69 from verl's source, not assumed (F-79). The
   restart is ~1 min, serialized; escaping it means verl owning the
   serving (a different estate) or a LoRA-only adapter swap (a shim
   verl does not ship). Today: restart, measured.
2. **Concurrency.** The moment collection and training overlap, A-13's
   drain rule needs a real barrier and P3's policy-version stamping
   goes live — both carried, both inert; every measured sync was
   serialized by construction (the loop states this at each boundary).
3. **Denser reward, a stronger actor — and more tasks.** 1-in-tens at
   0.6B makes a defensible single step and a hopeless training curve;
   and the committed bank is 12 tasks over 4 cases (9 train) — a
   proof-of-loop bank, not a training corpus. Growing it means running
   the library repo's corpus pipeline (F-38).
4. **Tuning instead of guarding.** The loop ships the F-08 guard
   (Dr.GRPO) and refuses unsupported entropy/KL flags — but those are guards; a
   real run TUNES clip ratios, lr, and the controls against its own
   curve rather than inheriting CP-17's knobs.
5. **Throughput.** Pooled collection is measured (above); collection
   overlapped with training is not — parallel submission at cadence is
   exactly where the async callback path and the drain rule start
   interacting for real.
6. **Checkpoint retention.** Each step writes a full HF export (~1.5 GB
   at 0.6B) under the run dir; retention, cleanup, and checkpoint
   identity are entirely unbuilt.

One more cost that is not this repo's to count: the SERVER side needs an
estate. The library README covers the two-role split;
[`gsj-rollout-demo`](https://github.com/MHGanainy/gsj-rollout-demo) is
the bring-your-own-estate walk, measured from nothing (cold `up` ~2.5
min, 6–7 GB of images, ~20–40 s per episode against a host-local 0.6B —
synthetic two-case corpus, you bring the engine; register rows
F-54–F-78 and F-80). Since CP-81 it also supplies thirty synthetic
decisions written for those cases, precedent prompts and a transcript's
decision-citation view. Acceptance is a trace/provenance result, not a
grade of the answer or its citations. The reference estate's full bring-up spans the library
repo's `estate/` plus the predecessor repo's BRINGUP walk (F-33).

## Layout

- **`example_project/`** — **start here.** The consumer's project: the
  commented `config.yaml` edited in place, the committed
  `taskbank.parquet` (12 tasks over 4 cases, 9 train),
  **`train_loop.py` (the loop — the one command above)**, `train.py`
  (the audited single step underneath it: same stages, one step, the
  CP-44 front door), `sync_engine_local.sh` (the restart sync for a
  train-where-you-serve host; the workstation-side form is the
  library's `estate/serving/serve-updated.sh` — F-29 says which side
  runs which), `install.sh` (CUDA-12.x driver hosts additionally need
  the documented torch `+cu126` re-pin — RUNBOOK §Install), and
  `RUNBOOK.md`, the document to read first. **Thinking ships ON**
  (`thinking: "medium"`, library CP-31); the serve leg must export
  `GSJ_PINS_PATH` to the on-mode pins or every episode quarantines
  server-side while the trainer side looks healthy — RUNBOOK §Thinking;
  `--thinking off` collects the control.
- **`verl_bridge/`** — callback body → verl `DataProto` (0.9.0.dev @
  `1ae9455`), built at library CP-20; CP-21 ran its loop (`cp21_loop/`,
  kept verbatim as evidence; the reusable machinery is
  `verl_bridge/loop.py`). Since CP-69 it also carries the grader
  (`reward_cited_pages.py`) and the sync probe (`probe_sync.py`) — both
  moved from the slime tree, which the verl path always used. Route
  decision: `verl_bridge/decisions/ADR-0003`; the loop's shape:
  `ADR-0004`.

## Conventions

- No packaging: each project is a directory with its own
  `requirements.txt`, `.venv`, and run book. Nothing here is installed.
- The library arrives as a **wheel**, not a checkout — that is the
  point. Since library CP-29 the wheel comes from PyPI
  (`pip install gsj-harness-rollout-server` — 0.1.7 verified at library
  CP-83), so the trainer role holds ONE tree: this repo. A library
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
