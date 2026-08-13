# RUNBOOK — training against a gsj rollout estate, start to finish

This is the one document to read before running anything. It assumes you
have read nothing else; where a fact has a longer story, the pointer is
given, but you do not need it to run.

## What this is

`gsj-harness-rollout-server` is a rollout server for a legal-corpus agent
task: given `(case, timestep, prompt)` it runs a pinned pi agent in an
isolated sandbox with temporally-scoped retrieval (nothing past page
`timestep` is reachable) and emits a training-ready trajectory — token
ids, a loss mask, and the behaviour policy's captured logprobs, validated
by `gsj_rollout.checks` on both sides of the wire. Episode execution is
NVIDIA Polar's, vendored inside the library repo.

This project is the consumer half: `config.yaml` (what you edit),
`taskbank.parquet` (what you submit), `train.py` (collect → convert →
train → sync, one GRPO step per run, verl underneath).

## The two roles — read this before installing anything

**The trainer role** is a pip install: `gsj_rollout.client` submits tasks
and `gsj_rollout.checks` re-validates what comes back; the bridge and
`train.py` here turn accepted bodies into one verl optimizer step. This
runs anywhere with a GPU (and the desk halves run without one).

**The server role is not an install and no wheel will change that.** It
needs an *estate*: one host (the measured one is a single H200 box)
running, together —

- a **vLLM engine** serving the model with four pinned flags ("the four
  legs"): `--chat-template` (the symmetric template), `--generation-config`,
  `--enable-auto-tool-choice --tool-call-parser hermes`, and
  `--max-model-len 32768` — without the tool-call flags every episode
  errors "no completions";
- a **Forgejo git host** with one repo per case, `timestep-{T}` branches
  (built by the library repo's `corpus/` + `forgejo/` components);
- the **MCP retrieval service** (`mcp-service/`), sharing an HMAC secret
  with the gateway (env var `GSJ_MCP_TOKEN_SECRET` on both);
- **Polar's two processes** (rollout + gateway) from the library repo's
  `vendor/polar/.venv` — with `gsj_rollout` pip-installed into that venv
  (`uv pip install -p .venv/bin/python -e ../..`), or the gateway dies at
  import with `ModuleNotFoundError: No module named 'gsj_rollout'`;
- the **receiver** (`gsj-rollout serve`), which validates every callback
  and quarantines bad traces with findings attached.

Estate bring-up is the library repo's `staging/` recipes. If someone
already runs the estate, you only need this project and the four URLs.

## Install (trainer host)

```
bash install.sh
```

That is: a python3.12 venv; `requirements.txt` (the verl import closure,
stated explicitly — including five packages verl's own metadata does not
declare); the library wheel (PyPI once published; the locally built wheel
until then); and verl itself `--no-deps` from git at the pinned SHA. Why
it cannot be one `pip install X`: ADR-0023 in the library repo. Two traps
the file comments repeat: match the torch wheel to your driver (the
measured box needed `torch==2.13.0+cu126`), and never install verl *with*
its declared deps (it drags flash-attn-shaped pins; the loop runs
flash-free by construction — F-12).

**Foreign estate? Set `GSJ_PINS_PATH`.** The wheel ships the *reference
estate's* approved sets (tool roster, system prompt, settings hashes). On
your own estate every hash gate will fail `*_not_approved` — loudly, by
design — until you derive your own pins and export `GSJ_PINS_PATH` before
the first import of `gsj_rollout.checks`. On the reference estate, set
nothing.

## Configure

Copy `config.yaml` and edit the six values at the top — four estate
endpoints, the gateway `public_url`, and `traces_dir`. Everything else is
a commented default; the file itself documents each. The two that bite:

- `clone_url_for` and `mcp_url_base` must be reachable **from inside
  episode containers** (the sandbox clones and searches itself), so on a
  compose estate also set `runtime.network`.
- `polar.gateway.public_url` must work from the host **and** from inside
  containers — one URL, both audiences; never localhost.

## Run

Server side (the estate), once per session:

```
gsj-rollout serve --config config.yaml     # receiver + rendered topology;
                                           # it prints the two Polar
                                           # commands to run beside it
```

Trainer side, from this directory:

```
python train.py --collect-only             # submit the bank's train rows
python train.py --dry-run                  # grade+convert on CPU, no GPU
python train.py --snapshot <pinned HF dir> # the optimizer step + HF export
```

Then the sync, estate-side: point `staging/serving/serve-updated.sh` at
the printed HF export, probing logprobs before and after with
`slime_bridge/cp17_loop/probe_sync.py` (identical weights probe exactly
0.0; a real sync moves nearly every position). Engine downtime is about a
minute. Drain in-flight episodes before syncing — the loop is safe
serialized; overlapping collection with a sync is not yet instrumented
(A-13).

## What to expect — measured, so you don't debug a healthy system

- **Collection costs multiples of what it yields.** Under the strict
  qualification standard CP-09′ needed **19 attempts for 1 qualifying
  episode**; under the relaxed standard the loops train on, CP-17 got
  27/28 and CP-21 ran **112 attempts** (extended loudly from a stated 28)
  because the *reward*, not the pipeline, was zero at 28 and 56. Attempts
  that come back COMPLETED-but-refused are the normal shape, not a bug.
  ~36 episodes take about an hour through one serially-fed engine.
- **Reward is sparse at 0.6B, and the rate is not a stable constant**:
  one measured collection landed the citation reward 1 in 27, the next
  1 in 112 — CP-21's own conclusion is that the ~1/24 heuristic does not
  hold. A batch too small to contain a nonzero reward trains on zeros;
  `train.py` prints the reward distribution so you see it before the GPU
  does.
- **`--episodes N` means N attempts** (Polar's `num_samples`), not N
  collected episodes; rejected traces are consumed attempts, quarantined
  with findings, never auto-retried.
- **The recompute-vs-captured logprob delta sits near 0.008 mean** (tail
  to ~0.21 per position) on the measured estate — three independent
  instruments agree. That is the platform's capture floor, not an error;
  `train.py` prints it against the shipped constants.
- **Singleton GRPO groups are dropped loudly** (F-10): a group of one has
  no baseline and verl hands back the raw reward as its "advantage".
- **Entropy/KL are OFF** in this one-step shape — right for an audited
  step, wrong for a run: the measured post-sync distribution visibly
  narrowed (format-copying onset). Arm them in `loop.make_worker` before
  any multi-step schedule.
- **Thinking stays off** (`harness.thinking: "off"`): gate G6 verifies
  every assistant turn opens from the pinned no-think glue; a thinking-on
  estate fails every episode by design until the gates are re-conceived.
- Suites, if you check out the library repo: root 136, corpus 58,
  mcp-service 89, vendored Polar 175 passed / 3 pre-existing failures.

## The bank

`taskbank.parquet` — 12 rows over 4 cases (train 9 / eval 3), sha256
`ae9e0bbd…`, one flat row per `(case, timestep, prompt)` with the skill
card text resolved at build time; every column is either the triple or a
`render_task_request` argument, so a row submits with zero translation.
`train.py` trains only on `split == "train"` rows — the label is carried,
the trainer enforces it (that's this script, not the server). Rebuild:
`bash rebuild_taskbank.sh` (runs the library repo's corpus pipeline).

## slime instead of verl

The first measured loop trained with slime v0.3.0 — it needs its own
container image (`slimerl/slime:v0.3.0`, 24.4 GB; the image's Megatron,
not Polar's documented pin) and lives in `../slime_bridge/cp17_loop/`
with its own run book. This project uses verl because the trainer is a
host venv rather than an image; the traces are the same either way.

## What can still surprise you

Honest list — these are written down only as findings, and CP-26 exists
to discover which others aren't written anywhere yet: GPU topology drifts
between sessions (discover free GPUs at run time); the artifacts dir must
be visible to the grading host (sync it if you collect and train on
different machines); `python -m gsj_rollout.cli` exits 0 doing nothing —
use the `gsj-rollout` console script (library wishlist row 19, the CP-21
entry); a card edit lands as quarantines until the estate re-pins
(wishlist row 19, the CP-24 entry — the register records two rows under
that number, its own numbering slip).
