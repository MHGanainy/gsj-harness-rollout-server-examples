# RUNBOOK — training against a gsj rollout estate, start to finish

This is the one document to read before running anything. It assumes you
have read nothing else; where a fact has a longer story, the pointer is
given, but you do not need it to run.

## What you need in hand — from your operator, before any command

The stranger test (CP-26) measured this list; nothing on it is fetchable
or derivable from the documents alone (F-14, F-22, F-23):

- **Two directories, side by side**: this examples repo AND the library
  repo (`gsj-harness-rollout-server`) as its SIBLING — install.sh builds
  the wheel from `../../gsj-harness-rollout-server/`. Neither is
  published: the examples repo has no public remote at all, and the
  library's GitHub mirror (github.com/MHGanainy/gsj-harness-rollout-server)
  is public but STALE (last pushed at CP-19 — it predates this entire
  consumer surface; do not install from it). Until PyPI (library
  wishlist 17), your operator hands you both trees.
- **Note**: `train.py` is NOT self-contained in `example_project/` — it
  imports `../verl_bridge/` (the conversion) and
  `../slime_bridge/reward_cited_pages.py` (the grader). Move the whole
  examples repo, never the one directory (F-32).
- **The estate handover**: the four endpoint values (Configure below),
  the **MCP HMAC secret's value** (the service was started with it; a
  mismatch surfaces only mid-episode as failed tool calls — there is no
  verify-it-first probe today), and the **served model's HF revision**
  (the engine pins one; on the reference estate:
  `c1899de289a04d12100db370d81485cdf75e47ca` for `Qwen/Qwen3-0.6B`).
- **Tools on the box**: `python3.12`, and `uv` for the server-side Polar
  venv (not preinstalled everywhere: `curl -LsSf
  https://astral.sh/uv/install.sh | sh` as your own uid).

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
  `vendor/polar/.venv`. That venv does NOT exist in a fresh checkout
  (F-21); provision it once, from the library repo root:
  `cd vendor/polar && uv venv --python 3.12 && uv pip install -e .`
  (Polar's own README recipe). `gsj_rollout` must be importable inside
  it, or the gateway dies with `ModuleNotFoundError: No module named
  'gsj_rollout'` — the commands `gsj-rollout serve` prints already
  handle this by inlining a `PYTHONPATH` to your trainer venv, so you
  need no extra install if you run them as printed;
- the **receiver** (`gsj-rollout serve`), which validates every callback
  and quarantines bad traces with findings attached.

Estate bring-up: THIS repo pair does not carry the full recipe. The
library repo's `staging/README.md` holds only this estate's deltas and
the serving scripts; the authoritative cold-start walk
(`staging/BRINGUP.md`) lives in a third repo, the predecessor
`gsj-envloader`, which your operator holds (F-33). If someone already
runs the estate, you need only this project and the handover list above.

## Install (trainer host)

```
bash install.sh
```

That is: a python3.12 venv; `requirements.txt` (the verl import closure,
stated explicitly — the desk closure verl's metadata does not declare
PLUS the GPU-path three found at CP-26: uvicorn, fastapi, peft — F-17);
the library wheel (PyPI once published; the locally built wheel until
then — if `dist/` is empty the script stops FIRST, before any download,
and prints copy-pasteable build commands that work with no system pip,
F-15/F-31); and verl itself `--no-deps` from git at the pinned SHA. Why
it cannot be one `pip install X`: ADR-0023 in the library repo.

Three traps, measured (F-16, F-28, F-30):

- **torch**: the PyPI default is a cu13x build; on a CUDA-12.x driver it
  imports fine, `install.sh` prints `install OK`, and the failure waits
  until the first CUDA call of the GPU step ("driver too old"). The
  script now warns at install time when it sees this. The cure — and the
  `+cu126` local tag is REQUIRED, a bare `==2.13.0` re-pin is already
  "satisfied" and silently no-ops:
  `./.venv/bin/pip install 'torch==2.13.0+cu126' --index-url
  https://download.pytorch.org/whl/cu126`
- **pip noise**: after the `--no-deps` verl install, any later install in
  this venv prints an ERROR-labelled "verl … requires pylatexenc /
  tensorboard / wandb …" grumble. Deliberate (ADR-0023); the measured
  loop runs without them; ignore it.
- **the venv is never activated for you**: every command below assumes
  `example_project/.venv` — either `source .venv/bin/activate` once, or
  prefix `./.venv/bin/` as this document's commands do from here on.
  Bare `python` does not even exist on a stock Ubuntu box.

**Foreign estate? Set `GSJ_PINS_PATH`.** The wheel ships the *reference
estate's* approved sets (tool roster, system prompt, settings hashes). On
your own estate every hash gate will fail `*_not_approved` — loudly, by
design — until you derive your own pins and export `GSJ_PINS_PATH` before
the first import of `gsj_rollout.checks`. On the reference estate, set
nothing — and know that the warning still prints, in full, on EVERY
import (`install.sh`, `gsj-rollout serve`, every `train.py` run). On the
reference estate it is noise by design; the correct response is to do
nothing (F-39).

## Configure

Edit `config.yaml` **in place** (train.py and `gsj-rollout serve` read
the file beside the script by default, and the commands below pass no
`--config`; an edited copy under another name is read by nothing —
F-19). Six values marked SUPPLY: three estate service URLs
(`clone_url_for`, `mcp_url_base`, `serving_base_url`), the served-model
name (`estate.model` — a name, not a URL, F-24), the gateway
`public_url`, and `traces_dir`. Everything else is a commented default;
the file itself documents each. The ones that bite:

- `clone_url_for` and `mcp_url_base` must be reachable **from inside
  episode containers** (the sandbox clones and searches itself), so on a
  compose estate also set `runtime.network`.
- `polar.gateway.public_url` must work from the host **and** from inside
  containers — one URL, both audiences; never localhost. On a compose
  estate the usual answer is the estate network's own gateway IP:
  `docker network inspect <runtime.network>
  --format '{{(index .IPAM.Config 0).Gateway}}'`.
- `traces_dir` must be writable by the uid that runs `gsj-rollout serve`
  (it mkdirs at startup; a root-owned parent fails there — F-26).
- the MCP secret travels OUTSIDE this file: export the value your
  operator gave you as `GSJ_MCP_TOKEN_SECRET` when starting the gateway
  (the printed command has the slot). Today there is no way to test a
  candidate value against the running service before an episode spends
  it (F-22).

Sanity-check the four endpoint values in-band before running anything:
`curl <forgejo>/api/healthz`, `curl <mcp>/health`, `curl
<serving>/health`, and `curl <serving>/v1/models` — the last one's `id`
must equal `estate.model` byte-for-byte.

## Run

Server side (the estate), once per session:

```
./.venv/bin/gsj-rollout serve --config config.yaml
```

Run it in the FOREGROUND first: it prints the rendered topology path and
the two Polar commands to run beside it, and that print is python
stdout — under `nohup … > log` it sits block-buffered and the log shows
nothing while the receiver listens silently (F-20). Read the commands,
then background it with `PYTHONUNBUFFERED=1` if you want the log live.
The two printed Polar commands use a path RELATIVE to the library repo
root (`vendor/polar/.venv/bin/polar`) — run them with the library repo
as your cwd (F-21), the gateway one with your operator's secret in the
`GSJ_MCP_TOKEN_SECRET=<secret>` slot.

Trainer side, from this directory (`./.venv/bin/python`, or activate):

```
./.venv/bin/python train.py --collect-only        # submit the bank's train rows
./.venv/bin/python train.py --dry-run             # grade+convert on CPU, no GPU
./.venv/bin/python train.py --snapshot <hf dir>   # the optimizer step + HF export
```

`<hf dir>` is the SERVED model's weights, pinned to the ENGINE's
revision — fetch it with the revision from your handover (F-23):

```
./.venv/bin/hf download Qwen/Qwen3-0.6B --revision <engine revision> \
    --local-dir ./snapshot
```

An unpinned download that happens to equal the engine's weights works;
one that doesn't shows up ONLY as an elevated recompute-vs-captured
delta (the replay line), indistinguishable from capture noise — pin it.

The GPU leg, measured at CP-26 on the reference estate (F-34, F-35): it
takes the first visible CUDA device (`cuda:0`) — on a shared box set
`CUDA_VISIBLE_DEVICES=<free gpu>` yourself after checking `nvidia-smi`;
expect ~20 minutes and ~90 GB for a 71-row 0.6B batch with long rows,
during which the console is silent — and a
`CUDACachingAllocator … memory allocation failed` WARNING mid-leg is
the allocator retrying, not the crash it resembles (F-12/F-13's
padded-width shape; the leg completed through it).

Two stores, one relationship (F-37): the receiver writes every ACCEPTED
trace to `traces_dir` (the durable archive); `train.py --collect-only`
independently saves the bodies it collected to `./collected/` (the
working set the later stages read). Deleting `collected/` loses nothing
the archive doesn't hold; deleting `traces_dir` loses the training data.

Then the sync — **workstation-side, not estate-side** (F-29):
`staging/serving/serve-updated.sh` drives the estate over ssh and
assumes the alias in `GSJ_VLLM_SSH_HOST` (default `h200-admin`)
resolves — that is the operator workstation, never the estate box
itself, where it dies with a misleading `ssh: Could not resolve
hostname` after announcing the engine stop. (train.py's closing
printout still says "estate-side" — it is wrong the same way; library
wishlist.) Point it at the printed HF export, probing logprobs before
and after with `slime_bridge/cp17_loop/probe_sync.py` (identical
weights probe exactly 0.0; a real sync moves nearly every position).
Engine downtime is about a minute. Drain in-flight episodes before
syncing — the loop is safe serialized; overlapping collection with a
sync is not yet instrumented (A-13).

## What to expect — measured, so you don't debug a healthy system

- **Qualification is NOT where the attrition is** — reward is. Under the
  relaxed standard this loop trains on, expect nearly every attempt to
  qualify: CP-17 measured 27/28 (2026-08-11), CP-26 measured **71/72 in
  under 7 minutes** with Polar pooling six episode containers against one
  engine (2026-08-13) — a 98% yield is the healthy shape, not a broken
  gate (F-18). The famous attrition numbers belong elsewhere: **19
  attempts per qualifying episode** is the STRICT qualification standard
  (CP-09′, not this loop), and CP-21's **112 attempts** (extended loudly
  from a stated 28) were about the *reward*, not the pipeline, being
  zero at 28 and 56. Attempts that come back COMPLETED-but-refused are
  still normal, just rare here. The old "~36 episodes/hour" figure was a
  serially-fed engine; pooled collection runs ~6× that.
- **Reward is sparse at 0.6B, and the rate is not a stable constant**:
  one measured collection landed the citation reward 1 in 27, the next
  1 in 112 — CP-21's own conclusion is that the ~1/24 heuristic does not
  hold. A batch too small to contain a nonzero reward trains on zeros;
  `train.py` prints one `reward=` line per body so you see it before the
  GPU does — there is no aggregate line (F-27), count with
  `grep -o 'reward=[0-9.]*' | sort | uniq -c`.
- **`--episodes N` means N attempts** (Polar's `num_samples`), not N
  collected episodes; rejected traces are consumed attempts, quarantined
  with findings, never auto-retried.
- **The recompute-vs-captured logprob delta sits near 0.008 mean** (tail
  to ~0.21 per position) on the measured estate — three independent
  instruments agree. That is the platform's capture floor, not an error;
  `train.py` prints it against the shipped constants. The floor is
  composition-dependent: CP-26 measured 0.0137 mean on byte-identical
  weights with a different batch mix, so read the printout as an order-
  of-magnitude check, not a threshold — same order as 0.008 with a
  sub-percent share of positions over 0.21 is healthy; a 10× mean or
  positions-over-0.21 in the tens of percent means the snapshot is not
  the engine's weights (re-check the revision, F-23).
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
card text resolved at build time; every column is the triple, a
`render_task_request` argument, or `sandbox_image` (the row's
reproducibility pin, asserted against `runtime.image` at collect), so a
row submits with no translation beyond that check. `train.py` trains
only on `split == "train"` rows — the label is carried, the trainer
enforces it (that's this script, not the server). Rebuild:
`bash rebuild_taskbank.sh` — note it runs the library repo's corpus
pipeline from the LIBRARY repo's own venv, which the trainer install
does not create (F-38); you rarely need it — the shipped parquet IS the
bank.

## slime instead of verl

The first measured loop trained with slime v0.3.0 — it needs its own
container image (`slimerl/slime:v0.3.0`, 24.4 GB; the image's Megatron,
not Polar's documented pin) and lives in `../slime_bridge/cp17_loop/`
with its own run book. This project uses verl because the trainer is a
host venv rather than an image; the traces are the same either way.

## What can still surprise you

Honest list, post-stranger-test. CP-26 ran this document cold on the
reference estate and logged every stumble as `FINDINGS.md` F-14–F-39
(the fixes are folded in above; the rows record what a stranger hits
when a fix is only words). Still live, unfixed by any document:

- GPU topology drifts between sessions — discover free GPUs at run time
  (`nvidia-smi`) and pass `CUDA_VISIBLE_DEVICES` yourself (F-34);
- the artifacts dir must be visible to the grading host (sync it if you
  collect and train on different machines) — stated here, enforced
  nowhere;
- `python -m gsj_rollout.cli` exits 0 doing nothing — use the
  `gsj-rollout` console script (library wishlist row 19, the CP-21
  entry; measured again at CP-26: exit 0, zero bytes of output);
- a card edit lands as quarantines until the estate re-pins (wishlist
  row 19, the CP-24 entry — the register records two rows under that
  number, its own numbering slip);
- a wrong MCP secret or a `/v1`-suffixed `serving_base_url` or a
  gateway port/`public_url` mismatch all pass `load_config` and fail
  only at run time, as failed tool calls / a bare engine 404 / a
  connection refused on a URL nothing listens on (library wishlist 21 —
  the schema could catch the last two).
