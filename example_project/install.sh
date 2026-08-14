#!/usr/bin/env bash
# The one command (ADR-0023, library repo): venv, the library (PyPI, or a
# sibling-checkout wheel when one exists — library CP-29), the stated
# closure, then verl --no-deps at the pinned SHA — in that order.
# Two steps are genuinely two (the --no-deps git line cannot ride a
# requirements file or a pip extra, and PyPI refuses direct git URLs in
# published metadata); this script is where they become one command.
#
# Layout note (CP-26 F-15, revised at library CP-29): the library now
# installs from PyPI (`gsj-harness-rollout-server`, published v0.1.0), so
# the trainer role needs NO library checkout. A SIBLING checkout —
#     <parent>/gsj-harness-rollout-server/          (the library)
#     <parent>/gsj-harness-rollout-server-examples/ (this repo)
# — with a built wheel in its dist/ takes precedence when present, so a
# developer can run against unreleased library changes; the server role
# still needs that checkout for vendor/polar/ either way. Network note:
# the verl step clones github.com; the torch step downloads from PyPI or
# download.pytorch.org.
set -euo pipefail
cd "$(dirname "$0")"

VERL_SHA=1ae945592754cbeb1350cbe092fe6117070fd4c7   # == bridge.VERL_SHA

python3.12 -m venv .venv

# The library: PyPI by default, published at library CP-29 (ADR-0023's
# local-wheel-first ordering was justified by the name being UNCLAIMED —
# an index-first install would have trusted whoever claimed it; the name
# is now ours, so the index is the default). A sibling-checkout wheel,
# when one has been built, takes precedence: a developer running against
# unreleased library changes keeps the old path. The CP-26 F-31 property
# (fail before the multi-GB requirements download, not after) is kept by
# installing the library FIRST — it is the small step, and the likeliest
# to fail on a box with no network or a broken sibling build.
WHEEL=(../../gsj-harness-rollout-server/dist/gsj_harness_rollout_server-*.whl)
if [ -e "${WHEEL[0]}" ]; then
  echo "install.sh: using the sibling-checkout wheel ${WHEEL[0]} (developer path;" >&2
  echo "delete the library's dist/ to install the published PyPI wheel instead)" >&2
  ./.venv/bin/pip install "${WHEEL[0]}"
else
  ./.venv/bin/pip install 'gsj-harness-rollout-server>=0.1.0'
fi

./.venv/bin/pip install -r requirements.txt

./.venv/bin/pip install --no-deps \
  "verl @ git+https://github.com/volcengine/verl.git@${VERL_SHA}"

./.venv/bin/python -c "import verl.protocol, gsj_rollout.checks, pyarrow; print('install OK')"

# CP-26 F-16: the default PyPI torch is a cu13x build; on a CUDA-12.x
# driver it imports fine and fails only at the first CUDA call ("driver
# too old"). Surface that NOW, with the working cure — note the +cu126
# local tag is REQUIRED (a bare ==2.13.0 re-pin is satisfied by the
# already-installed build and silently no-ops).
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! ./.venv/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "" >&2
    echo "install.sh WARNING: a GPU is present but torch.cuda is NOT usable" >&2
    echo "(cu13x wheel vs this driver, most likely). The desk halves" >&2
    echo "(--collect-only, --dry-run) run anyway; before --snapshot, re-pin:" >&2
    echo "  ./.venv/bin/pip install 'torch==2.13.0+cu126' --index-url https://download.pytorch.org/whl/cu126" >&2
    echo "(match the cuXXX suffix to YOUR driver — nvidia-smi's CUDA Version)" >&2
  fi
fi
