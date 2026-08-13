#!/usr/bin/env bash
# The one command (ADR-0023, library repo): venv, the library wheel, the
# stated closure, then verl --no-deps at the pinned SHA — in that order.
# Two steps are genuinely two (the --no-deps git line cannot ride a
# requirements file or a pip extra, and PyPI refuses direct git URLs in
# published metadata); this script is where they become one command.
#
# Layout requirement (CP-26 F-15): the LIBRARY repo must be checked out
# as a SIBLING of this examples repo —
#     <parent>/gsj-harness-rollout-server/          (the library)
#     <parent>/gsj-harness-rollout-server-examples/ (this repo)
# — because the wheel is built from that checkout until the PyPI name
# is claimed (library wishlist 17). Network note: the verl step clones
# github.com; the torch step downloads from PyPI or download.pytorch.org.
set -euo pipefail
cd "$(dirname "$0")"

VERL_SHA=1ae945592754cbeb1350cbe092fe6117070fd4c7   # == bridge.VERL_SHA

python3.12 -m venv .venv

# The library, as a wheel — the LOCAL build, deliberately not the index:
# until the first upload lands (library wishlist 17) the PyPI name is
# unclaimed, and an index-first install would trust whoever claims it.
# The gate runs FIRST (CP-26 F-31): fail here, before the multi-GB
# requirements download, not after it.
WHEEL=(../../gsj-harness-rollout-server/dist/gsj_harness_rollout_server-*.whl)
if [ ! -e "${WHEEL[0]}" ]; then
  echo "install.sh: no local wheel found — build it first (CP-26 F-15: the" >&2
  echo "commands below work on a box with no system pip; they use THIS venv):" >&2
  echo "  $PWD/.venv/bin/pip install build" >&2
  echo "  (cd ../../gsj-harness-rollout-server && $PWD/.venv/bin/python -m build --wheel)" >&2
  echo "then re-run bash install.sh. Or, once the package is published" >&2
  echo "(library wishlist 17):" >&2
  echo "  ./.venv/bin/pip install 'gsj-harness-rollout-server>=0.1.0'" >&2
  exit 1
fi

./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install "${WHEEL[0]}"

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
