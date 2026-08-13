#!/usr/bin/env bash
# The one command (ADR-0023, library repo): venv, the library wheel, the
# stated closure, then verl --no-deps at the pinned SHA — in that order.
# Two steps are genuinely two (the --no-deps git line cannot ride a
# requirements file or a pip extra, and PyPI refuses direct git URLs in
# published metadata); this script is where they become one command.
set -euo pipefail
cd "$(dirname "$0")"

VERL_SHA=1ae945592754cbeb1350cbe092fe6117070fd4c7   # == bridge.VERL_SHA

python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# The library, as a wheel — the LOCAL build, deliberately not the index:
# until the first upload lands (library wishlist 17) the PyPI name is
# unclaimed, and an index-first install would trust whoever claims it.
WHEEL=(../../gsj-harness-rollout-server/dist/gsj_harness_rollout_server-*.whl)
if [ -e "${WHEEL[0]}" ]; then
  ./.venv/bin/pip install "${WHEEL[0]}"
else
  echo "install.sh: no local wheel found — build it first:" >&2
  echo "  (cd ../../gsj-harness-rollout-server && python -m build --wheel)" >&2
  echo "or, once the package is published (library wishlist 17):" >&2
  echo "  ./.venv/bin/pip install 'gsj-harness-rollout-server>=0.1.0'" >&2
  exit 1
fi

./.venv/bin/pip install --no-deps \
  "verl @ git+https://github.com/volcengine/verl.git@${VERL_SHA}"

./.venv/bin/python -c "import verl.protocol, gsj_rollout.checks, pyarrow; print('install OK')"
