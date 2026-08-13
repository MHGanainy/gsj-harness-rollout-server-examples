#!/usr/bin/env bash
# taskbank.parquet is COMMITTED here (12 rows, train 9 / eval 3 — the
# CP-24 build; its sha256 is recorded in the library repo's
# corpus/staging/corpus.lock.json `taskbank` block). The generator is the
# library repo's corpus pipeline (ADR-0022); this rebuilds and re-copies,
# then prints the lock's sha beside the file's so drift is visible.
set -euo pipefail
cd "$(dirname "$0")"
LIB="${1:-../../gsj-harness-rollout-server}"

"$LIB/.venv/bin/python" "$LIB/corpus/ingest_corpus.py" taskbank --corpus "$LIB/corpus/staging"
cp "$LIB/corpus/staging/taskbank.parquet" taskbank.parquet

echo "lock sha256: $("$LIB/.venv/bin/python" -c \
  "import json; print(json.load(open('$LIB/corpus/staging/corpus.lock.json'))['taskbank']['sha256'])")"
if command -v sha256sum >/dev/null; then SHA=(sha256sum); else SHA=(shasum -a 256); fi
echo "file sha256: $("${SHA[@]}" taskbank.parquet | cut -d' ' -f1)"
