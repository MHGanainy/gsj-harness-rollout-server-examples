#!/usr/bin/env bash
# sync_engine_local.sh — the restart sync, run ON the serving host (CP-69).
#
# The estate's `estate/serving/serve-updated.sh` is workstation-side: it
# drives the box over ssh via GSJ_VLLM_SSH_HOST and dies with a
# misleading DNS error when run on the box itself (F-29). A loop that
# trains where it serves needs the same two remote bodies executed
# LOCALLY — this script is exactly serve-updated.sh's REMOTE_STOP +
# REMOTE_START, nothing else. The caller (train_loop.py) owns the
# /v1/models wait and the before/after probe; this script only swaps the
# process.
#
# Same discipline as serve-updated.sh: the WEIGHTS change, the identity
# the wire speaks does not (--served-model-name stays the pinned id), and
# the four comparability legs (template, genconfig, max-model-len, tool
# parsing) ride unchanged from ~/<RDIR>/'s serve-time copies.
#
# Usage: sync_engine_local.sh <hf-checkpoint-dir>
# Env (defaults = the H200 estate's): GSJ_VLLM_REMOTE_DIR (gsj-vllm),
#   GSJ_VLLM_PORT (8000), GSJ_VLLM_GPU (3), GSJ_VLLM_GPU_FRAC (0.30).
#   The served name comes from ~/<RDIR>/model.env (serve.sh put it there).
set -euo pipefail

CKPT="${1:?usage: sync_engine_local.sh <HF checkpoint dir>}"
RDIR="${GSJ_VLLM_REMOTE_DIR:-gsj-vllm}"
PORT="${GSJ_VLLM_PORT:-8000}"
GPU="${GSJ_VLLM_GPU:-3}"
GPU_FRAC="${GSJ_VLLM_GPU_FRAC:-0.30}"

cd ~/"$RDIR"
[ -f model.env ] || { echo "ERROR: ~/$RDIR/model.env missing — run the estate's serve.sh once first (it ships the model pin + template)"; exit 1; }
source model.env
[ -f "$CKPT/config.json" ] || { echo "ERROR: $CKPT is not an HF checkpoint dir"; exit 1; }

# STOP (serve-updated.sh's REMOTE_STOP, verbatim semantics): pidfile kill,
# 60s grace, then SIGKILL. The A-13 drain point — the caller guarantees no
# collection is in flight before invoking this script.
if [ -f run/vllm.pid ] && kill -0 "$(cat run/vllm.pid)" 2>/dev/null; then
  kill "$(cat run/vllm.pid)"
  for _ in $(seq 1 60); do
    kill -0 "$(cat run/vllm.pid)" 2>/dev/null || break
    sleep 1
  done
  kill -9 "$(cat run/vllm.pid)" 2>/dev/null || true
  echo "engine stopped (was pid $(cat run/vllm.pid))"
else
  echo "no live engine"
fi
rm -f run/vllm.pid

# START (REMOTE_START, verbatim semantics): the updated checkpoint under
# the SAME served name and the same four legs.
CUDA_VISIBLE_DEVICES="$GPU" VLLM_LOGGING_LEVEL=DEBUG \
VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0 \
nohup ./venv/bin/vllm serve "$CKPT" \
  --served-model-name "$GSJ_MODEL_ID" \
  --host 127.0.0.1 --port "$PORT" \
  --max-model-len 32768 \
  --gpu-memory-utilization "$GPU_FRAC" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --chat-template "$HOME/$RDIR/qwen3_training.jinja" \
  --generation-config "$HOME/$RDIR/genconfig" \
  --enable-log-requests \
  --enforce-eager \
  > run/vllm.log 2>&1 &
echo $! > run/vllm.pid
echo "vllm started on updated weights (pid $(cat run/vllm.pid), gpu $GPU, ckpt $CKPT, served as $GSJ_MODEL_ID)"
