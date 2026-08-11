#!/usr/bin/env bash
# CP-17: one optimizer step in real slime, inside `slimerl/slime:v0.3.0`.
#
# Expects (host paths under /host):
#   /host/cp17-slime          THUDM slime v0.3.0 (bf14dc21, == the image's
#                             own /root/slime) + the router-tokens patch
#
# Megatron comes from the IMAGE (/root/Megatron-LM @ 1dcf0dafa, the
# Dockerfile's MEGATRON_COMMIT), NOT from a separate checkout. Polar's
# launch_e2e.sh defaults MEGATRON_REF=26.04-alpha.rc1 with the note that
# slime v0.3.0 needs `megatron.training.tokenizer`; that tag does not carry
# the module (`megatron/training/` exists, `tokenizer/` does not) and the
# conversion dies on ModuleNotFoundError. The image's pin does carry it, and
# is the pair the image was built and tested with. See FINDINGS F-06.
#   /host/cp17-examples       this repo (slime_bridge/ on PYTHONPATH, flat)
#   /host/gsj-harness-rollout-server
#                             the library checkout (the vendored reward
#                             post-processor file; the wheel lives in cp17/)
#   /host/cp17                scratch: wheel, collected/, ckpt/, logs
#   /host/cp04prime/artifacts pi_harness artifacts_dir (the deliverables the
#                             citation grader reads)
#   /host/.cache/huggingface  HF cache holding the pinned Qwen3-0.6B snapshot
#
# Two subcommands: `convert` (HF -> torch_dist, once) and `train` (the one
# step + save). Weight sync back to the estate (torch_dist -> HF -> vLLM
# restart) is the host's job — see README.
set -euo pipefail

MODE="${1:?convert|train}"
SLIME=/host/cp17-slime
MEGATRON=/root/Megatron-LM     # the image's own, matched to slime v0.3.0
EXAMPLES=/host/cp17-examples
SERVER=/host/gsj-harness-rollout-server
CP17=/host/cp17
ARTIFACTS=/host/cp04prime/artifacts
export HF_HOME=/host/.cache/huggingface
export HF_HUB_OFFLINE=1
# The pinned codec snapshot — the same revision serve.sh serves (model.env
# GSJ_MODEL_REVISION), resolved from the estate's own HF cache.
SNAP="$(ls -d ${HF_HOME}/hub/models--Qwen--Qwen3-0.6B/snapshots/*/ | head -1)"
[ -f "${SNAP}/config.json" ] || { echo "ERROR: no Qwen3-0.6B snapshot under ${HF_HOME}"; exit 1; }

export PYTHONPATH="${MEGATRON}:${SLIME}:${EXAMPLES}/slime_bridge"
export POLAR_REWARD_PP_FILE="${SERVER}/vendor/polar/src/slime_bridge/reward_post_process.py"
export CUDA_DEVICE_MAX_CONNECTIONS=1

pip install --quiet --no-index "$(ls ${CP17}/gsj_harness_rollout_server-*.whl)"

# shellcheck source=/dev/null
source "${SLIME}/scripts/models/qwen3-0.6B.sh"   # MODEL_ARGS for the dense 0.6B

REF_LOAD="${CP17}/ckpt/qwen3-0.6B_torch_dist"
SAVE_DIR="${CP17}/ckpt/cp17_save"

if [ "$MODE" = "convert" ]; then
    mkdir -p "$REF_LOAD"
    torchrun --nproc_per_node 1 "${SLIME}/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "$SNAP" \
        --save "$REF_LOAD" \
        --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
        --context-parallel-size 1 --expert-model-parallel-size 1 \
        --expert-tensor-parallel-size 1 \
        --no-gradient-accumulation-fusion
    exit 0
fi

mkdir -p "$SAVE_DIR" "${CP17}/logs"
printf '{"prompt": "cp17-unused"}\n' > "${CP17}/dummy.jsonl"

# `--start-rollout-id 0` is load-bearing, not decoration: `--load` points at
# a checkpoint carrying `latest_checkpointed_iteration.txt`, which slime reads
# as a RESUME (`start_rollout_id = loaded_rollout_id + 1` — actor.py:99), so
# `range(start_rollout_id, num_rollout)` = `range(1, 1)` = empty and the whole
# train loop no-ops while the Ray job still reports SUCCESS. Hit live, once.
# See FINDINGS F-07.
#
# One sample per collected session (chains_total==1 asserted upstream), so
# the returned count == the file count; slime's global batch derives from
# rollout_batch_size (1) x n_samples_per_prompt (N) / num_steps_per_rollout (1).
N_EPISODES="$(ls "${CP17}/collected"/*.json | wc -l)"
echo "CP-17: training on ${N_EPISODES} collected sessions"

ray start --head --num-gpus 1 --disable-usage-stats
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"POLAR_REWARD_PP_FILE\": \"${POLAR_REWARD_PP_FILE}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"HF_HOME\": \"/host/.cache/huggingface\",
    \"HF_HUB_OFFLINE\": \"1\",
    \"GSJ_COLLECTED_DIR\": \"${CP17}/collected\",
    \"GSJ_ARTIFACTS_ROOT\": \"${ARTIFACTS}\",
    \"GSJ_CUTOFF\": \"12\",
    \"GSJ_PAGE_COUNT\": \"18\",
    \"GSJ_MAX_TOKENS\": \"32768\",
    \"GSJ_LOOP_SUMMARY\": \"${CP17}/logs/rollout_summary.json\"
  }
}"

ray job submit --address=http://127.0.0.1:8265 \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME}/train.py" \
    --actor-num-nodes 1 --actor-num-gpus-per-node 1 \
    --debug-train-only \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$SNAP" \
    --ref-load "$REF_LOAD" \
    --load "$REF_LOAD" \
    --start-rollout-id 0 \
    --save "$SAVE_DIR" \
    --save-interval 1 \
    --rollout-function-path cp17_loop.rollout_fn.generate_rollout \
    --custom-reward-post-process-path cp17_loop.polar_pp.post_process_rewards \
    --prompt-data "${CP17}/dummy.jsonl" --input-key prompt \
    --num-rollout 1 --num-steps-per-rollout 1 \
    --rollout-batch-size 1 --n-samples-per-prompt "${N_EPISODES}" \
    --reward-key score \
    --advantage-estimator grpo \
    --disable-grpo-std-normalization \
    --use-tis \
    --eps-clip 0.2 --eps-clip-high 0.28 \
    --optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.98 \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
    --attention-backend auto \
    --no-gradient-accumulation-fusion \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size --max-tokens-per-gpu 32768 \
    --log-probs-chunk-size 256 \
    --distributed-timeout-minutes 30 \
    2>&1 | tee "${CP17}/logs/train.log"

# torch_dist -> HF for the estate's vLLM (the checkpoint-reload sync half);
# CLI per slime docs quick_start.md: input-dir is the iter_xxx dir, origin
# HF dir supplies tokenizer/config/generation_config verbatim.
ITER="$(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt")"
ITER_DIR="$(printf '%s/iter_%07d' "$SAVE_DIR" "$ITER")"
HF_OUT="${CP17}/ckpt/cp17_hf_updated"
python3 "${SLIME}/tools/convert_torch_dist_to_hf.py" \
    --input-dir "$ITER_DIR" \
    --output-dir "$HF_OUT" \
    --origin-hf-dir "$SNAP" \
    --vocab-size 151936 \
    --force \
    2>&1 | tee "${CP17}/logs/convert_to_hf.log"
echo "CP-17 train step + HF export done: ${HF_OUT} (iteration ${ITER})"
