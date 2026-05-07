#!/bin/bash
set -euo pipefail

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=${1:-"8"}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

source "$(dirname "${BASH_SOURCE[0]}")/common_data.sh"
build_pile_dataset_args

# 512 * 1k * 60k = 30b tokens.
TRAIN_ITERS=${2:-"60000"}
MICRO_BATCH_SIZE=${3:-"32"}
NUM_EXPERTS=${4:-"8"}
GRANILARITY=${5:-"1"}
PROJECT_NAME=${6:-"moe-830m-baseline-8e1"}
SAVE_INTERVAL=${12:-"8000"}
SAVE_RETAIN_INTERVAL=${13:-"16000"}
EXIT_DURATION_MIN=${14:-""}  # wall-clock time budget in minutes; on first iter past the budget, save (if not just saved) and exit

CHECKPOINT_PATH="./new_logs/$PROJECT_NAME"
mkdir -p "$CHECKPOINT_PATH"


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 1024
    --max-position-embeddings 1024
    --num-layers 48
    --hidden-size 1024
    --ffn-hidden-size $((1024 * 4))
    --num-attention-heads 16
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 4
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
    --use-flash-attn
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

MOE_ARGS=(
    --num-experts $NUM_EXPERTS
    --moe-router-topk 1
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
    --moe-router-pre-softmax
    --moe-grouped-gemm
    # --moe-layer-recompute
    # --moe-granularity $GRANILARITY
)

DATA_ARGS=(
    --vocab-file "$VOCAB_FILE"
    --merge-file "$MERGE_FILE"
    --make-vocab-size-divisible-by 1024
    --data-path "${PILE_DATASET[@]}"
    --split 969,30,1
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size 512
    --lr 5e-4
    --train-iters $TRAIN_ITERS
    --lr-decay-style cosine
    --min-lr 5e-5
    --lr-warmup-fraction 0.01
    --clip-grad 1.0
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 10
    --log-throughput
    --save-interval $SAVE_INTERVAL
    --save-retain-interval $SAVE_RETAIN_INTERVAL
    --eval-interval 1000
    --eval-iters 100
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
    --ckpt-format torch
    --auto-detect-ckpt-format
)

if [ -n "${WANDB_API_KEY:-}" ]; then
    LOGGING_ARGS+=(
        --wandb-project "UniPool"
        --wandb-exp-name $PROJECT_NAME
    )
fi

if [ -n "$EXIT_DURATION_MIN" ]; then
    LOGGING_ARGS+=(--exit-duration-in-mins $EXIT_DURATION_MIN)
fi


torchrun "${DISTRIBUTED_ARGS[@]}" pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" 2>&1 | tee -a "$CHECKPOINT_PATH/train.log"
