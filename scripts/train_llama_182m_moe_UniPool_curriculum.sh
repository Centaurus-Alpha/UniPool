#!/bin/bash
set -euo pipefail
####
# UniPool 182m with the progressive curriculum.
# 12 layers share one global pool of M = 8 * L = 96 experts (top-1, NormRouter,
# pool aux loss 1e-2). The per-layer routers explore the whole pool until the
# lock, then every layer keeps its own K = M / L = 8 exclusively owned experts:
#   iter 1000  candidate partition frozen from validation routing counts
#   iter 1000-2000  cosine fade-out of each layer's off-partition scores
#   iter 2000  partition hard-locked for the rest of training
# After the lock each layer routes and computes over 8 experts only (compact
# dispatch + compact router + mask-aware overlapped grad reduce), so per-token
# compute and per-layer expert parameters match an 8-expert vanilla MoE layer.
#
# Usage: bash scripts/train_llama_182m_moe_UniPool_curriculum.sh [GPUS] [TRAIN_ITERS] [MICRO_BATCH] [PROJECT_NAME]
# The schedule is expressed as fractions of TRAIN_ITERS; --eval-interval must
# divide both boundaries (the defaults do for 60000 iterations).
####
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=${1:-"8"}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}

source "$(dirname "${BASH_SOURCE[0]}")/common_data.sh"
build_pile_dataset_args

# 512 * 1k * 60k = 30b tokens.
TRAIN_ITERS=${2:-"60000"}
MICRO_BATCH_SIZE=${3:-"64"}
SEED=${SEED:-"1234"}
POOL_AUX=${POOL_AUX:-"1e-2"}
SAVE_INTERVAL=${SAVE_INTERVAL:-"2500"}
SAVE_RETAIN_INTERVAL=${SAVE_RETAIN_INTERVAL:-"15000"}
EVAL_INTERVAL=${EVAL_INTERVAL:-"1000"}
EVAL_ITERS=${EVAL_ITERS:-"100"}

PROJECT_NAME=${4:-"unipool-182m-96e-norm-poolaux${POOL_AUX}-curriculum-seed${SEED}-cc"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"./new_logs/$PROJECT_NAME"}
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
    --num-layers 12
    --hidden-size 768
    --ffn-hidden-size $((768 * 4))
    --num-attention-heads 12
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
)

MOE_ARGS=(
    --num-experts 96
    --moe-router-topk 1
    --moe-norm-routing
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --moe-layer-recompute
    --moe-expert-pool-mode hyper
    --moe-pool-aux-loss-coeff $POOL_AUX
)

CURRICULUM_ARGS=(
    --moe-progressive-curriculum
    --moe-progressive-compact-dispatch
    --moe-progressive-compact-router
    --moe-progressive-overlap-grad-reduce
    --overlap-grad-reduce
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
    --seed $SEED
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
    --eval-interval $EVAL_INTERVAL
    --eval-iters $EVAL_ITERS
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


torchrun "${DISTRIBUTED_ARGS[@]}" pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${CURRICULUM_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" 2>&1 | tee -a "$CHECKPOINT_PATH/train.log"
