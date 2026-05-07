#!/bin/bash
set -euo pipefail
####
# UniPool 650m (36 layers): All MoE layers share one global expert pool with per-layer routers.
# Dense-matched setting: top-1 routing, same ffn_hidden_size.
####
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
SAVE_INTERVAL=${10:-"2500"}
SAVE_RETAIN_INTERVAL=${11:-"15000"}
MICRO_BATCH_SIZE=${3:-"64"}
NUM_EXPERTS=${4:-"288"}
NUM_LAYERS=${12:-"36"}
TOP_K=${13:-"1"}
NORM_ROUTING=${5:-"1"}  # set to "1" to enable NormRouter, empty for default softmax
LAYER_AUX=${6:-""}      # set to coeff for per-layer aux loss, empty to disable
POOL_AUX=${7:-"2e-2"}   # set to coeff to enable pool-level aux loss, empty to disable
POOL_SIZE=${8:-""}       # set to N (e.g. "12") for grouped UniPool: every N layers share one expert pool, empty for global
EP_SIZE=${14:-"1"}       # expert-model-parallel size (must divide NUM_EXPERTS)

ROUTER_TAG=""
if [ -n "$NORM_ROUTING" ]; then
    ROUTER_TAG="-norm"
fi
LAYER_AUX_TAG=""
if [ -n "$LAYER_AUX" ]; then
    LAYER_AUX_TAG="-layeraux${LAYER_AUX}"
fi
POOL_AUX_TAG=""
if [ -n "$POOL_AUX" ]; then
    POOL_AUX_TAG="-poolaux${POOL_AUX}"
fi
POOL_SIZE_TAG=""
if [ -n "$POOL_SIZE" ]; then
    POOL_SIZE_TAG="-grp${POOL_SIZE}"
fi
LAYERS_TAG=""
if [ "$NUM_LAYERS" != "36" ]; then
    LAYERS_TAG="-L${NUM_LAYERS}"
fi
TOPK_TAG=""
if [ "$TOP_K" != "1" ]; then
    TOPK_TAG="-top${TOP_K}"
fi
EP_TAG=""
if [ "$EP_SIZE" != "1" ]; then
    EP_TAG="-ep${EP_SIZE}"
fi
PROJECT_NAME=${9:-"unipool-650m-${NUM_EXPERTS}e${LAYERS_TAG}${TOPK_TAG}${EP_TAG}-mb${MICRO_BATCH_SIZE}${ROUTER_TAG}${LAYER_AUX_TAG}${POOL_AUX_TAG}${POOL_SIZE_TAG}-cc"}

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
    --num-layers $NUM_LAYERS
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
    --moe-router-topk $TOP_K
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff ${LAYER_AUX:-0}
    --moe-token-dispatcher-type alltoall
    --moe-grouped-gemm
    --moe-expert-pool-mode hyper
)

if [ -n "$NORM_ROUTING" ]; then
    MOE_ARGS+=(--moe-norm-routing)
else
    MOE_ARGS+=(--moe-router-pre-softmax)
fi

if [ -n "$POOL_AUX" ]; then
    MOE_ARGS+=(--moe-pool-aux-loss-coeff $POOL_AUX)
fi

if [ -n "$POOL_SIZE" ]; then
    MOE_ARGS+=(--moe-expert-pool-size $POOL_SIZE)
fi

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
    --expert-model-parallel-size $EP_SIZE
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


torchrun "${DISTRIBUTED_ARGS[@]}" pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" 2>&1 | tee -a "$CHECKPOINT_PATH/train.log"
