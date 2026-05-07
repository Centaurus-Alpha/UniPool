#!/usr/bin/env bash

# Shared path setup for UniPool training scripts.
# Scripts may be launched from the repository root or from scripts/.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT" || exit 1

# Expected output of tools/preprocess_data.py:
#   ${DATA_ROOT}/00_text_document.{bin,idx} ... ${DATA_ROOT}/29_text_document.{bin,idx}
DATA_ROOT=${DATA_ROOT:-../pile_gpt_test}
DATA_START=${DATA_START:-0}
DATA_END=${DATA_END:-29}

VOCAB_FILE=${VOCAB_FILE:-./gpt2-vocab.json}
MERGE_FILE=${MERGE_FILE:-./gpt2-merges.txt}

build_pile_dataset_args() {
    local start=${1:-$DATA_START}
    local end=${2:-$DATA_END}
    local shard

    PILE_DATASET=()
    for ((i = start; i <= end; i++)); do
        shard=$(printf "%02d" "$i")
        PILE_DATASET+=(1.0 "${DATA_ROOT%/}/${shard}_text_document")
    done
}
