#!/bin/bash
# Tokenize 30 Pile shards (00..29) with the GPT-2 BPE tokenizer into
# Megatron indexed dataset prefixes under ../pile_gpt_test/.
#
# Prerequisites:
#   - Raw Pile shards at ../pile/{00..29}.jsonl (download from
#     https://huggingface.co/datasets/monology/pile-uncopyrighted or any
#     equivalent JSONL mirror of The Pile).
#   - ./gpt2-vocab.json and ./gpt2-merges.txt at the repo root (shipped).
#
# Output:
#   ../pile_gpt_test/{00..29}_text_document.{bin,idx}
#
# Adjust --workers to match your CPU count.
set -euo pipefail

INPUT_DIR=${INPUT_DIR:-../pile}
OUTPUT_DIR=${OUTPUT_DIR:-../pile_gpt_test}
VOCAB_FILE=${VOCAB_FILE:-./gpt2-vocab.json}
MERGE_FILE=${MERGE_FILE:-./gpt2-merges.txt}
WORKERS=${WORKERS:-32}

mkdir -p "${OUTPUT_DIR}"

for i in $(seq -w 0 29); do
    python tools/preprocess_data.py \
        --input "${INPUT_DIR}/${i}.jsonl" \
        --output-prefix "${OUTPUT_DIR}/${i}" \
        --vocab-file "${VOCAB_FILE}" \
        --tokenizer-type GPT2BPETokenizer \
        --merge-file "${MERGE_FILE}" \
        --append-eod \
        --workers "${WORKERS}"
done
