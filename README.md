# UniPool

UniPool is a research code release for shared expert-pool Mixture-of-Experts
(MoE) training on top of Megatron-LM/Megatron Core. It adds a mode where MoE
layers keep separate routers while sharing a global or grouped expert pool.

This repository is a derivative of NVIDIA Megatron-LM. The original upstream
README is preserved as [README_MEGATRON.md](README_MEGATRON.md); this README
documents the UniPool-specific entry points and reproduction scripts.

## Scope

The supported UniPool surface is the expert-pool implementation and the
training scripts under `scripts/train_llama_*_moe_UniPool.sh`. Some
additional experimental MoE/MoD files from the research fork are retained for
compatibility with the Megatron module graph, but they are not required for the
UniPool experiments documented here.

## What Changed

UniPool adds a generalized expert-pool mode for MoE layers. The paper-facing
configuration uses:

- `--moe-expert-pool-mode hyper`: each MoE layer has its own router while
  sharing a global or grouped expert pool.
- `--moe-expert-pool-size`: optional group size. Empty/default means one global
  pool across all MoE layers; a positive value groups adjacent MoE layers.
- `--moe-pool-aux-loss-coeff`: pool-level load balancing across all layers that
  share a pool.
- `--moe-norm-routing`: normalized routing used by the default UniPool scripts.

Core implementation files:

- `megatron/core/transformer/moe/moe_layer.py`
- `megatron/core/transformer/moe/moe_utils.py`
- `megatron/core/transformer/moe/router.py`
- `megatron/core/transformer/transformer_block.py`
- `megatron/core/transformer/transformer_config.py`
- `megatron/training/arguments.py`
- `megatron/training/checkpointing.py`

## Environment

Use the same dependency stack as Megatron-LM/Megatron Core. For GPU training,
the recommended route is an NVIDIA PyTorch/NGC container with PyTorch, CUDA,
NCCL, Transformer Engine, and Triton installed. See
[README_MEGATRON.md](README_MEGATRON.md) for the full upstream installation
notes.

From the repository root:

```bash
pip install --no-build-isolation -e ".[mlm,dev]"
```

The distribution package name is `unipool-megatron` to avoid confusion with the
upstream `megatron-core` package. The Python import path remains `megatron`
because this is a Megatron fork.

The scripts also use these runtime variables when present:

- `MASTER_ADDR`, default `localhost`
- `MASTER_PORT`, default `6000`
- `SLURM_NNODES`, mapped to `NNODES`
- `RANK`, mapped to `NODE_RANK`
- `WANDB_API_KEY`, enables W&B logging when set

## Data Preprocessing

The Pile dataset is not bundled with this code release. Download the raw shards
from the public Hugging Face mirror
[`monology/pile-uncopyrighted`](https://huggingface.co/datasets/monology/pile-uncopyrighted)
and place them under `../pile/` as `00.jsonl ... 29.jsonl` before running
preprocessing. Any equivalent local copy of The Pile in JSONL form works.

Training uses Megatron indexed dataset prefixes, not raw `.jsonl` files. A
prefix such as `../pile_gpt_test/00_text_document` must correspond to
`../pile_gpt_test/00_text_document.bin` and
`../pile_gpt_test/00_text_document.idx`.

Prepare Pile shards with the standard Megatron preprocessing command, also
provided as `data_preprocessing.sh`:

```bash
for i in $(seq -w 0 29); do
  python tools/preprocess_data.py \
    --input ../pile/${i}.jsonl \
    --output-prefix ../pile_gpt_test/${i} \
    --vocab-file ./gpt2-vocab.json \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file ./gpt2-merges.txt \
    --append-eod \
    --workers 32
done
```

The training scripts source `scripts/common_data.sh`. Defaults:

- `DATA_ROOT=../pile_gpt_test`
- `DATA_START=0`
- `DATA_END=29`
- `VOCAB_FILE=./gpt2-vocab.json`
- `MERGE_FILE=./gpt2-merges.txt`

Override these if your data is elsewhere:

```bash
DATA_ROOT=/path/to/pile_gpt_test \
VOCAB_FILE=/path/to/gpt2-vocab.json \
MERGE_FILE=/path/to/gpt2-merges.txt \
bash scripts/train_llama_182m_moe_UniPool.sh
```

## Training Scripts

Run scripts from the repository root. Each script also works when launched from
`scripts/` because it changes back to the repository root internally.

UniPool shared-pool runs:

```bash
bash scripts/train_llama_182m_moe_UniPool.sh
bash scripts/train_llama_469m_moe_UniPool.sh
bash scripts/train_llama_650m_moe_UniPool.sh
bash scripts/train_llama_830m_moe_UniPool.sh
bash scripts/train_llama_978m_moe_UniPool.sh
```

Shared-pool script signature:

```text
bash scripts/train_llama_<size>_moe_UniPool.sh \
  [gpus_per_node] [train_iters] [micro_batch_size] [num_experts] \
  [norm_routing] [layer_aux_coeff] [pool_aux_coeff] [pool_size] \
  [project_name] [save_interval] [save_retain_interval] [num_layers] [top_k]
```

For `650m` and `830m`, additional expert-parallel and wall-clock arguments are
available in the script headers:

- `EP_SIZE`: expert model parallel size.
- `EXIT_DURATION_MIN`: save and exit after the time budget.

Dense and vanilla MoE baselines are also included:

```bash
bash scripts/train_llama_182m_dense.sh
bash scripts/train_llama_182m_moe.sh
bash scripts/train_llama_469m_dense.sh
bash scripts/train_llama_469m_moe.sh
bash scripts/train_llama_650m_dense.sh
bash scripts/train_llama_650m_moe.sh
bash scripts/train_llama_830m_dense.sh
bash scripts/train_llama_830m_moe.sh
bash scripts/train_llama_978m_dense.sh
bash scripts/train_llama_978m_moe.sh
```

By default, the scripts use sequence length 1024, global batch size 512, and
`train_iters=60000`, corresponding to roughly 30B training tokens.

Outputs are written to `new_logs/<project_name>` for MoE/UniPool runs and
`logs/<project_name>` for the older dense baselines. Checkpoints are passed to
both `--save` and `--load` so training can resume from the same directory.

## Quick Checks

These checks do not require the full GPU training stack:

```bash
python tests/test_unipool_release_checks.py
find scripts -maxdepth 1 -name "*.sh" -exec bash -n {} \;
PYTHONPYCACHEPREFIX=/tmp/unipool_pycache python -m py_compile \
  megatron/core/transformer/moe/moe_layer.py \
  megatron/core/transformer/moe/moe_utils.py \
  megatron/core/transformer/moe/router.py \
  megatron/core/transformer/transformer_block.py \
  megatron/training/arguments.py \
  megatron/training/checkpointing.py
```

Full pytest runs require the Megatron GPU dependency stack, including Triton
and Transformer Engine. Note that `tests/unit_tests/` and
`tests/functional_tests/` are inherited from the upstream Megatron-LM project;
many of them reference NVIDIA-internal data paths (e.g. `/opt/data/...`,
`/workspace/data/...`) and are not expected to pass on standard installations.
Only `tests/test_unipool_release_checks.py` is part of the supported UniPool
surface.

## License

UniPool modifications are released under the license terms in [LICENSE](LICENSE).
This repository includes derivative work from Megatron-LM and other upstream
projects; upstream notices are retained in source files and summarized in
[NOTICE](NOTICE).
