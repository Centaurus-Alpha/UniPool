# Contributing to UniPool

UniPool is maintained as a research code release. Contributions should stay
focused on the shared expert-pool implementation, reproduction scripts,
documentation, and minimal tests needed to keep the release usable.

## Development Setup

Install the project from the repository root:

```bash
pip install --no-build-isolation -e ".[mlm,dev]"
```

GPU training and full Megatron tests require the Megatron dependency stack,
including CUDA, Triton, Transformer Engine, and NCCL.

## Lightweight Checks

Run these before opening a pull request:

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

## Pull Requests

- Keep changes scoped to UniPool behavior or release maintenance.
- Preserve upstream Megatron copyright and license notices.
- Include a short explanation of the behavior changed and the checks run.
- Do not commit datasets, checkpoints, logs, W&B runs, API keys, or machine-local
  paths.

## Relationship to Megatron-LM

This repository is a derivative research fork of Megatron-LM. For general
Megatron Core development guidance, see the upstream documentation in
[README_MEGATRON.md](README_MEGATRON.md) and the NVIDIA Megatron-LM project.
