#!/bin/bash

echo "=== Package Versions ==="

python3 -c "
import sys

# PyTorch & CUDA
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA (PyTorch): {torch.version.cuda}')
    print(f'cuDNN: {torch.backends.cudnn.version()}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'CUDA capability: {torch.cuda.get_device_capability(0)}')
except Exception as e:
    print(f'PyTorch error: {e}')

# Apex
try:
    import apex
    from apex.normalization import FusedLayerNorm
    print(f'Apex: installed')
except ImportError as e:
    print(f'Apex: NOT INSTALLED - {e}')

# Flash Attention
try:
    import flash_attn
    print(f'Flash Attention: {flash_attn.__version__}')
except ImportError:
    print('Flash Attention: NOT INSTALLED')

# TransformerEngine
try:
    import transformer_engine as te
    print(f'TransformerEngine: {te.__version__}')
except ImportError:
    print('TransformerEngine: NOT INSTALLED')

# grouped_gemm
try:
    import grouped_gemm
    print(f'grouped_gemm: installed')
except ImportError:
    print('grouped_gemm: NOT INSTALLED')

# NumPy
try:
    import numpy as np
    print(f'NumPy: {np.__version__}')
except:
    pass

print()
print('=== bf16 Numerical Test ===')
try:
    x = torch.randn(100, 100, dtype=torch.bfloat16, device='cuda')
    y = torch.matmul(x, x.T)
    print(f'bf16 matmul test: PASSED (max={y.max().item():.4f}, min={y.min().item():.4f})')
except Exception as e:
    print(f'bf16 matmul test: FAILED - {e}')

print()
print('=== RMSNorm bf16 Test ===')
try:
    import transformer_engine.pytorch as te_pytorch
    rms = te_pytorch.RMSNorm(768).cuda()
    x = torch.randn(16, 1024, 768, dtype=torch.bfloat16, device='cuda')
    out = rms(x)
    print(f'TE RMSNorm bf16 test: PASSED (max={out.max().item():.4f}, std={out.std().item():.4f})')
except Exception as e:
    print(f'TE RMSNorm bf16 test: FAILED - {e}')

print()
print('=== Flash Attention bf16 Test ===')
try:
    from flash_attn import flash_attn_func
    q = torch.randn(2, 1024, 12, 64, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(2, 1024, 12, 64, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(2, 1024, 12, 64, dtype=torch.bfloat16, device='cuda')
    out = flash_attn_func(q, k, v, causal=True)
    print(f'Flash Attention bf16 test: PASSED (max={out.max().item():.4f}, std={out.std().item():.4f})')
except Exception as e:
    print(f'Flash Attention bf16 test: FAILED - {e}')
"

echo ""
echo "=== NVIDIA Driver & CUDA Toolkit ==="
nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
nvcc --version 2>/dev/null | grep release || echo "nvcc not available"

echo ""
echo "=== Docker Image Info ==="
cat /etc/os-release 2>/dev/null | grep -E "^NAME=|^VERSION=" || echo "Not in container or no os-release"
