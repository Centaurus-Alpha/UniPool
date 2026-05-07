#!/bin/bash

echo "=== bf16 Training Diagnosis ==="

python3 -c "
import torch
import transformer_engine.pytorch as te

print('Testing TransformerEngine RMSNorm with explicit params_dtype...')

# Test 1: With explicit bf16 params_dtype (like Megatron does)
try:
    rms_bf16 = te.RMSNorm(
        hidden_size=768,
        eps=1e-5,
        params_dtype=torch.bfloat16,
        device='cuda'
    )
    x = torch.randn(16, 1024, 768, dtype=torch.bfloat16, device='cuda')
    out = rms_bf16(x)
    print(f'  Test 1 (explicit bf16 dtype): PASSED')
    print(f'    weight dtype: {rms_bf16.weight.dtype}')
    print(f'    output max: {out.max().item():.4f}, std: {out.std().item():.4f}')
except Exception as e:
    print(f'  Test 1 (explicit bf16 dtype): FAILED - {e}')

# Test 2: Without params_dtype (default behavior)
try:
    rms_default = te.RMSNorm(
        hidden_size=768,
        eps=1e-5,
        device='cuda'
    )
    # Convert weight to bf16 manually
    rms_default.weight.data = rms_default.weight.data.to(torch.bfloat16)
    x = torch.randn(16, 1024, 768, dtype=torch.bfloat16, device='cuda')
    out = rms_default(x)
    print(f'  Test 2 (manual bf16 conversion): PASSED')
except Exception as e:
    print(f'  Test 2 (manual bf16 conversion): FAILED - {e}')

# Test 3: With autocast
try:
    rms_fp32 = te.RMSNorm(hidden_size=768, eps=1e-5, device='cuda')
    x = torch.randn(16, 1024, 768, dtype=torch.bfloat16, device='cuda')
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = rms_fp32(x)
    print(f'  Test 3 (with autocast): PASSED')
except Exception as e:
    print(f'  Test 3 (with autocast): FAILED - {e}')

print()
print('Testing MoE Router softmax precision...')

# Test 4: Softmax precision
try:
    logits = torch.randn(16384, 8, dtype=torch.bfloat16, device='cuda') * 10  # large values
    # Method 1: Direct bf16 softmax
    probs_bf16 = torch.softmax(logits, dim=-1)
    # Method 2: fp32 intermediate (like Megatron does)
    probs_fp32 = torch.softmax(logits, dim=-1, dtype=torch.float32).to(torch.bfloat16)

    diff = (probs_bf16 - probs_fp32).abs().max().item()
    print(f'  Softmax precision diff (bf16 vs fp32->bf16): {diff:.6f}')
    print(f'  bf16 softmax sum: {probs_bf16.sum(dim=-1).mean().item():.6f}')
    print(f'  fp32 softmax sum: {probs_fp32.sum(dim=-1).mean().item():.6f}')
except Exception as e:
    print(f'  Softmax test FAILED: {e}')

print()
print('Testing gradient flow with bf16...')

# Test 5: Gradient flow
try:
    linear = torch.nn.Linear(768, 768, dtype=torch.bfloat16, device='cuda')
    x = torch.randn(16, 1024, 768, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    y = linear(x)
    loss = y.sum()
    loss.backward()
    grad_norm = x.grad.norm().item()
    print(f'  Gradient norm: {grad_norm:.4f}')
    if grad_norm > 1e6 or torch.isnan(torch.tensor(grad_norm)):
        print(f'  WARNING: Gradient may be unstable!')
    else:
        print(f'  Gradient flow: PASSED')
except Exception as e:
    print(f'  Gradient test FAILED: {e}')

print()
print('Testing large matmul stability...')

# Test 6: Large matmul (simulating expert computation)
try:
    # Simulate expert MLP: hidden -> 4*hidden -> hidden
    hidden = 768
    ffn_hidden = 768 * 4
    x = torch.randn(16384, hidden, dtype=torch.bfloat16, device='cuda')
    w1 = torch.randn(hidden, ffn_hidden, dtype=torch.bfloat16, device='cuda') * 0.01
    w2 = torch.randn(ffn_hidden, hidden, dtype=torch.bfloat16, device='cuda') * 0.01

    h = torch.matmul(x, w1)
    h = torch.nn.functional.silu(h)
    out = torch.matmul(h, w2)

    print(f'  MLP output max: {out.max().item():.4f}')
    print(f'  MLP output min: {out.min().item():.4f}')
    print(f'  MLP output has NaN: {torch.isnan(out).any().item()}')
    print(f'  MLP output has Inf: {torch.isinf(out).any().item()}')

    if torch.isnan(out).any() or torch.isinf(out).any():
        print(f'  WARNING: Numerical instability detected!')
    else:
        print(f'  Large matmul: PASSED')
except Exception as e:
    print(f'  Large matmul test FAILED: {e}')

print()
print('=== Recommendation ===')
print('If Test 1 passed, bf16 with explicit params_dtype should work.')
print('Make sure your training uses --bf16 flag correctly.')
"
