# Advanced FLOP Counting in VLLM

This document describes the comprehensive FLOP (Floating Point Operations) counting system in VLLM, including support for custom CUDA kernels.

## Overview

VLLM's FLOP counting system provides accurate measurement of computational workload by:

1. **Automatic registration** of all VLLM custom operations with PyTorch's FLOP counter
2. **Categorized breakdown** of operations by type (GEMM, attention, activations, etc.)
3. **Real-time monitoring** and analysis utilities

## Quick Start

### Basic FLOP Counting

```python
from vllm import LLM, SamplingParams
from vllm.profiler import FlopContextManager, format_flops

llm = LLM(model="facebook/opt-125m")
sampling_params = SamplingParams(temperature=0.8, max_tokens=50)

# Count FLOPs for generation
with FlopContextManager() as flop_counter:
    outputs = llm.generate(["Hello, world!"], sampling_params)

total_flops = flop_counter.get_total_flops()
print(f"Total FLOPs: {format_flops(total_flops)}")

# Get detailed breakdown
breakdown = flop_counter.get_flop_breakdown()
for category, flops in breakdown.items():
    print(f"{category}: {format_flops(flops)}")
```

## Architecture

### FLOP Registry System

The FLOP counting system consists of two main components:

#### 1. VLLM FLOP Registry (`vllm_flop_registry.py`)

Registers FLOP counting formulas for all VLLM custom operations:

- **150+ custom operations** across all categories
- **Automatic registration** when module is imported
- **Categorized by operation type** (attention, GEMM, activation, etc.)
- **Safe registration** handling missing operations gracefully

#### 2. Enhanced FLOP Counter (`flop_counter.py`)

Integrates PyTorch's FLOP counter with VLLM-specific enhancements:

- **Auto-imports** VLLM registry on module load
- **Enhanced categorization** of FLOP breakdown

## Supported Operations

### High-Impact Operations (Major FLOP Contributors)

#### Quantized GEMM Operations

| Operation | Description | FLOP Formula |
|-----------|-------------|--------------|
| `torch.ops._C.awq_gemm` | AWQ quantized GEMM | `2 * M * N * K` |
| `torch.ops._C.gptq_marlin_gemm` | GPTQ Marlin GEMM | `2 * M * N * K` |
| `torch.ops._C.cutlass_scaled_mm` | CUTLASS scaled GEMM | `2 * M * N * K` |
| `torch.ops._C.machete_mm` | Machete mixed precision | `2 * M * N * K` |

#### Attention Operations

| Operation | Description | FLOP Formula |
|-----------|-------------|--------------|
| `torch.ops._C.paged_attention_v1/v2` | PagedAttention kernels | `2 * B * H * S * D * C + 3 * B * H * S * C + 2 * B * H * S * C * D` |
| `torch.ops._C.cutlass_mla_decode` | CUTLASS MLA decode | `2 * B * H * S * D * C` |
| `torch.ops._C.flash_mla_fwd_kvcache` | FlashMLA forward | `2 * B * H * S * D * C` |

#### MoE Operations

| Operation | Description | FLOP Formula |
|-----------|-------------|--------------|
| `torch.ops._moe_C.moe_wna16_gemm` | MoE W16A16 GEMM | `2 * M * N * K * num_experts` |
| `torch.ops._moe_C.topk_softmax` | TopK softmax | `3 * total_elements + 4 * total_elements` |

### Medium-Impact Operations

#### Normalization Operations

| Operation | Description | FLOP Formula |
|-----------|-------------|--------------|
| `torch.ops._C.rms_norm` | RMS normalization | `5 * total_elements` |
| `torch.ops._C.fused_add_rms_norm` | Fused Add + RMS norm | `6 * total_elements` |

#### Activation Operations

| Operation | Description | FLOP Formula |
|-----------|-------------|--------------|
| `torch.ops._C.silu_and_mul` | SwiGLU activation | `5 * total_elements` |
| `torch.ops._C.gelu_and_mul` | GeGLU activation | `9 * total_elements` |

### Low-Impact Operations

#### Quantization Operations

| Operation | Description | FLOP Formula |
|-----------|-------------|--------------|
| `torch.ops._C.static_scaled_fp8_quant` | FP8 quantization | `2 * total_elements` |
| `torch.ops._C.dynamic_scaled_int8_quant` | INT8 quantization | `3 * total_elements` |

#### Memory Operations (Zero FLOPs)

- `torch.ops._C.permute_cols`
- `torch.ops._C_cache_ops.copy_blocks`
- `torch.ops._C.gptq_marlin_repack`

## Usage Patterns

### 1. Production Monitoring

```python
from vllm.profiler import FlopCounter

# Create counter for production monitoring
counter = FlopCounter()

with counter:
    # Your VLLM inference code
    outputs = llm.generate(prompts, sampling_params)

# Analyze FLOP breakdown
breakdown = counter.get_flop_breakdown()
```

### 2. Performance Analysis

```python
from vllm.profiler import FlopContextManager, format_flops

def analyze_model_performance(model_name: str, prompts: List[str]):
    llm = LLM(model=model_name)
    
    with FlopContextManager() as flop_counter:
        outputs = llm.generate(prompts, sampling_params)
    
    total_flops = flop_counter.get_total_flops()
    breakdown = flop_counter.get_flop_breakdown()
    
    print(f"Model: {model_name}")
    print(f"Total FLOPs: {format_flops(total_flops)}")
    print(f"GEMM FLOPs: {format_flops(breakdown['mm_flops'])} ({breakdown['mm_flops']/total_flops*100:.1f}%)")
    print(f"Attention FLOPs: {format_flops(breakdown['attention_flops'])} ({breakdown['attention_flops']/total_flops*100:.1f}%)")
    
    return total_flops, breakdown
```

### 3. Custom Operation Development

When adding new VLLM operations, register FLOP counting:

```python
from torch.utils.flop_counter import register_flop_formula
import torch

@register_flop_formula(torch.ops._C.my_custom_gemm)
def my_custom_gemm_flops(input_shape, weight_shape, **kwargs):
    M = input_shape[0] * input_shape[1]  # Batch * sequence
    K = input_shape[2]  # Input features
    N = weight_shape[1]  # Output features
    return 2 * M * N * K  # Standard GEMM formula

# Test the registration
with FlopContextManager() as flop_counter:
    # Call your custom operation
    result = torch.ops._C.my_custom_gemm(input_tensor, weight_tensor)
    
    # Check if FLOPs are counted
    total_flops = flop_counter.get_total_flops()
    if total_flops > 0:
        print("✅ Custom operation properly registered!")
```

## Advanced Features

### Categorical FLOP Breakdown

The system automatically categorizes operations:

```python
breakdown = flop_counter.get_flop_breakdown()

# Categories available:
# - mm_flops: Matrix multiplication operations
# - attention_flops: Attention mechanism operations  
# - activation_flops: Activation function operations
# - normalization_flops: Normalization operations
```

### Integration with Layerwise Profiling

```python
from vllm.profiler import layerwise_profile

with layerwise_profile(enable_flop_counting=True) as profiler:
    outputs = llm.generate(prompts, sampling_params)

results = profiler.results
results.print_flop_summary()  # Shows FLOP breakdown by layer
```

### Model-Specific FLOP Patterns

Different model architectures show distinct FLOP patterns:

#### **Transformer LLMs (e.g., OPT, GPT-2)**

```python
# Example: facebook/opt-125m
breakdown = flop_counter.get_flop_breakdown()
# Expected pattern:
# mm_flops: 75-85% (standard GEMM operations)
# attention_flops: 10-15% (PagedAttention kernels)
# activation_flops: 2-5% (SwiGLU, GeGLU)
# normalization_flops: 1-3% (RMS norm, layer norm)
```

#### **Mixture-of-Experts LLMs (e.g., Mixtral)**

```python
# Example: mistralai/Mixtral-8x7B-Instruct-v0.1
breakdown = flop_counter.get_flop_breakdown()
# Expected pattern:
# mm_flops: 60-70% (expert GEMM operations)
# moe_flops: 15-25% (expert routing, topk_softmax)
# attention_flops: 10-15% (standard attention)
# Key operations: moe_wna16_gemm, topk_softmax
```

#### **Embedding Models (e.g., E5-Mistral)**

```python
# Example: intfloat/e5-mistral-7b-instruct
breakdown = flop_counter.get_flop_breakdown()
# Expected pattern:
# mm_flops: 80-90% (representation learning optimized)
# attention_flops: 5-15% (encoder-style attention)
# Lower token generation, focus on encoding
```

#### **Multi-modal LLMs (e.g., LLaVA)**

```python
# Example: llava-hf/llava-1.5-7b-hf
breakdown = flop_counter.get_flop_breakdown()
# Expected pattern:
# attention_flops: 20-35% (vision-text cross-attention)
# mm_flops: 60-75% (vision encoder + text processing)
# Complex attention patterns for modality fusion
```

## Troubleshooting

### Common Issues

1. **Zero FLOP counts**: Ensure VLLM registry is imported

   ```python
   from vllm.profiler.vllm_flop_registry import register_all_vllm_flop_formulas
   register_all_vllm_flop_formulas()
   ```

2. **Inaccurate FLOP counts**: Verify that custom operations are properly registered

   ```python
   # Check FLOP breakdown for expected operations
   breakdown = flop_counter.get_flop_breakdown()
   print(breakdown)
   ```

### Performance Considerations

- FLOP counting is lightweight with ~0.1% overhead
- Registry auto-imports on first use

## Implementation Details

### FLOP Formula Categories

#### Matrix Operations

- **Standard GEMM**: `2 * M * N * K`
- **Sparse GEMM**: `2 * M * N * K * sparsity_factor`
- **Quantized GEMM**: Same as standard, quantization affects precision not compute

#### Attention Operations

- **Query-Key**: `2 * batch * heads * seq_len * head_dim * context_len`
- **Softmax**: `3 * batch * heads * seq_len * context_len`
- **Attention-Value**: `2 * batch * heads * seq_len * context_len * head_dim`

#### Element-wise Operations

- **Activations**: `4-9 * total_elements` (depends on complexity)
- **Normalization**: `5-6 * total_elements`
- **Quantization**: `2-4 * total_elements`

### Operation Priority for Registration

1. **Critical (Must have)**: Quantized GEMM, PagedAttention
2. **High**: MoE operations, normalization
3. **Medium**: Activations, positional encoding
4. **Low**: Memory operations, utilities

This prioritization helps focus development effort on operations with the highest FLOP impact.
