# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Comprehensive FLOP counting registry for all VLLM custom operations.

This module registers FLOP counting functions for all custom CUDA kernels
and operations in VLLM with PyTorch's FLOP counter registry.
"""

import logging
import math

import torch
from torch.utils.flop_counter import register_flop_formula

__all__ = [
    "register_all_vllm_flop_formulas",
]

logger = logging.getLogger(__name__)


def _safe_register_flop_formula(op, flop_func, op_name: str):
    """Safely register a FLOP formula, handling missing operations."""
    try:
        register_flop_formula(op)(flop_func)
        logger.debug("Registered FLOP formula for %s", op_name)
    except (AttributeError, RuntimeError) as e:
        logger.debug("Skipping registration for %s: %s", op_name, e)


# ==================== ATTENTION OPERATIONS ====================


def paged_attention_v1_flops(
    query_shape: list[int],
    key_cache_shape: list[int],
    value_cache_shape: list[int],
    context_lens_shape: list[int],
    **kwargs,
) -> int:
    """FLOP count for PagedAttention V1."""
    batch_size, num_heads, seq_len, head_dim = query_shape
    block_size = key_cache_shape[3]

    # Approximate context length from shapes
    max_context_len = key_cache_shape[2] * block_size
    avg_context_len = max_context_len // 2  # Conservative estimate

    # Q @ K^T: batch_size * num_heads * seq_len * head_dim * avg_context_len
    qk_flops = 2 * batch_size * num_heads * seq_len * head_dim * avg_context_len

    # Softmax: batch_size * num_heads * seq_len * avg_context_len * 3
    # (exp + sum + div)
    softmax_flops = 3 * batch_size * num_heads * seq_len * avg_context_len

    # Attn @ V: batch_size * num_heads * seq_len * avg_context_len * head_dim
    av_flops = 2 * batch_size * num_heads * seq_len * avg_context_len * head_dim

    return qk_flops + softmax_flops + av_flops


def paged_attention_v2_flops(
    out_shape: list[int],
    query_shape: list[int],
    key_cache_shape: list[int],
    value_cache_shape: list[int],
    **kwargs,
) -> int:
    """FLOP count for PagedAttention V2 (similar to V1 but more optimized)."""
    return paged_attention_v1_flops(query_shape, key_cache_shape,
                                    value_cache_shape, [], **kwargs)


def merge_attn_states_flops(states_shape: list[int],
                            split_sizes_shape: list[int], **kwargs) -> int:
    """FLOP count for merging attention states."""
    # Simple tensor concatenation and reduction
    total_elements = math.prod(states_shape)
    return total_elements  # One operation per element


def cutlass_mla_decode_flops(
    input_shape: list[int],
    k_cache_shape: list[int],
    v_cache_shape: list[int],
    **kwargs,
) -> int:
    """FLOP count for CUTLASS MLA decode."""
    batch_size, seq_len, hidden_dim = input_shape
    context_len = k_cache_shape[2] if len(k_cache_shape) > 2 else seq_len

    # MLA decode involves complex attention computation
    # Approximate as standard attention
    num_heads = 32  # Default approximation
    head_dim = hidden_dim // num_heads

    return 2 * batch_size * num_heads * seq_len * head_dim * context_len


def flash_mla_fwd_kvcache_flops(
    query_shape: list[int],
    k_cache_shape: list[int],
    v_cache_shape: list[int],
    **kwargs,
) -> int:
    """FLOP count for FlashMLA forward with KV cache."""
    return cutlass_mla_decode_flops(query_shape, k_cache_shape, v_cache_shape,
                                    **kwargs)


# ==================== QUANTIZED GEMM OPERATIONS ====================


def _gemm_flops(M: int, N: int, K: int) -> int:
    """Standard GEMM FLOP count: 2*M*N*K."""
    return 2 * M * N * K


def awq_gemm_flops(input_shape: list[int], weight_shape: list[int],
                   **kwargs) -> int:
    """FLOP count for AWQ quantized GEMM."""
    M = math.prod(input_shape[:-1])
    K = input_shape[-1]
    N = weight_shape[-1]
    return _gemm_flops(M, N, K)


def aqlm_gemm_flops(input_shape: list[int], codes_shape: list[int],
                    **kwargs) -> int:
    """FLOP count for AQLM quantized GEMM."""
    M = math.prod(input_shape[:-1])
    K = input_shape[-1]
    N = codes_shape[-1] * 8  # AQLM expands codes
    return _gemm_flops(M, N, K)


def marlin_gemm_flops(input_shape: list[int], weight_shape: list[int],
                      **kwargs) -> int:
    """FLOP count for Marlin quantized GEMM."""
    M = math.prod(input_shape[:-1])
    K = input_shape[-1]
    N = weight_shape[0]  # Marlin weights are transposed
    return _gemm_flops(M, N, K)


def gptq_marlin_gemm_flops(input_shape: list[int], b_q_weight_shape: list[int],
                           **kwargs) -> int:
    """FLOP count for GPTQ Marlin quantized GEMM."""
    M = math.prod(input_shape[:-1])
    K = input_shape[-1]
    N = b_q_weight_shape[1] * 8  # 4-bit quantization
    return _gemm_flops(M, N, K)


def gptq_marlin_24_gemm_flops(input_shape: list[int],
                              b_q_weight_shape: list[int], **kwargs) -> int:
    """FLOP count for GPTQ Marlin 2:4 sparse quantized GEMM."""
    base_flops = gptq_marlin_gemm_flops(input_shape, b_q_weight_shape,
                                        **kwargs)
    return base_flops // 2  # 2:4 sparsity reduces computation by ~50%


def machete_mm_flops(input_shape: list[int], weight_shape: list[int],
                     **kwargs) -> int:
    """FLOP count for Machete mixed precision GEMM."""
    M = math.prod(input_shape[:-1])
    K = input_shape[-1]
    N = weight_shape[0]
    return _gemm_flops(M, N, K)


def cutlass_scaled_mm_flops(input_shape: list[int], weight_shape: list[int],
                            **kwargs) -> int:
    """FLOP count for CUTLASS scaled GEMM."""
    M = math.prod(input_shape[:-1])
    K = input_shape[-1]
    N = weight_shape[-1]
    return _gemm_flops(M, N, K)


def gptq_gemm_flops(input_shape: list[int], qweight_shape: list[int],
                    **kwargs) -> int:
    """FLOP count for GPTQ quantized GEMM."""
    M = math.prod(input_shape[:-1])
    K = input_shape[-1]
    N = qweight_shape[1] * 8  # 4-bit quantization
    return _gemm_flops(M, N, K)


def ggml_mul_mat_flops(input_shape: list[int], weight_shape: list[int],
                       **kwargs) -> int:
    """FLOP count for GGML matrix multiplication."""
    M = math.prod(input_shape[:-1])
    K = input_shape[-1]
    N = weight_shape[0]
    return _gemm_flops(M, N, K)


def allspark_w8a16_gemm_flops(input_shape: list[int], weight_shape: list[int],
                              **kwargs) -> int:
    """FLOP count for AllSpark W8A16 GEMM."""
    M = math.prod(input_shape[:-1])
    K = input_shape[-1]
    N = weight_shape[-1]
    return _gemm_flops(M, N, K)


# ==================== MOE OPERATIONS ====================


def moe_wna16_gemm_flops(input_shape: list[int],
                         expert_weights_shape: list[int], **kwargs) -> int:
    """FLOP count for MoE W16A16 GEMM."""
    M = math.prod(input_shape[:-1])
    K = input_shape[-1]
    num_experts, N, _ = expert_weights_shape
    # Assume we compute all experts (worst case)
    return _gemm_flops(M, N, K) * num_experts


def moe_wna16_marlin_gemm_flops(input_shape: list[int],
                                expert_weights_shape: list[int],
                                **kwargs) -> int:
    """FLOP count for MoE W16A16 Marlin GEMM."""
    return moe_wna16_gemm_flops(input_shape, expert_weights_shape, **kwargs)


def cutlass_moe_mm_flops(input_shape: list[int],
                         weight_shapes: list[list[int]], **kwargs) -> int:
    """FLOP count for CUTLASS MoE GEMM."""
    total_flops = 0
    M = math.prod(input_shape[:-1])
    K = input_shape[-1]

    for weight_shape in weight_shapes:
        N = weight_shape[-1]
        total_flops += _gemm_flops(M, N, K)

    return total_flops


def topk_softmax_flops(input_shape: list[int], **kwargs) -> int:
    """FLOP count for topk softmax operation."""
    total_elements = math.prod(input_shape)

    # Softmax: 3 ops per element (exp, sum, div)
    softmax_flops = 3 * total_elements

    # TopK: O(n log k) complexity, approximate as 4 * total_elements
    topk_flops = 4 * total_elements

    return softmax_flops + topk_flops


# ==================== ACTIVATION OPERATIONS ====================


def silu_and_mul_flops(input_shape: list[int], **kwargs) -> int:
    """FLOP count for SiLU and multiply (SwiGLU)."""
    total_elements = math.prod(input_shape)
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x)) ≈ 4 ops per element
    # Multiply with gate: 1 op per element
    return 5 * total_elements


def gelu_and_mul_flops(input_shape: list[int], **kwargs) -> int:
    """FLOP count for GELU and multiply (GeGLU)."""
    total_elements = math.prod(input_shape)
    # GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    # ≈ 8 ops per element
    # Multiply with gate: 1 op per element
    return 9 * total_elements


def gelu_new_flops(input_shape: list[int], **kwargs) -> int:
    """FLOP count for GELU new implementation."""
    total_elements = math.prod(input_shape)
    # GELU approximation: ≈ 8 ops per element
    return 8 * total_elements


def gelu_fast_flops(input_shape: list[int], **kwargs) -> int:
    """FLOP count for fast GELU."""
    total_elements = math.prod(input_shape)
    # Fast GELU: x * sigmoid(1.702 * x) ≈ 4 ops per element
    return 4 * total_elements


def gelu_tanh_and_mul_flops(input_shape: list[int], **kwargs) -> int:
    """FLOP count for GELU tanh and multiply."""
    return gelu_and_mul_flops(input_shape, **kwargs)


# ==================== NORMALIZATION OPERATIONS ====================


def rms_norm_flops(input_shape: list[int], **kwargs) -> int:
    """FLOP count for RMS normalization."""
    total_elements = math.prod(input_shape)
    # RMS norm: sqrt(mean(x^2)) normalization ≈ 5 ops per element
    return 5 * total_elements


def fused_add_rms_norm_flops(input_shape: list[int], residual_shape: list[int],
                             **kwargs) -> int:
    """FLOP count for fused add + RMS norm."""
    total_elements = math.prod(input_shape)
    # Add: 1 op per element, RMS norm: 5 ops per element
    return 6 * total_elements


# ==================== ROTARY EMBEDDING OPERATIONS ====================


def rotary_embedding_flops(input_shape: list[int], **kwargs) -> int:
    """FLOP count for rotary embedding."""
    total_elements = math.prod(input_shape)
    # Rotary embedding involves sin/cos computation and rotation
    # ≈ 6 ops per element
    return 6 * total_elements


def batched_rotary_embedding_flops(input_shape: list[int], **kwargs) -> int:
    """FLOP count for batched rotary embedding."""
    return rotary_embedding_flops(input_shape, **kwargs)


# ==================== QUANTIZATION OPERATIONS ====================


def static_scaled_fp8_quant_flops(input_shape: list[int], **kwargs) -> int:
    """FLOP count for static FP8 quantization."""
    total_elements = math.prod(input_shape)
    # Quantization: scale + round ≈ 2 ops per element
    return 2 * total_elements


def dynamic_scaled_fp8_quant_flops(input_shape: list[int], **kwargs) -> int:
    """FLOP count for dynamic FP8 quantization."""
    total_elements = math.prod(input_shape)
    # Dynamic quantization: compute scale + quantize ≈ 4 ops per element
    return 4 * total_elements


def static_scaled_int8_quant_flops(input_shape: list[int], **kwargs) -> int:
    """FLOP count for static INT8 quantization."""
    total_elements = math.prod(input_shape)
    # INT8 quantization: scale + clamp + round ≈ 3 ops per element
    return 3 * total_elements


# ==================== CACHE OPERATIONS ====================


def reshape_and_cache_flops(key_shape: list[int], value_shape: list[int],
                            **kwargs) -> int:
    """FLOP count for reshape and cache operation."""
    # Reshape operations are typically memory-bound, minimal FLOPs
    key_elements = math.prod(key_shape)
    value_elements = math.prod(value_shape)
    return key_elements + value_elements  # 1 op per element for copy


def copy_blocks_flops(src_shape: list[int], **kwargs) -> int:
    """FLOP count for copying cache blocks."""
    total_elements = math.prod(src_shape)
    return total_elements  # 1 op per element for copy


# ==================== MAMBA OPERATIONS ====================


def selective_scan_fwd_flops(
    u_shape: list[int],
    delta_shape: list[int],
    A_shape: list[int],
    B_shape: list[int],
    C_shape: list[int],
    **kwargs,
) -> int:
    """FLOP count for Mamba selective scan forward."""
    batch_size, seq_len, d_inner = u_shape
    # Selective scan involves state space computation ≈ 10 ops per element
    return 10 * batch_size * seq_len * d_inner


def causal_conv1d_fwd_flops(input_shape: list[int], weight_shape: list[int],
                            **kwargs) -> int:
    """FLOP count for causal 1D convolution."""
    batch_size, d_inner, seq_len = input_shape
    d_conv = weight_shape[1]
    # 1D convolution: 2 * batch_size * d_inner * seq_len * d_conv
    return 2 * batch_size * d_inner * seq_len * d_conv


# ==================== REGISTRATION FUNCTION ====================


def register_all_vllm_flop_formulas():
    """Register FLOP counting formulas for all VLLM custom operations."""

    logger.info("Registering FLOP formulas for VLLM custom operations...")

    # Attention Operations
    _safe_register_flop_formula(
        torch.ops._C.paged_attention_v1,
        paged_attention_v1_flops,
        "torch.ops._C.paged_attention_v1",
    )
    _safe_register_flop_formula(
        torch.ops._C.paged_attention_v2,
        paged_attention_v2_flops,
        "torch.ops._C.paged_attention_v2",
    )
    _safe_register_flop_formula(
        torch.ops._C.merge_attn_states,
        merge_attn_states_flops,
        "torch.ops._C.merge_attn_states",
    )
    _safe_register_flop_formula(
        torch.ops._C.cutlass_mla_decode,
        cutlass_mla_decode_flops,
        "torch.ops._C.cutlass_mla_decode",
    )
    _safe_register_flop_formula(
        torch.ops._C.flash_mla_fwd_kvcache,
        flash_mla_fwd_kvcache_flops,
        "torch.ops._C.flash_mla_fwd_kvcache",
    )

    # Quantized GEMM Operations
    _safe_register_flop_formula(torch.ops._C.awq_gemm, awq_gemm_flops,
                                "torch.ops._C.awq_gemm")
    _safe_register_flop_formula(torch.ops._C.aqlm_gemm, aqlm_gemm_flops,
                                "torch.ops._C.aqlm_gemm")
    _safe_register_flop_formula(torch.ops._C.marlin_gemm, marlin_gemm_flops,
                                "torch.ops._C.marlin_gemm")
    _safe_register_flop_formula(
        torch.ops._C.gptq_marlin_gemm,
        gptq_marlin_gemm_flops,
        "torch.ops._C.gptq_marlin_gemm",
    )
    _safe_register_flop_formula(
        torch.ops._C.gptq_marlin_24_gemm,
        gptq_marlin_24_gemm_flops,
        "torch.ops._C.gptq_marlin_24_gemm",
    )
    _safe_register_flop_formula(torch.ops._C.machete_mm, machete_mm_flops,
                                "torch.ops._C.machete_mm")
    _safe_register_flop_formula(
        torch.ops._C.cutlass_scaled_mm,
        cutlass_scaled_mm_flops,
        "torch.ops._C.cutlass_scaled_mm",
    )
    _safe_register_flop_formula(
        torch.ops._C.cutlass_scaled_mm_azp,
        cutlass_scaled_mm_flops,
        "torch.ops._C.cutlass_scaled_mm_azp",
    )
    _safe_register_flop_formula(torch.ops._C.gptq_gemm, gptq_gemm_flops,
                                "torch.ops._C.gptq_gemm")
    _safe_register_flop_formula(
        torch.ops._C.ggml_mul_mat_a8,
        ggml_mul_mat_flops,
        "torch.ops._C.ggml_mul_mat_a8",
    )
    _safe_register_flop_formula(
        torch.ops._C.allspark_w8a16_gemm,
        allspark_w8a16_gemm_flops,
        "torch.ops._C.allspark_w8a16_gemm",
    )

    # Additional GEMM variants
    _safe_register_flop_formula(
        torch.ops._C.marlin_qqq_gemm,
        marlin_gemm_flops,
        "torch.ops._C.marlin_qqq_gemm",
    )
    _safe_register_flop_formula(
        torch.ops._C.cutlass_moe_mm,
        cutlass_moe_mm_flops,
        "torch.ops._C.cutlass_moe_mm",
    )
    _safe_register_flop_formula(
        torch.ops._C.cutlass_scaled_sparse_mm,
        cutlass_scaled_mm_flops,
        "torch.ops._C.cutlass_scaled_sparse_mm",
    )
    _safe_register_flop_formula(
        torch.ops._C.cutlass_scaled_fp4_mm,
        cutlass_scaled_mm_flops,
        "torch.ops._C.cutlass_scaled_fp4_mm",
    )
    _safe_register_flop_formula(
        torch.ops._C.cutlass_fp4_group_mm,
        cutlass_scaled_mm_flops,
        "torch.ops._C.cutlass_fp4_group_mm",
    )

    # MoE Operations
    _safe_register_flop_formula(
        torch.ops._moe_C.moe_wna16_gemm,
        moe_wna16_gemm_flops,
        "torch.ops._moe_C.moe_wna16_gemm",
    )
    _safe_register_flop_formula(
        torch.ops._moe_C.moe_wna16_marlin_gemm,
        moe_wna16_marlin_gemm_flops,
        "torch.ops._moe_C.moe_wna16_marlin_gemm",
    )
    _safe_register_flop_formula(
        torch.ops._moe_C.marlin_gemm_moe,
        moe_wna16_marlin_gemm_flops,
        "torch.ops._moe_C.marlin_gemm_moe",
    )
    _safe_register_flop_formula(
        torch.ops._moe_C.topk_softmax,
        topk_softmax_flops,
        "torch.ops._moe_C.topk_softmax",
    )

    # Activation Operations
    _safe_register_flop_formula(
        torch.ops._C.silu_and_mul,
        silu_and_mul_flops,
        "torch.ops._C.silu_and_mul",
    )
    _safe_register_flop_formula(
        torch.ops._C.silu_and_mul_quant,
        silu_and_mul_flops,
        "torch.ops._C.silu_and_mul_quant",
    )
    _safe_register_flop_formula(
        torch.ops._C.gelu_and_mul,
        gelu_and_mul_flops,
        "torch.ops._C.gelu_and_mul",
    )
    _safe_register_flop_formula(
        torch.ops._C.gelu_tanh_and_mul,
        gelu_tanh_and_mul_flops,
        "torch.ops._C.gelu_tanh_and_mul",
    )
    _safe_register_flop_formula(torch.ops._C.gelu_new, gelu_new_flops,
                                "torch.ops._C.gelu_new")
    _safe_register_flop_formula(torch.ops._C.gelu_fast, gelu_fast_flops,
                                "torch.ops._C.gelu_fast")
    _safe_register_flop_formula(torch.ops._C.gelu_quick, gelu_fast_flops,
                                "torch.ops._C.gelu_quick")

    # Normalization Operations
    _safe_register_flop_formula(torch.ops._C.rms_norm, rms_norm_flops,
                                "torch.ops._C.rms_norm")
    _safe_register_flop_formula(
        torch.ops._C.fused_add_rms_norm,
        fused_add_rms_norm_flops,
        "torch.ops._C.fused_add_rms_norm",
    )
    _safe_register_flop_formula(
        torch.ops._C.rms_norm_static_fp8_quant,
        rms_norm_flops,
        "torch.ops._C.rms_norm_static_fp8_quant",
    )
    _safe_register_flop_formula(
        torch.ops._C.fused_add_rms_norm_static_fp8_quant,
        fused_add_rms_norm_flops,
        "torch.ops._C.fused_add_rms_norm_static_fp8_quant",
    )
    _safe_register_flop_formula(
        torch.ops._C.rms_norm_dynamic_per_token_quant,
        rms_norm_flops,
        "torch.ops._C.rms_norm_dynamic_per_token_quant",
    )

    # Rotary Embedding Operations
    _safe_register_flop_formula(
        torch.ops._C.rotary_embedding,
        rotary_embedding_flops,
        "torch.ops._C.rotary_embedding",
    )
    _safe_register_flop_formula(
        torch.ops._C.batched_rotary_embedding,
        batched_rotary_embedding_flops,
        "torch.ops._C.batched_rotary_embedding",
    )

    # Quantization Operations
    _safe_register_flop_formula(
        torch.ops._C.static_scaled_fp8_quant,
        static_scaled_fp8_quant_flops,
        "torch.ops._C.static_scaled_fp8_quant",
    )
    _safe_register_flop_formula(
        torch.ops._C.dynamic_scaled_fp8_quant,
        dynamic_scaled_fp8_quant_flops,
        "torch.ops._C.dynamic_scaled_fp8_quant",
    )
    _safe_register_flop_formula(
        torch.ops._C.dynamic_per_token_scaled_fp8_quant,
        dynamic_scaled_fp8_quant_flops,
        "torch.ops._C.dynamic_per_token_scaled_fp8_quant",
    )
    _safe_register_flop_formula(
        torch.ops._C.static_scaled_int8_quant,
        static_scaled_int8_quant_flops,
        "torch.ops._C.static_scaled_int8_quant",
    )
    _safe_register_flop_formula(
        torch.ops._C.dynamic_scaled_int8_quant,
        static_scaled_int8_quant_flops,
        "torch.ops._C.dynamic_scaled_int8_quant",
    )

    # Cache Operations
    _safe_register_flop_formula(
        torch.ops._C_cache_ops.reshape_and_cache,
        reshape_and_cache_flops,
        "torch.ops._C_cache_ops.reshape_and_cache",
    )
    _safe_register_flop_formula(
        torch.ops._C_cache_ops.reshape_and_cache_flash,
        reshape_and_cache_flops,
        "torch.ops._C_cache_ops.reshape_and_cache_flash",
    )
    _safe_register_flop_formula(
        torch.ops._C_cache_ops.copy_blocks,
        copy_blocks_flops,
        "torch.ops._C_cache_ops.copy_blocks",
    )
    _safe_register_flop_formula(
        torch.ops._C_cache_ops.copy_blocks_mla,
        copy_blocks_flops,
        "torch.ops._C_cache_ops.copy_blocks_mla",
    )

    # Mamba Operations
    _safe_register_flop_formula(
        torch.ops._C.selective_scan_fwd,
        selective_scan_fwd_flops,
        "torch.ops._C.selective_scan_fwd",
    )
    _safe_register_flop_formula(
        torch.ops._C.causal_conv1d_fwd,
        causal_conv1d_fwd_flops,
        "torch.ops._C.causal_conv1d_fwd",
    )
    _safe_register_flop_formula(
        torch.ops._C.causal_conv1d_update,
        causal_conv1d_fwd_flops,
        "torch.ops._C.causal_conv1d_update",
    )

    # Additional operations with generic flop counting
    # (These are operations that have minimal compute or are hard to estimate)

    def zero_flops(*args, **kwargs) -> int:
        """Generic zero FLOP count for memory-only operations."""
        return 0

    def minimal_flops(input_shape: list[int], **kwargs) -> int:
        """Minimal FLOP count for simple operations."""
        return math.prod(input_shape)

    # Memory and utility operations (zero FLOPs)
    memory_ops = [
        "torch.ops._C.permute_cols",
        "torch.ops._C.awq_dequantize",
        "torch.ops._C.aqlm_dequant",
        "torch.ops._C.gptq_marlin_repack",
        "torch.ops._C.awq_marlin_repack",
        "torch.ops._C.machete_prepack_B",
        "torch.ops._C_cache_ops.swap_blocks",
        "torch.ops._C_cache_ops.convert_fp8",
        "torch.ops._C_cache_ops.gather_cache",
        "torch.ops._moe_C.moe_align_block_size",
        "torch.ops._moe_C.moe_permute",
        "torch.ops._moe_C.moe_unpermute",
    ]

    for op_name in memory_ops:
        try:
            op = eval(op_name)
            _safe_register_flop_formula(op, zero_flops, op_name)
        except Exception:
            pass

    # Simple compute operations (minimal FLOPs)
    simple_ops = [
        "torch.ops._C.apply_repetition_penalties_",
        "torch.ops._moe_C.moe_sum",
        "torch.ops._moe_C.shuffle_rows",
    ]

    for op_name in simple_ops:
        try:
            op = eval(op_name)
            _safe_register_flop_formula(op, minimal_flops, op_name)
        except Exception:
            pass

    logger.info("VLLM FLOP formula registration completed!")


# Auto-register when module is imported
register_all_vllm_flop_formulas()
