# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from dataclasses import dataclass, field

from torch.utils.flop_counter import FlopCounterMode

__all__ = [
    "FlopCounter", "FlopCount", "DetailedFlopCount", "FlopContextManager",
    "format_flops"
]


@dataclass
class FlopCount:
    total_flops: int = 0
    flop_counts: dict[str, int] = field(default_factory=dict)

    def total(self) -> int:
        return self.total_flops

    def to_dict(self) -> dict[str, int]:
        return {"total_flops": self.total_flops, **self.flop_counts}

    def __add__(self, other: 'FlopCount') -> 'FlopCount':
        result_counts = self.flop_counts.copy()
        for op, count in other.flop_counts.items():
            result_counts[op] = result_counts.get(op, 0) + count

        return FlopCount(total_flops=self.total_flops + other.total_flops,
                         flop_counts=result_counts)

    def __iadd__(self, other: 'FlopCount') -> 'FlopCount':
        self.total_flops += other.total_flops
        for op, count in other.flop_counts.items():
            self.flop_counts[op] = self.flop_counts.get(op, 0) + count
        return self


@dataclass
class DetailedFlopCount:
    operation_counts: dict[str, int] = field(default_factory=dict)
    layer_counts: dict[str, 'FlopCount'] = field(default_factory=dict)
    total_flops: int = 0
    mm_flops: int = 0
    attention_flops: int = 0
    activation_flops: int = 0
    normalization_flops: int = 0
    # Additional categorizations for offline analysis
    embedding_flops: int = 0
    convolution_flops: int = 0
    other_flops: int = 0

    def add_operation(self, op_name: str, flops: int):
        self.operation_counts[op_name] = (
            self.operation_counts.get(op_name, 0) + flops)
        self.total_flops += flops

    def get_breakdown_dict(self) -> dict[str, int]:
        """Get a dictionary breakdown of FLOP categories."""
        return {
            'total_flops': self.total_flops,
            'mm_flops': self.mm_flops,
            'attention_flops': self.attention_flops,
            'activation_flops': self.activation_flops,
            'normalization_flops': self.normalization_flops,
            'embedding_flops': self.embedding_flops,
            'convolution_flops': self.convolution_flops,
            'other_flops': self.other_flops
        }

    def get_percentage_breakdown(self) -> dict[str, float]:
        """Get percentage breakdown of FLOP categories."""
        if self.total_flops == 0:
            return {k: 0.0 for k in self.get_breakdown_dict()}

        breakdown = self.get_breakdown_dict()
        return {
            k: (v / self.total_flops * 100.0) if k != 'total_flops' else 100.0
            for k, v in breakdown.items()
        }


class ModelFlopEstimator:
    """Estimates FLOPs based on model architecture when PyTorch counter fails."""

    def __init__(self, model_config):
        self.config = model_config
        self.vocab_size = getattr(model_config, 'vocab_size', 32000)
        self.hidden_size = getattr(model_config, 'hidden_size', 4096)
        self.num_layers = getattr(model_config, 'num_hidden_layers', 32)
        self.num_heads = getattr(model_config, 'num_attention_heads', 32)
        self.intermediate_size = getattr(model_config, 'intermediate_size',
                                         4 * self.hidden_size)
        self.head_dim = self.hidden_size // self.num_heads

    def estimate_forward_pass_flops(self,
                                    batch_size: int,
                                    seq_len: int,
                                    past_length: int = 0) -> int:
        """Estimate FLOPs for a single forward pass."""
        total_seq_len = seq_len + past_length
        flops = 0

        # For each transformer layer
        for _ in range(self.num_layers):
            # Self-attention
            # Q, K, V projections: 3 * (batch * seq_len * hidden_size^2)
            flops += (3 * batch_size * seq_len * self.hidden_size *
                      self.hidden_size)

            # Attention computation: Q @ K^T
            flops += (batch_size * self.num_heads * seq_len * total_seq_len *
                      self.head_dim)

            # Softmax (approximate as 3 ops per element)
            flops += batch_size * self.num_heads * seq_len * total_seq_len * 3

            # Attention @ V
            flops += (batch_size * self.num_heads * seq_len * total_seq_len *
                      self.head_dim)

            # Output projection
            flops += batch_size * seq_len * self.hidden_size * self.hidden_size

            # MLP
            # Up projection
            flops += (batch_size * seq_len * self.hidden_size *
                      self.intermediate_size)
            # Activation (SiLU/GELU - approximate as 8 ops per element)
            flops += batch_size * seq_len * self.intermediate_size * 8
            # Down projection
            flops += (batch_size * seq_len * self.intermediate_size *
                      self.hidden_size)

            # Layer norms (approximate as 5 ops per element)
            # Operations: mean, var, sub, div, mul+add
            flops += 2 * batch_size * seq_len * self.hidden_size * 5

        # Final layer norm
        flops += batch_size * seq_len * self.hidden_size * 5

        # LM head projection (only for last token in generation)
        if seq_len == 1:  # Generation mode
            flops += batch_size * self.hidden_size * self.vocab_size
        else:  # Prefill mode
            flops += batch_size * seq_len * self.hidden_size * self.vocab_size

        return flops

    def estimate_generation_flops(self, input_ids_shape,
                                  num_generated_tokens: int) -> dict:
        """Estimate total FLOPs for prefill + generation."""
        batch_size, input_seq_len = input_ids_shape

        # Prefill phase
        prefill_flops = self.estimate_forward_pass_flops(
            batch_size, input_seq_len, 0)

        # Generation phase (decode one token at a time)
        decode_flops = 0
        for i in range(num_generated_tokens):
            current_past_length = input_seq_len + i
            decode_flops += self.estimate_forward_pass_flops(
                batch_size, 1, current_past_length)

        total_flops = prefill_flops + decode_flops

        return {
            'total_flops': total_flops,
            'prefill_flops': prefill_flops,
            'decode_flops': decode_flops,
            'tokens_generated': num_generated_tokens
        }


class FlopCounter:

    def __init__(self, display: bool = False, use_model_flops: bool = True):
        self._display = display
        self._flop_mode: FlopCounterMode | None = None
        self._detailed_counts = DetailedFlopCount()
        self._use_model_flops = use_model_flops
        self._model_flop_estimator = None

    def get_total_flops(self) -> int:
        if self._flop_mode is None:
            return self._detailed_counts.total_flops

        pytorch_total = self._flop_mode.get_total_flops()
        if pytorch_total == 0 and self._detailed_counts.total_flops > 0:
            # PyTorch counter failed, use model estimation
            return self._detailed_counts.total_flops

        return pytorch_total

    def get_flop_breakdown(self) -> dict[str, int]:
        """Get categorized FLOP breakdown with enhanced categorization."""
        if self._flop_mode is None:
            return {
                'mm_flops': 0,
                'attention_flops': 0,
                'activation_flops': 0,
                'normalization_flops': 0,
                'embedding_flops': 0,
                'convolution_flops': 0,
                'other_flops': 0
            }
        raw_flops = self._flop_mode.get_flop_counts()

        # Extract operations from the 'Global' module which contains
        # aggregated counts
        global_flops = raw_flops.get('Global', {})

        mm_flops = 0
        attention_flops = 0
        activation_flops = 0
        normalization_flops = 0
        embedding_flops = 0
        convolution_flops = 0
        other_flops = 0

        for op, count in global_flops.items():
            op_name = str(op).lower()
            if any(mm_op in op_name
                   for mm_op in ['mm', 'bmm', 'addmm', 'matmul']):
                mm_flops += count
            elif 'attention' in op_name or 'attn' in op_name:
                attention_flops += count
            elif any(activation in op_name for activation in
                     ['relu', 'gelu', 'silu', 'swish', 'tanh', 'sigmoid']):
                activation_flops += count
            elif any(norm in op_name for norm in
                     ['layer_norm', 'group_norm', 'rms_norm', 'batch_norm']):
                normalization_flops += count
            elif 'embedding' in op_name or 'embed' in op_name:
                embedding_flops += count
            elif any(conv_op in op_name
                     for conv_op in ['conv', 'convolution']):
                convolution_flops += count
            else:
                other_flops += count

        return {
            'mm_flops': mm_flops,
            'attention_flops': attention_flops,
            'activation_flops': activation_flops,
            'normalization_flops': normalization_flops,
            'embedding_flops': embedding_flops,
            'convolution_flops': convolution_flops,
            'other_flops': other_flops
        }

    def set_model_for_estimation(self, model_config, generation_stats=None):
        """Set model configuration for FLOP estimation when PyTorch counting fails."""
        if self._use_model_flops:
            self._model_flop_estimator = ModelFlopEstimator(model_config)
            if generation_stats:
                self._apply_model_flop_estimation(generation_stats)

    def _apply_model_flop_estimation(self, generation_stats):
        """Apply model-based FLOP estimation using generation statistics."""
        if not self._model_flop_estimator:
            return

        input_shape = generation_stats.get('input_shape', (1, 10))
        num_generated = generation_stats.get('num_generated_tokens', 20)

        flop_breakdown = self._model_flop_estimator.estimate_generation_flops(
            input_shape, num_generated)

        # Update detailed counts with estimated values
        self._detailed_counts.total_flops = flop_breakdown['total_flops']

        # Rough breakdown by operation type (percentages based on typical transformer models)
        total = flop_breakdown['total_flops']
        self._detailed_counts.mm_flops = int(total *
                                             0.85)  # Matrix multiply dominates
        self._detailed_counts.attention_flops = int(total *
                                                    0.10)  # Attention overhead
        self._detailed_counts.activation_flops = int(total *
                                                     0.02)  # Activations
        self._detailed_counts.normalization_flops = int(total *
                                                        0.01)  # LayerNorms
        self._detailed_counts.embedding_flops = int(total * 0.01)  # Embeddings
        self._detailed_counts.other_flops = int(total * 0.01)  # Misc

        # Create layer counts (simplified)
        avg_flops_per_layer = total // self._model_flop_estimator.num_layers
        for i in range(self._model_flop_estimator.num_layers):
            layer_name = f"model.layers.{i}"
            self._detailed_counts.layer_counts[layer_name] = FlopCount(
                total_flops=avg_flops_per_layer,
                flop_counts={f"layer_{i}_ops": avg_flops_per_layer})

    def get_detailed_counts(self) -> DetailedFlopCount:
        if self._flop_mode is None:
            return self._detailed_counts

        raw_flops = self._flop_mode.get_flop_counts()
        global_flops = raw_flops.get('Global', {})

        # Check if PyTorch FLOP counter captured anything
        pytorch_total = self._flop_mode.get_total_flops()

        if pytorch_total == 0 and self._model_flop_estimator:
            # PyTorch counter failed, but we have estimates - use them
            print(
                "PyTorch FLOP counter captured 0 FLOPs, using model-based estimation"
            )
            return self._detailed_counts

        # PyTorch counter worked - use its data
        self._detailed_counts.total_flops = pytorch_total
        self._detailed_counts.operation_counts = global_flops

        layer_counts = {}
        for module_name, ops in raw_flops.items():
            if module_name != 'Global':
                total_flops = sum(ops.values())
                layer_counts[module_name] = FlopCount(total_flops=total_flops,
                                                      flop_counts=dict(ops))
        self._detailed_counts.layer_counts = layer_counts

        # Get categorized breakdown
        breakdown = self.get_flop_breakdown()
        self._detailed_counts.mm_flops = breakdown['mm_flops']
        self._detailed_counts.attention_flops = breakdown['attention_flops']
        self._detailed_counts.activation_flops = breakdown['activation_flops']
        self._detailed_counts.normalization_flops = (
            breakdown['normalization_flops'])
        self._detailed_counts.embedding_flops = breakdown['embedding_flops']
        self._detailed_counts.convolution_flops = breakdown[
            'convolution_flops']
        self._detailed_counts.other_flops = breakdown['other_flops']

        return self._detailed_counts

    def get_efficiency_metrics(self,
                               elapsed_time_sec: float) -> dict[str, float]:
        """Calculate efficiency metrics for offline analysis."""
        total_flops = self.get_total_flops()
        if elapsed_time_sec <= 0 or total_flops == 0:
            return {
                'gflops_per_sec': 0.0,
                'tflops_per_sec': 0.0,
                'flops_per_microsec': 0.0
            }

        return {
            'gflops_per_sec': total_flops / (elapsed_time_sec * 1e9),
            'tflops_per_sec': total_flops / (elapsed_time_sec * 1e12),
            'flops_per_microsec': total_flops / (elapsed_time_sec * 1e6)
        }

    def print_analysis_summary(self, elapsed_time_sec: float = None):
        """Print a comprehensive analysis summary for offline use."""
        total_flops = self.get_total_flops()
        breakdown = self.get_flop_breakdown()

        print("\n=== FLOP Analysis Summary ===")
        print(f"Total FLOPs: {format_flops(total_flops)}")

        if elapsed_time_sec:
            efficiency = self.get_efficiency_metrics(elapsed_time_sec)
            print(f"Elapsed Time: {elapsed_time_sec:.3f} seconds")
            print(
                f"Performance: {efficiency['gflops_per_sec']:.2f} GFLOPS/sec")
            print(
                f"Performance: {efficiency['tflops_per_sec']:.4f} TFLOPS/sec")

        print("\n=== FLOP Breakdown ===")
        for category, flops in breakdown.items():
            if flops > 0:
                percentage = ((flops / total_flops *
                               100) if total_flops > 0 else 0)
                flop_str = format_flops(flops)
                print(f"{category:20s}: {flop_str:>12s} ({percentage:5.1f}%)")

    def reset(self):
        self._detailed_counts = DetailedFlopCount()

    def get_table(self) -> str:
        if self._flop_mode is None:
            return "No FLOP data available"
        return self._flop_mode.get_table()

    def __enter__(self):
        self._flop_mode = FlopCounterMode(display=self._display)
        self._flop_mode.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._flop_mode.__exit__(exc_type, exc_val, exc_tb)


@contextmanager
def FlopContextManager(display: bool = False, auto_print: bool = False):
    """Context manager for FLOP counting in offline analysis.
    
    Args:
        display: Whether to display detailed PyTorch FLOP table
        auto_print: Whether to automatically print analysis summary on exit
    """
    counter = FlopCounter(display=display)
    start_time = None

    try:
        import time
        start_time = time.time()
        with counter:
            yield counter
    finally:
        if auto_print and start_time is not None:
            elapsed_time = time.time() - start_time
            counter.print_analysis_summary(elapsed_time)


def format_flops(flops: int) -> str:
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f} KFLOPs"
    else:
        return f"{flops} FLOPs"
