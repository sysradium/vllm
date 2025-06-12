# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.profiler.flop_counter import (DetailedFlopCount, FlopContextManager,
                                        FlopCounter, format_flops)
from vllm.profiler.layerwise_profile import (LayerwiseProfileResults,
                                             layerwise_profile)

# Import VLLM FLOP registry (optional)
try:
    import importlib.util
    _spec = importlib.util.find_spec("vllm.profiler.vllm_flop_registry")
    if _spec is not None:
        from vllm.profiler.vllm_flop_registry import (  # noqa: F401
            register_all_vllm_flop_formulas)
        _VLLM_FLOP_REGISTRY_AVAILABLE = True
    else:
        _VLLM_FLOP_REGISTRY_AVAILABLE = False
except ImportError:
    _VLLM_FLOP_REGISTRY_AVAILABLE = False

__all__ = [
    "layerwise_profile",
    "LayerwiseProfileResults",
    "FlopContextManager",
    "FlopCounter",
    "DetailedFlopCount",
    "format_flops",
]

# Add VLLM FLOP registry to __all__ if available
if _VLLM_FLOP_REGISTRY_AVAILABLE:
    __all__.extend([
        "register_all_vllm_flop_formulas",
    ])
