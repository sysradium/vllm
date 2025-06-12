# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example script demonstrating FLOP counting capabilities in vLLM.

This example shows how to:
1. Use the FlopContextManager for basic FLOP counting
2. Use layerwise profiling with FLOP counting enabled
3. Display FLOP summaries and performance metrics
4. Analyze FLOP breakdowns by operation category

Run with:
    python flop_counting.py
    python flop_counting.py --basic          # Run basic FLOP counting example
    python flop_counting.py --profiling      # Run layerwise profiling example
    python flop_counting.py --analysis       # Run performance analysis
    python flop_counting.py --all            # Run all examples
"""

import argparse
import time

from vllm import LLM, SamplingParams
from vllm.profiler import (
    FlopContextManager,
    format_flops,
    layerwise_profile,
)


def basic_flop_counting_example():
    """Example using FlopContextManager for basic FLOP counting."""
    print("=== Basic FLOP Counting Example ===")

    # Create LLM instance
    llm = LLM(model="facebook/opt-125m", max_num_seqs=1)

    # Sampling parameters
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

    # Test prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "Explain quantum computing in simple terms:",
    ]

    # Generate with FLOP counting
    with FlopContextManager() as flop_counter:
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.time()

    # Display results
    total_flops = flop_counter.get_total_flops()
    flop_breakdown = flop_counter.get_flop_breakdown()
    elapsed_time = end_time - start_time

    print(f"Total FLOPs: {format_flops(total_flops)}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"GFLOPS/sec: {total_flops / (elapsed_time * 1e9):.2f}")

    print("\nTop operations by FLOP count:")
    sorted_ops = sorted(flop_breakdown.items(), key=lambda x: x[1], reverse=True)
    for op_name, flops in sorted_ops[:10]:
        print(f"  {op_name}: {format_flops(flops)}")

    print("\nGenerated outputs:")
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt {i + 1}: {prompt}")
        print(f"Output: {generated_text}\n")


def layerwise_profiling_with_flops_example():
    """Example using layerwise profiling with FLOP counting."""
    print("\n=== Layerwise Profiling with FLOPs Example ===")

    # Create LLM instance
    llm = LLM(model="facebook/opt-125m", max_num_seqs=1)

    # Sampling parameters
    sampling_params = SamplingParams(temperature=0.0, max_tokens=20)

    prompt = "The quick brown fox"

    # Profile with FLOP counting enabled
    with layerwise_profile(num_running_seqs=1, enable_flop_counting=True) as profiler:
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        end_time = time.time()

    # Get profiling results
    results = profiler.results

    print(f"Generated text: {outputs[0].outputs[0].text}")
    print(f"Elapsed time: {end_time - start_time:.2f} seconds")

    # Display FLOP summary
    results.print_flop_summary()

    # Display timing and FLOP model table (first 20 entries)
    print("\n=== Model Stats (Top 20 by CUDA time) ===")
    results.print_model_table(
        column_widths={
            "name": 50,
            "cuda_time_us": 12,
            "flops": 12,
            "gflops_per_sec": 12,
            "trace": 40,
        }
    )


def performance_analysis_example():
    """Example showing performance analysis with different model types."""
    print("\n=== Performance Analysis Example ===")

    # Test different model architectures to showcase VLLM operations
    model_configs = [
        {
            "name": "facebook/opt-125m",
            "type": "Transformer LLM (OPT)",
            "description": "Standard transformer architecture",
            "max_tokens": 20,
        },
        {
            "name": "microsoft/DialoGPT-small",
            "type": "Transformer LLM (GPT-2 based)",
            "description": "Conversation-focused transformer",
            "max_tokens": 15,
        },
    ]

    # Additional models available for testing (commented out due to size/requirements)
    # Uncomment these if you have sufficient resources and the models are available:
    # advanced_models = [
    #     {
    #         "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    #         "type": "Mixture-of-Experts LLM",
    #         "description": "8 expert MoE architecture - showcases moe_flops",
    #         "max_tokens": 10,
    #     },
    #     {
    #         "name": "intfloat/e5-mistral-7b-instruct",
    #         "type": "Embedding Model",
    #         "description": "Embedding-focused model - different FLOP patterns",
    #         "max_tokens": 5,
    #     },
    #     {
    #         "name": "llava-hf/llava-1.5-7b-hf",
    #         "type": "Multi-modal LLM",
    #         "description": "Vision-language model - complex attention patterns",
    #         "max_tokens": 10,
    #     },
    # ]

    print("Testing different model architectures to showcase VLLM FLOP counting...")
    print(
        "💡 Tip: Uncomment advanced models in the code for MoE, "
        "embedding, and multimodal examples"
    )

    for config in model_configs:
        model_name = config["name"]
        model_type = config["type"]
        description = config["description"]
        max_tokens = config["max_tokens"]

        print(f"\n--- {model_type}: {model_name} ---")
        print(f"Description: {description}")

        try:
            llm = LLM(model=model_name, max_num_seqs=1)
            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

            with FlopContextManager() as flop_counter:
                start_time = time.time()
                outputs = llm.generate(
                    ["Explain artificial intelligence briefly"], sampling_params
                )
                end_time = time.time()

            total_flops = flop_counter.get_total_flops()
            elapsed_time = end_time - start_time
            gflops_per_sec = (
                total_flops / (elapsed_time * 1e9) if elapsed_time > 0 else 0
            )

            print(f"  Total FLOPs: {format_flops(total_flops)}")
            print(f"  Time: {elapsed_time:.3f}s")
            print(f"  GFLOPS/sec: {gflops_per_sec:.2f}")
            print(
                f"  Generated: {outputs[0].outputs[0].text[:100]}"
                f"{'...' if len(outputs[0].outputs[0].text) > 100 else ''}"
            )

            # Show detailed breakdown
            breakdown = flop_counter.get_flop_breakdown()
            if any(flops > 0 for flops in breakdown.values()):
                print("  FLOP Breakdown:")
                for category, flops in breakdown.items():
                    if flops > 0:
                        percentage = flops / total_flops * 100 if total_flops > 0 else 0
                        print(
                            f"    {category}: {format_flops(flops)} ({percentage:.1f}%)"
                        )

                if breakdown["mm_flops"] > total_flops * 0.7:
                    print(
                        "  Analysis: GEMM-heavy workload typical of "
                        "transformer inference"
                    )
                if breakdown["attention_flops"] > total_flops * 0.1:
                    print(
                        "  Analysis: Significant attention computation "
                        "(PagedAttention/custom kernels)"
                    )
                if breakdown["moe_flops"] > 0:
                    print("  Analysis: MoE operations detected - using expert routing")
                if breakdown["quantization_flops"] > 0:
                    print("  Analysis: Quantization operations active")

        except Exception as e:
            print(f"  Error with {model_name}: {e}")
            print(
                "     This might be due to model size, availability, "
                "or hardware requirements"
            )

    print("\nTo test more advanced model types:")
    print("   1. MoE Models (showcases moe_flops):")
    print("      - Uncomment Mixtral-8x7B in the code")
    print("      - Requires significant GPU memory")
    print("      - Will show moe_flops from expert routing")
    print("   2. Embedding Models:")
    print("      - Uncomment e5-mistral-7b-instruct")
    print("      - Different FLOP patterns optimized for embeddings")
    print("   3. Multi-modal Models:")
    print("      - Uncomment LLaVA model")
    print("      - Complex attention patterns from vision-text fusion")


def comprehensive_flop_analysis_example():
    """Comprehensive example showing FLOP analysis capabilities across
    different model types."""
    print("\n=== Comprehensive Multi-Model FLOP Analysis ===")

    # Define test scenarios for different model types
    test_scenarios = [
        {
            "name": "Standard Transformer",
            "model": "facebook/opt-125m",
            "task": "text_generation",
            "prompts": ["Explain machine learning"],
            "sampling_params": SamplingParams(temperature=0.0, max_tokens=25),
            "description": "Standard autoregressive text generation",
        },
        # Add more models here as needed - commented out for resource considerations
        # {
        #     "name": "MoE Model",
        #     "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        #     "task": "text_generation",
        #     "prompts": ["Explain AI"],
        #     "sampling_params": SamplingParams(temperature=0.0, max_tokens=15),
        #     "description": "Mixture-of-Experts with expert routing (shows moe_flops)"
        # },
        # {
        #     "name": "Embedding Model",
        #     "model": "intfloat/e5-mistral-7b-instruct",
        #     "task": "embedding",
        #     "prompts": ["Query: machine learning fundamentals"],
        #     "sampling_params": SamplingParams(temperature=0.0, max_tokens=1),
        #     "description": "Embedding extraction (different FLOP patterns)"
        # },
    ]

    print("Comparing FLOP patterns across different model architectures...")
    print(
        "Note: Uncomment additional models in the code to see MoE, "
        "embedding, and multimodal patterns\n"
    )

    results = []

    for scenario in test_scenarios:
        print(f"Testing: {scenario['name']}")
        print(f"Model: {scenario['model']}")
        print(f"Description: {scenario['description']}")

        try:
            llm = LLM(model=scenario["model"], max_num_seqs=len(scenario["prompts"]))

            with FlopContextManager() as flop_counter:
                start_time = time.time()
                outputs = llm.generate(scenario["prompts"], scenario["sampling_params"])
                end_time = time.time()

            total_flops = flop_counter.get_total_flops()
            elapsed_time = end_time - start_time
            breakdown = flop_counter.get_flop_breakdown()

            result = {
                "name": scenario["name"],
                "model": scenario["model"],
                "total_flops": total_flops,
                "time": elapsed_time,
                "gflops_per_sec": total_flops / (elapsed_time * 1e9)
                if elapsed_time > 0
                else 0,
                "breakdown": breakdown,
            }
            results.append(result)

            print("  Results:")
            print(f"    Total FLOPs: {format_flops(total_flops)}")
            print(f"    Time: {elapsed_time:.3f}s")
            print(f"    GFLOPS/sec: {result['gflops_per_sec']:.2f}")

            print("  FLOP Breakdown:")
            for category, flops in breakdown.items():
                if flops > 0:
                    percentage = flops / total_flops * 100
                    print(f"    {category}: {format_flops(flops)} ({percentage:.1f}%)")

            print("  Architecture Analysis:")
            if breakdown["mm_flops"] > total_flops * 0.8:
                print("    - GEMM-dominated: Typical transformer pattern")
            if breakdown["attention_flops"] > total_flops * 0.15:
                print("    - High attention compute: Complex attention patterns")
            if breakdown["moe_flops"] > 0:
                print("    - MoE detected: Expert routing active")
            if breakdown["quantization_flops"] > total_flops * 0.05:
                print("    - Quantization heavy: Using quantized operations")

            if outputs and len(outputs) > 0:
                sample_output = outputs[0].outputs[0].text[:80]
                print(
                    f"  Sample output: {sample_output}"
                    f"{'...' if len(outputs[0].outputs[0].text) > 80 else ''}"
                )

        except Exception as e:
            print(f"  Error: {e}")
            print("    (Model may require more resources or different setup)")

        print()

    if len(results) > 1:
        print("Comparative Analysis:")
        print("=" * 50)

        for result in results:
            print(f"\n{result['name']} ({result['model']}):")
            print(f"  FLOPs: {format_flops(result['total_flops'])}")
            print(f"  Performance: {result['gflops_per_sec']:.2f} GFLOPS/sec")

            breakdown = result["breakdown"]
            max_category = max(breakdown.items(), key=lambda x: x[1])
            if max_category[1] > 0:
                percentage = max_category[1] / result["total_flops"] * 100
                print(f"  Dominant ops: {max_category[0]} ({percentage:.1f}%)")

    print("\n" + "=" * 60)
    print("EXTENDED TESTING GUIDE")
    print("=" * 60)
    print("\n1. MoE Models (Mixture-of-Experts):")
    print("   - Uncomment Mixtral-8x7B-Instruct")
    print("   - Expected: High moe_flops from expert routing")
    print("   - Pattern: moe_wna16_gemm, topk_softmax operations")

    print("\n2. Embedding Models:")
    print("   - Uncomment e5-mistral-7b-instruct")
    print("   - Expected: Different mm_flops patterns")
    print("   - Pattern: Optimized for representation learning")

    print("\n3. Multi-modal Models:")
    print("   - Uncomment LLaVA-1.5-7b")
    print("   - Expected: Complex attention_flops patterns")
    print("   - Pattern: Vision-text cross-attention operations")

    print("\n4. Quantized Models:")
    print("   - Try AWQ/GPTQ quantized versions")
    print("   - Expected: High quantization_flops")
    print("   - Pattern: awq_gemm, gptq_marlin_gemm operations")

    print("\nEach model type will show different FLOP patterns,")
    print("demonstrating the comprehensive coverage of VLLM's FLOP counting!")


def main():
    parser = argparse.ArgumentParser(
        description="vLLM FLOP counting examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python flop_counting.py --basic          # Run basic FLOP counting example
  python flop_counting.py --profiling      # Run layerwise profiling example  
  python flop_counting.py --analysis       # Run performance analysis
  python flop_counting.py --comprehensive  # Run comprehensive analysis
  python flop_counting.py --all            # Run all examples
        """,
    )

    parser.add_argument(
        "--basic", action="store_true", help="Run basic FLOP counting example"
    )
    parser.add_argument(
        "--profiling",
        action="store_true",
        help="Run layerwise profiling with FLOPs example",
    )
    parser.add_argument(
        "--analysis", action="store_true", help="Run performance analysis example"
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive FLOP analysis example",
    )
    parser.add_argument("--all", action="store_true", help="Run all examples")

    args = parser.parse_args()

    if not any(
        [args.basic, args.profiling, args.analysis, args.comprehensive, args.all]
    ):
        # Default to basic example if no specific example is chosen
        args.basic = True

    print("VLLM FLOP Counting Examples")
    print("=" * 50)
    print("PyTorch FLOP counter with VLLM custom operations enabled")
    print()

    if args.basic or args.all:
        basic_flop_counting_example()

    if args.profiling or args.all:
        layerwise_profiling_with_flops_example()

    if args.analysis or args.all:
        performance_analysis_example()

    if args.comprehensive or args.all:
        comprehensive_flop_analysis_example()

    print("\n" + "=" * 50)
    print("FLOP counting examples completed!")


if __name__ == "__main__":
    main()
