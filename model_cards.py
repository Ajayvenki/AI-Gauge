"""Model Cards — Comprehensive reference data for LLM selection.

Contains detailed information about all major models from OpenAI, Anthropic (Claude),
and Google (Gemini) including architecture, context windows, output limits, strengths,
weaknesses, best-use cases, and carbon factors.

The decision module consults this registry to recommend the optimal model for a given task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass
class ModelCard:
    """Comprehensive metadata for a single LLM."""

    # Identity
    provider: str  # "openai", "anthropic", "google"
    model_id: str  # API model name
    display_name: str
    family: str  # e.g., "GPT-5", "Claude 4", "Gemini 3"

    # Architecture & Size
    architecture: str  # e.g., "Transformer", "MoE", "Hybrid"
    parameter_count: str  # e.g., "~200B", "~70B", "unknown"
    training_cutoff: str  # e.g., "2024-10", "unknown"

    # Token Limits
    context_window: int  # max input tokens
    max_output_tokens: int  # max output tokens

    # Capabilities
    supports_vision: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    supports_function_calling: bool = True
    supports_structured_output: bool = True
    supports_streaming: bool = True
    supports_reasoning: bool = False  # explicit CoT/reasoning mode

    # Performance characteristics
    latency_tier: str = "medium"  # "ultra-fast", "fast", "medium", "slow"
    throughput_tier: str = "medium"  # "high", "medium", "low"

    # Detailed capabilities
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    best_for: List[str] = field(default_factory=list)
    avoid_for: List[str] = field(default_factory=list)

    # Cost & Carbon
    input_cost_per_1m: float = 0.0  # USD per 1M input tokens
    output_cost_per_1m: float = 0.0  # USD per 1M output tokens
    carbon_factor: float = 1.0  # relative multiplier vs baseline (gpt-3.5)

    # Status
    status: str = "stable"  # "stable", "preview", "deprecated", "experimental"
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# OpenAI Models
# ---------------------------------------------------------------------------

OPENAI_MODELS: List[ModelCard] = [
    # GPT-5 Family (Frontier)
    ModelCard(
        provider="openai",
        model_id="gpt-5.2",
        display_name="GPT-5.2",
        family="GPT-5",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2025-08-31",
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_audio=False,
        supports_reasoning=True,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=[
            "Flagship model for coding and agentic tasks",
        ],
        weaknesses=[
            "unknown",
        ],
        best_for=[
            "unknown",
        ],
        avoid_for=["unknown"],
        input_cost_per_1m=1.75,
        output_cost_per_1m=14.0,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.openai.com/docs/models/gpt-5.2 (image input only; audio/video not supported).",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-5.2-pro",
        display_name="GPT-5.2 Pro",
        family="GPT-5",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2025-08-31",
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_audio=False,
        supports_reasoning=True,
        supports_structured_output=False,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=[
            "Version of GPT-5.2 that produces smarter and more precise responses",
        ],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=21.0,
        output_cost_per_1m=168.0,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.openai.com/docs/models/gpt-5.2-pro (Responses API only; structured outputs not supported).",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-5",
        display_name="GPT-5",
        family="GPT-5",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2024-09-30",
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_reasoning=True,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=[
            "Previous model for coding, reasoning, and agentic tasks",
        ],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=1.25,
        output_cost_per_1m=10.0,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.openai.com/docs/models/gpt-5 (image input only; audio/video not supported).",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-5-mini",
        display_name="GPT-5 Mini",
        family="GPT-5",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2024-05-31",
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_reasoning=True,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Faster, more cost-efficient version of GPT-5 for well-defined tasks"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=0.25,
        output_cost_per_1m=2.0,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.openai.com/docs/models/gpt-5-mini (image input only; audio/video not supported).",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-5-nano",
        display_name="GPT-5 Nano",
        family="GPT-5",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2024-05-31",
        context_window=400_000,
        max_output_tokens=128_000,
        supports_vision=True,
        supports_reasoning=True,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Fastest, most cost-efficient version of GPT-5"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=0.05,
        output_cost_per_1m=0.40,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.openai.com/docs/models/gpt-5-nano (image input only; audio/video not supported).",
    ),
    # GPT-4 Family
    ModelCard(
        provider="openai",
        model_id="gpt-4.1",
        display_name="GPT-4.1",
        family="GPT-4",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2024-06-01",
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_audio=False,
        supports_video=False,
        supports_function_calling=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_reasoning=False,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Instruction following and tool calling"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=2.0,
        output_cost_per_1m=8.0,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.openai.com/docs/models/gpt-4.1 (image input only; audio/video not supported).",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-4.1-mini",
        display_name="GPT-4.1 Mini",
        family="GPT-4",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2024-06-01",
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_audio=False,
        supports_video=False,
        supports_function_calling=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_reasoning=False,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Smaller, faster version of GPT-4.1"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=0.4,
        output_cost_per_1m=1.6,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.openai.com/docs/models/gpt-4.1-mini (image input only; audio/video not supported).",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-4.1-nano",
        display_name="GPT-4.1 Nano",
        family="GPT-4",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2024-06-01",
        context_window=1_047_576,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_audio=False,
        supports_video=False,
        supports_function_calling=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_reasoning=False,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Fastest, most cost-efficient version of GPT-4.1"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=0.1,
        output_cost_per_1m=0.4,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.openai.com/docs/models/gpt-4.1-nano (image input only; audio/video not supported).",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-4o",
        display_name="GPT-4o",
        family="GPT-4o",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2023-10-01",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_audio=False,
        supports_video=False,
        supports_function_calling=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_reasoning=False,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Versatile, high-intelligence model"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=2.5,
        output_cost_per_1m=10.0,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.openai.com/docs/models/gpt-4o (image input only; audio/video not supported).",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        family="GPT-4o",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2023-10-01",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_audio=False,
        supports_video=False,
        supports_function_calling=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_reasoning=False,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Fast, affordable small model for focused tasks"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.openai.com/docs/models/gpt-4o-mini (image input only; audio/video not supported).",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-3.5-turbo",
        display_name="GPT-3.5 Turbo",
        family="GPT-3.5",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2021-09-01",
        context_window=16_385,
        max_output_tokens=4_096,
        supports_vision=False,
        supports_audio=False,
        supports_video=False,
        supports_function_calling=False,
        supports_structured_output=False,
        supports_streaming=False,
        supports_reasoning=False,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Legacy GPT model"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=0.5,
        output_cost_per_1m=1.5,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.openai.com/docs/models/gpt-3.5-turbo (streaming/function calling/structured outputs not supported on the model page).",
    ),
    # O-series (Reasoning)
    ModelCard(
        provider="openai",
        model_id="o3",
        display_name="o3",
        family="O-series",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2024-06-01",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_reasoning=True,
        supports_audio=False,
        supports_video=False,
        supports_function_calling=True,
        supports_structured_output=True,
        supports_streaming=True,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=[
            "Well-rounded reasoning model across domains",
        ],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=2.0,
        output_cost_per_1m=8.0,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.openai.com/docs/models/o3 (image input only; audio/video not supported).",
    ),
    ModelCard(
        provider="openai",
        model_id="o4-mini",
        display_name="o4-mini",
        family="O-series",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2024-06-01",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_reasoning=True,
        supports_audio=False,
        supports_video=False,
        supports_function_calling=True,
        supports_structured_output=True,
        supports_streaming=True,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Latest small o-series model optimized for fast reasoning"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=1.1,
        output_cost_per_1m=4.4,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.openai.com/docs/models/o4-mini (image input only; audio/video not supported).",
    ),
    # Open-weight models
    ModelCard(
        provider="openai",
        model_id="gpt-oss-120b",
        display_name="GPT-OSS-120B",
        family="GPT-OSS",
        architecture="unknown",
        parameter_count="117B parameters (5.1B active)",
        training_cutoff="2024-06-01",
        context_window=131_072,
        max_output_tokens=131_072,
        supports_vision=False,
        supports_audio=False,
        supports_video=False,
        supports_function_calling=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_reasoning=True,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Open-weight", "Self-hostable", "No API costs"],
        weaknesses=["Requires significant compute", "No vision"],
        best_for=["self-hosting", "privacy-sensitive applications", "customization"],
        avoid_for=["teams without GPU infrastructure"],
        input_cost_per_1m=0.0,  # self-hosted
        output_cost_per_1m=0.0,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.openai.com/docs/models/gpt-oss-120b (Apache 2.0; 131,072 context/output).",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-oss-20b",
        display_name="GPT-OSS-20B",
        family="GPT-OSS",
        architecture="unknown",
        parameter_count="21B parameters (3.6B active)",
        training_cutoff="2024-06-01",
        context_window=131_072,
        max_output_tokens=131_072,
        supports_vision=False,
        supports_audio=False,
        supports_video=False,
        supports_function_calling=True,
        supports_structured_output=True,
        supports_streaming=True,
        supports_reasoning=True,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Open-weight", "Low latency", "Runs on consumer GPUs"],
        weaknesses=["Less capable than larger models"],
        best_for=["local development", "edge deployment", "low-latency apps"],
        avoid_for=["complex reasoning tasks"],
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.openai.com/docs/models/gpt-oss-20b (Apache 2.0; 131,072 context/output).",
    ),
]

# ---------------------------------------------------------------------------
# Anthropic (Claude) Models
# Source: https://platform.claude.com/docs/en/about-claude/models/all-models
# ---------------------------------------------------------------------------

ANTHROPIC_MODELS: List[ModelCard] = [
    # Claude 4.5 Family (Latest)
    ModelCard(
        provider="anthropic",
        model_id="claude-opus-4-5-20251101",
        display_name="Claude Opus 4.5",
        family="Claude 4.5",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2025-05",  # Reliable knowledge cutoff: May 2025
        context_window=200_000,
        max_output_tokens=64_000,
        supports_vision=True,
        supports_reasoning=True,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Premium model combining maximum intelligence with practical performance"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=5.0,
        output_cost_per_1m=25.0,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.claude.com/docs/en/about-claude/models/all-models (200K context, 64K output, training data cutoff Aug 2025).",
    ),
    ModelCard(
        provider="anthropic",
        model_id="claude-sonnet-4-5-20250929",
        display_name="Claude Sonnet 4.5",
        family="Claude 4.5",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2025-01",  # Reliable knowledge cutoff: Jan 2025
        context_window=200_000,  # 1M with beta header
        max_output_tokens=64_000,
        supports_vision=True,
        supports_reasoning=True,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Smart model for complex agents and coding"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=3.0,
        output_cost_per_1m=15.0,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.claude.com/docs/en/about-claude/models/all-models (200K context, 64K output, 1M beta available).",
    ),
    ModelCard(
        provider="anthropic",
        model_id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        family="Claude 4.5",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2025-02",  # Reliable knowledge cutoff: Feb 2025
        context_window=200_000,
        max_output_tokens=64_000,
        supports_vision=True,
        supports_reasoning=True,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Fastest model with near-frontier intelligence"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=1.0,
        output_cost_per_1m=5.0,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://platform.claude.com/docs/en/about-claude/models/all-models (200K context, 64K output).",
    ),
    # Claude 3.5 Family (Legacy but still supported)
    ModelCard(
        provider="anthropic",
        model_id="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        family="Claude 3.5",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2024-08",
        context_window=200_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_reasoning=False,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["unknown"],
        weaknesses=["Superseded by Claude 4.5 Sonnet"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=3.0,
        output_cost_per_1m=15.0,
        carbon_factor=0.0,
        status="stable",
        notes="Legacy model. Consider Claude 4.5 Sonnet for new projects.",
    ),
    ModelCard(
        provider="anthropic",
        model_id="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        family="Claude 3.5",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2024-08",
        context_window=200_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_reasoning=False,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["unknown"],
        weaknesses=["Superseded by Claude 4.5 Haiku"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=0.80,
        output_cost_per_1m=4.0,
        carbon_factor=0.0,
        status="stable",
        notes="Legacy model. Consider Claude 4.5 Haiku for new projects.",
    ),
]

# ---------------------------------------------------------------------------
# Google (Gemini) Models
# Source: https://ai.google.dev/gemini-api/docs/pricing, https://ai.google.dev/gemini-api/docs/gemini-3
# ---------------------------------------------------------------------------

GOOGLE_MODELS: List[ModelCard] = [
    # Gemini 3 Family (Latest)
    ModelCard(
        provider="google",
        model_id="gemini-3-pro-preview",
        display_name="Gemini 3 Pro Preview",
        family="Gemini 3",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="2025-01",  # Knowledge cutoff: Jan 2025
        context_window=1_000_000,  # 1M input
        max_output_tokens=64_000,  # 64k output
        supports_vision=True,
        supports_audio=True,
        supports_video=True,
        supports_reasoning=True,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Best model for multimodal understanding", "Most powerful agentic and vibe-coding model"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=2.0,  # ≤200k tokens
        output_cost_per_1m=12.0,  # ≤200k tokens
        carbon_factor=0.0,
        status="preview",
        notes="Source: https://ai.google.dev/gemini-api/docs/gemini-3 ($4/$18 for >200k tokens).",
    ),
    # Gemini 2.5 Family
    ModelCard(
        provider="google",
        model_id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        family="Gemini 2.5",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="unknown",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_audio=True,
        supports_video=True,
        supports_reasoning=True,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["State-of-the-art multipurpose model", "Excels at coding and complex reasoning"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=1.25,  # ≤200k tokens
        output_cost_per_1m=10.0,  # ≤200k tokens
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://ai.google.dev/gemini-api/docs/pricing ($2.50/$15 for >200k tokens).",
    ),
    ModelCard(
        provider="google",
        model_id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        family="Gemini 2.5",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="unknown",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_audio=True,
        supports_video=True,
        supports_reasoning=True,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["First hybrid reasoning model", "1M context with thinking budgets"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=0.30,
        output_cost_per_1m=2.50,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://ai.google.dev/gemini-api/docs/pricing.",
    ),
    ModelCard(
        provider="google",
        model_id="gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash-Lite",
        family="Gemini 2.5",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="unknown",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_audio=False,
        supports_video=False,
        supports_reasoning=False,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Smallest and most cost effective model", "Built for at scale usage"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.40,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://ai.google.dev/gemini-api/docs/pricing.",
    ),
    # Gemini 2.0 Family
    ModelCard(
        provider="google",
        model_id="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        family="Gemini 2.0",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="unknown",
        context_window=1_000_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_audio=True,
        supports_video=True,
        supports_reasoning=False,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Most balanced multimodal model", "1M token context", "Built for agents"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.40,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://ai.google.dev/gemini-api/docs/pricing. Consider 2.5 Flash for new projects.",
    ),
    ModelCard(
        provider="google",
        model_id="gemini-2.0-flash-lite",
        display_name="Gemini 2.0 Flash-Lite",
        family="Gemini 2.0",
        architecture="unknown",
        parameter_count="unknown",
        training_cutoff="unknown",
        context_window=1_000_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_audio=False,
        supports_video=False,
        supports_reasoning=False,
        latency_tier="unknown",
        throughput_tier="unknown",
        strengths=["Smallest and most cost effective model"],
        weaknesses=["unknown"],
        best_for=["unknown"],
        avoid_for=["unknown"],
        input_cost_per_1m=0.075,
        output_cost_per_1m=0.30,
        carbon_factor=0.0,
        status="stable",
        notes="Source: https://ai.google.dev/gemini-api/docs/pricing.",
    ),
]

# ---------------------------------------------------------------------------
# Aggregated Registry
# ---------------------------------------------------------------------------

MODEL_CARDS: List[ModelCard] = OPENAI_MODELS + ANTHROPIC_MODELS + GOOGLE_MODELS


def get_model_card(model_id: str) -> ModelCard | None:
    """Return the ModelCard for a given model_id, or None if not found."""
    for card in MODEL_CARDS:
        if card.model_id == model_id:
            return card
    return None


def list_models_by_provider(provider: str) -> List[ModelCard]:
    """Return all models from a specific provider."""
    return [c for c in MODEL_CARDS if c.provider == provider]


def list_models_for_use_case(use_case: str) -> List[ModelCard]:
    """Return models whose best_for list contains a keyword match for use_case."""
    use_case_lower = use_case.lower()
    return [c for c in MODEL_CARDS if any(use_case_lower in bf.lower() for bf in c.best_for)]


def list_models_by_latency(tier: str) -> List[ModelCard]:
    """Return models matching a latency tier: ultra-fast, fast, medium, slow."""
    return [c for c in MODEL_CARDS if c.latency_tier == tier]


def get_cheapest_capable_model(
    min_context: int = 0,
    needs_vision: bool = False,
    needs_reasoning: bool = False,
) -> ModelCard | None:
    """Find the cheapest model meeting the given requirements."""
    candidates = [
        c
        for c in MODEL_CARDS
        if c.context_window >= min_context
        and (not needs_vision or c.supports_vision)
        and (not needs_reasoning or c.supports_reasoning)
        and c.status in ("stable", "preview")
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda c: c.input_cost_per_1m + c.output_cost_per_1m)


def format_model_card(card: ModelCard) -> str:
    """Format a model card as a human-readable string."""
    lines = [
        f"**{card.display_name}** ({card.provider}/{card.model_id})",
        f"  Family: {card.family} | Architecture: {card.architecture}",
        f"  Context: {card.context_window:,} tokens | Max Output: {card.max_output_tokens:,} tokens",
        f"  Latency: {card.latency_tier} | Throughput: {card.throughput_tier}",
        f"  Vision: {card.supports_vision} | Audio: {card.supports_audio} | Reasoning: {card.supports_reasoning}",
        f"  Cost: ${card.input_cost_per_1m}/1M in, ${card.output_cost_per_1m}/1M out",
        f"  Carbon Factor: {card.carbon_factor}x baseline",
        f"  Best for: {', '.join(card.best_for)}",
        f"  Avoid for: {', '.join(card.avoid_for)}",
    ]
    if card.notes:
        lines.append(f"  Note: {card.notes}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"=== Model Registry: {len(MODEL_CARDS)} models ===\n")

    for provider in ["openai", "anthropic", "google"]:
        models = list_models_by_provider(provider)
        print(f"\n--- {provider.upper()} ({len(models)} models) ---\n")
        for card in models:
            print(f"  {card.model_id}: {card.display_name} (carbon: {card.carbon_factor}x)")

    print("\n\n=== Cheapest model with 100k context and vision ===")
    cheapest = get_cheapest_capable_model(min_context=100_000, needs_vision=True)
    if cheapest:
        print(format_model_card(cheapest))
