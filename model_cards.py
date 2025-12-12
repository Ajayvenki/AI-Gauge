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
        architecture="Transformer (next-gen)",
        parameter_count="unknown",
        training_cutoff="2025-06",
        context_window=256_000,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_audio=True,
        supports_reasoning=True,
        latency_tier="medium",
        throughput_tier="medium",
        strengths=[
            "Best model for coding and agentic tasks",
            "State-of-the-art reasoning",
            "Excellent instruction following",
            "Multi-modal (text, vision, audio)",
            "Long context understanding",
        ],
        weaknesses=[
            "Highest cost tier",
            "Overkill for simple tasks",
            "Higher latency than smaller models",
        ],
        best_for=[
            "complex multi-step reasoning",
            "agentic workflows",
            "production code generation",
            "system design",
            "research synthesis",
        ],
        avoid_for=["simple Q&A", "boilerplate generation", "high-volume batch processing"],
        input_cost_per_1m=15.0,
        output_cost_per_1m=60.0,
        carbon_factor=5.0,
        status="stable",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-5.2-pro",
        display_name="GPT-5.2 Pro",
        family="GPT-5",
        architecture="Transformer (extended compute)",
        parameter_count="unknown",
        training_cutoff="2025-06",
        context_window=256_000,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_audio=True,
        supports_reasoning=True,
        latency_tier="slow",
        throughput_tier="low",
        strengths=[
            "Most precise responses",
            "Extended reasoning compute",
            "Best for mission-critical tasks",
        ],
        weaknesses=["Very expensive", "Slowest model", "Overkill for most tasks"],
        best_for=["mission-critical decisions", "legal/medical analysis", "complex math proofs"],
        avoid_for=["anything time-sensitive", "cost-sensitive applications"],
        input_cost_per_1m=30.0,
        output_cost_per_1m=120.0,
        carbon_factor=8.0,
        status="stable",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-5",
        display_name="GPT-5",
        family="GPT-5",
        architecture="Transformer (reasoning-optimized)",
        parameter_count="unknown",
        training_cutoff="2025-03",
        context_window=256_000,
        max_output_tokens=32_768,
        supports_vision=True,
        supports_reasoning=True,
        latency_tier="medium",
        throughput_tier="medium",
        strengths=[
            "Configurable reasoning effort",
            "Excellent for coding",
            "Strong agentic capabilities",
        ],
        weaknesses=["Superseded by GPT-5.2", "Still expensive"],
        best_for=["coding tasks", "agentic workflows", "complex reasoning"],
        avoid_for=["simple tasks", "budget-constrained projects"],
        input_cost_per_1m=10.0,
        output_cost_per_1m=40.0,
        carbon_factor=4.0,
        status="stable",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-5-mini",
        display_name="GPT-5 Mini",
        family="GPT-5",
        architecture="Transformer (distilled)",
        parameter_count="~20B (est.)",
        training_cutoff="2025-03",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_reasoning=True,
        latency_tier="fast",
        throughput_tier="high",
        strengths=[
            "Fast and cost-efficient",
            "Good reasoning for its size",
            "Well-suited for defined tasks",
        ],
        weaknesses=["Less capable on very complex reasoning"],
        best_for=["well-defined tasks", "moderate complexity coding", "structured data extraction"],
        avoid_for=["open-ended research", "highly complex reasoning"],
        input_cost_per_1m=3.0,
        output_cost_per_1m=12.0,
        carbon_factor=2.0,
        status="stable",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-5-nano",
        display_name="GPT-5 Nano",
        family="GPT-5",
        architecture="Transformer (highly distilled)",
        parameter_count="~8B (est.)",
        training_cutoff="2025-03",
        context_window=64_000,
        max_output_tokens=8_192,
        supports_vision=False,
        supports_reasoning=False,
        latency_tier="ultra-fast",
        throughput_tier="high",
        strengths=["Fastest GPT-5 variant", "Most cost-efficient", "High throughput"],
        weaknesses=["Limited reasoning", "Shorter context", "No vision"],
        best_for=["simple chat", "classification", "quick edits", "high-volume processing"],
        avoid_for=["complex reasoning", "long documents", "vision tasks"],
        input_cost_per_1m=0.5,
        output_cost_per_1m=2.0,
        carbon_factor=0.8,
        status="stable",
    ),
    # GPT-4 Family
    ModelCard(
        provider="openai",
        model_id="gpt-4.1",
        display_name="GPT-4.1",
        family="GPT-4",
        architecture="Transformer",
        parameter_count="~200B (rumored)",
        training_cutoff="2024-12",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        latency_tier="fast",
        throughput_tier="high",
        strengths=["Smartest non-reasoning model", "Excellent instruction following", "Fast"],
        weaknesses=["No explicit reasoning mode"],
        best_for=["general-purpose tasks", "code review", "content generation"],
        avoid_for=["tasks requiring explicit reasoning chains"],
        input_cost_per_1m=2.0,
        output_cost_per_1m=8.0,
        carbon_factor=2.5,
        status="stable",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-4.1-mini",
        display_name="GPT-4.1 Mini",
        family="GPT-4",
        architecture="Transformer (distilled)",
        parameter_count="~20B (est.)",
        training_cutoff="2024-12",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        latency_tier="fast",
        throughput_tier="high",
        strengths=["Good balance of speed and capability", "Cost-effective"],
        weaknesses=["Less capable than full GPT-4.1"],
        best_for=["routine tasks", "moderate complexity", "cost-sensitive apps"],
        avoid_for=["highly complex tasks"],
        input_cost_per_1m=0.4,
        output_cost_per_1m=1.6,
        carbon_factor=1.2,
        status="stable",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-4.1-nano",
        display_name="GPT-4.1 Nano",
        family="GPT-4",
        architecture="Transformer (highly distilled)",
        parameter_count="~8B (est.)",
        training_cutoff="2024-12",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        latency_tier="ultra-fast",
        throughput_tier="high",
        strengths=["Very fast", "Very cheap", "Good for simple tasks"],
        weaknesses=["Limited on complex tasks"],
        best_for=["simple Q&A", "classification", "quick summaries"],
        avoid_for=["complex reasoning", "nuanced writing"],
        input_cost_per_1m=0.1,
        output_cost_per_1m=0.4,
        carbon_factor=0.6,
        status="stable",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-4o",
        display_name="GPT-4o",
        family="GPT-4o",
        architecture="Transformer (omni)",
        parameter_count="~200B (rumored)",
        training_cutoff="2024-10",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        supports_audio=True,
        latency_tier="fast",
        throughput_tier="high",
        strengths=["Fast multimodal", "Good reasoning", "Reliable"],
        weaknesses=["Superseded by GPT-4.1 and GPT-5"],
        best_for=["multimodal tasks", "general-purpose", "vision analysis"],
        avoid_for=["tasks requiring latest capabilities"],
        input_cost_per_1m=2.5,
        output_cost_per_1m=10.0,
        carbon_factor=3.0,
        status="stable",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        family="GPT-4o",
        architecture="Transformer (distilled omni)",
        parameter_count="~8B (est.)",
        training_cutoff="2024-10",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=True,
        latency_tier="fast",
        throughput_tier="high",
        strengths=["Fast", "Affordable", "Good for focused tasks"],
        weaknesses=["Less capable on complex reasoning"],
        best_for=["simple Q&A", "classification", "quick edits", "bulk processing"],
        avoid_for=["complex multi-step reasoning"],
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
        carbon_factor=1.0,
        status="stable",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-3.5-turbo",
        display_name="GPT-3.5 Turbo",
        family="GPT-3.5",
        architecture="Transformer",
        parameter_count="~20B",
        training_cutoff="2023-09",
        context_window=16_385,
        max_output_tokens=4_096,
        supports_vision=False,
        latency_tier="ultra-fast",
        throughput_tier="high",
        strengths=["Very cheap", "Very fast", "Good for simple tasks"],
        weaknesses=["Outdated", "Weaker reasoning", "May hallucinate"],
        best_for=["boilerplate", "simple chat", "legacy systems"],
        avoid_for=["complex tasks", "accuracy-critical applications"],
        input_cost_per_1m=0.5,
        output_cost_per_1m=1.5,
        carbon_factor=1.0,  # baseline
        status="stable",
        notes="Legacy model, consider gpt-4o-mini instead",
    ),
    # O-series (Reasoning)
    ModelCard(
        provider="openai",
        model_id="o3",
        display_name="o3",
        family="O-series",
        architecture="Reasoning-optimized Transformer",
        parameter_count="unknown",
        training_cutoff="2024-12",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_reasoning=True,
        latency_tier="slow",
        throughput_tier="low",
        strengths=[
            "Advanced multi-step reasoning",
            "Chain-of-thought built-in",
            "Excellent for complex problems",
        ],
        weaknesses=["Slow", "Expensive", "Superseded by GPT-5"],
        best_for=["math proofs", "complex planning", "scientific reasoning"],
        avoid_for=["simple tasks", "real-time applications"],
        input_cost_per_1m=10.0,
        output_cost_per_1m=40.0,
        carbon_factor=5.0,
        status="stable",
        notes="Succeeded by GPT-5 for most use cases",
    ),
    ModelCard(
        provider="openai",
        model_id="o4-mini",
        display_name="o4-mini",
        family="O-series",
        architecture="Reasoning-optimized Transformer (distilled)",
        parameter_count="unknown",
        training_cutoff="2025-01",
        context_window=200_000,
        max_output_tokens=100_000,
        supports_vision=True,
        supports_reasoning=True,
        latency_tier="medium",
        throughput_tier="medium",
        strengths=["Fast reasoning", "Cost-efficient for reasoning tasks"],
        weaknesses=["Succeeded by GPT-5 mini"],
        best_for=["moderate reasoning tasks", "structured problem-solving"],
        avoid_for=["simple tasks that don't need reasoning"],
        input_cost_per_1m=1.1,
        output_cost_per_1m=4.4,
        carbon_factor=2.0,
        status="stable",
        notes="Succeeded by GPT-5 mini for most use cases",
    ),
    # Open-weight models
    ModelCard(
        provider="openai",
        model_id="gpt-oss-120b",
        display_name="GPT-OSS-120B",
        family="GPT-OSS",
        architecture="Transformer (open-weight)",
        parameter_count="120B",
        training_cutoff="2024-09",
        context_window=128_000,
        max_output_tokens=16_384,
        supports_vision=False,
        latency_tier="medium",
        throughput_tier="medium",
        strengths=["Open-weight", "Self-hostable", "No API costs"],
        weaknesses=["Requires significant compute", "No vision"],
        best_for=["self-hosting", "privacy-sensitive applications", "customization"],
        avoid_for=["teams without GPU infrastructure"],
        input_cost_per_1m=0.0,  # self-hosted
        output_cost_per_1m=0.0,
        carbon_factor=3.0,  # depends on infrastructure
        status="stable",
        notes="Apache 2.0 license, fits on H100 GPU",
    ),
    ModelCard(
        provider="openai",
        model_id="gpt-oss-20b",
        display_name="GPT-OSS-20B",
        family="GPT-OSS",
        architecture="Transformer (open-weight, small)",
        parameter_count="20B",
        training_cutoff="2024-09",
        context_window=64_000,
        max_output_tokens=8_192,
        supports_vision=False,
        latency_tier="fast",
        throughput_tier="high",
        strengths=["Open-weight", "Low latency", "Runs on consumer GPUs"],
        weaknesses=["Less capable than larger models"],
        best_for=["local development", "edge deployment", "low-latency apps"],
        avoid_for=["complex reasoning tasks"],
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        carbon_factor=1.0,
        status="stable",
        notes="Apache 2.0 license",
    ),
]

# ---------------------------------------------------------------------------
# Anthropic (Claude) Models
# ---------------------------------------------------------------------------

ANTHROPIC_MODELS: List[ModelCard] = [
    # Claude 4 Family
    ModelCard(
        provider="anthropic",
        model_id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        family="Claude 4",
        architecture="Constitutional AI (extended)",
        parameter_count="~200B (est.)",
        training_cutoff="2025-03",
        context_window=200_000,
        max_output_tokens=32_000,
        supports_vision=True,
        supports_reasoning=True,
        latency_tier="slow",
        throughput_tier="low",
        strengths=[
            "Top-tier reasoning",
            "Excellent for complex agentic tasks",
            "Best Claude for research",
            "Unique 'extended thinking' mode",
            "Strong code generation",
        ],
        weaknesses=["Expensive", "Slow", "Overkill for simple tasks"],
        best_for=["complex research", "agentic loops", "mission-critical code", "long analysis"],
        avoid_for=["simple Q&A", "high-volume processing", "real-time apps"],
        input_cost_per_1m=15.0,
        output_cost_per_1m=75.0,
        carbon_factor=5.0,
        status="stable",
    ),
    ModelCard(
        provider="anthropic",
        model_id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        family="Claude 4",
        architecture="Constitutional AI",
        parameter_count="~70B (est.)",
        training_cutoff="2025-03",
        context_window=200_000,
        max_output_tokens=16_000,
        supports_vision=True,
        supports_reasoning=True,
        latency_tier="medium",
        throughput_tier="medium",
        strengths=[
            "Excellent balance of capability and speed",
            "Strong instruction-following",
            "Good for coding",
            "Large context window",
        ],
        weaknesses=["Not as capable as Opus on hardest tasks"],
        best_for=["document analysis", "code generation", "long-form writing", "general-purpose"],
        avoid_for=["budget-critical high-volume processing"],
        input_cost_per_1m=3.0,
        output_cost_per_1m=15.0,
        carbon_factor=2.5,
        status="stable",
    ),
    ModelCard(
        provider="anthropic",
        model_id="claude-haiku-4-20250514",
        display_name="Claude Haiku 4",
        family="Claude 4",
        architecture="Constitutional AI (distilled)",
        parameter_count="~8B (est.)",
        training_cutoff="2025-03",
        context_window=200_000,
        max_output_tokens=8_000,
        supports_vision=True,
        latency_tier="ultra-fast",
        throughput_tier="high",
        strengths=["Very fast", "Very cheap", "Good for high-volume", "Still has large context"],
        weaknesses=["Less nuanced reasoning", "Simpler outputs"],
        best_for=["quick edits", "classification", "simple summaries", "batch processing"],
        avoid_for=["complex reasoning", "nuanced writing"],
        input_cost_per_1m=0.25,
        output_cost_per_1m=1.25,
        carbon_factor=0.8,
        status="stable",
    ),
    # Claude 3.5 Family
    ModelCard(
        provider="anthropic",
        model_id="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        family="Claude 3.5",
        architecture="Constitutional AI",
        parameter_count="~70B (est.)",
        training_cutoff="2024-08",
        context_window=200_000,
        max_output_tokens=8_192,
        supports_vision=True,
        latency_tier="fast",
        throughput_tier="high",
        strengths=["Fast", "Reliable", "Good coding", "Strong instruction-following"],
        weaknesses=["Superseded by Claude 4 Sonnet"],
        best_for=["code generation", "document analysis", "general tasks"],
        avoid_for=["tasks requiring latest capabilities"],
        input_cost_per_1m=3.0,
        output_cost_per_1m=15.0,
        carbon_factor=2.5,
        status="stable",
        notes="Excellent workhorse model",
    ),
    ModelCard(
        provider="anthropic",
        model_id="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        family="Claude 3.5",
        architecture="Constitutional AI (distilled)",
        parameter_count="~8B (est.)",
        training_cutoff="2024-08",
        context_window=200_000,
        max_output_tokens=8_192,
        supports_vision=True,
        latency_tier="ultra-fast",
        throughput_tier="high",
        strengths=["Fastest Claude", "Cheapest Claude", "Good for high-volume"],
        weaknesses=["Less nuanced reasoning"],
        best_for=["quick edits", "classification", "simple summaries"],
        avoid_for=["complex reasoning", "nuanced tasks"],
        input_cost_per_1m=0.80,
        output_cost_per_1m=4.0,
        carbon_factor=1.0,
        status="stable",
    ),
    # Claude 3 Family
    ModelCard(
        provider="anthropic",
        model_id="claude-3-opus-20240229",
        display_name="Claude 3 Opus",
        family="Claude 3",
        architecture="Constitutional AI",
        parameter_count="~200B (est.)",
        training_cutoff="2024-02",
        context_window=200_000,
        max_output_tokens=4_096,
        supports_vision=True,
        latency_tier="slow",
        throughput_tier="low",
        strengths=["Very capable", "Strong reasoning"],
        weaknesses=["Expensive", "Superseded by Claude 4 Opus"],
        best_for=["complex analysis", "research"],
        avoid_for=["simple tasks", "cost-sensitive apps"],
        input_cost_per_1m=15.0,
        output_cost_per_1m=75.0,
        carbon_factor=5.0,
        status="stable",
        notes="Consider Claude 4 Opus for new projects",
    ),
]

# ---------------------------------------------------------------------------
# Google (Gemini) Models
# ---------------------------------------------------------------------------

GOOGLE_MODELS: List[ModelCard] = [
    # Gemini 3 Family
    ModelCard(
        provider="google",
        model_id="gemini-3-pro",
        display_name="Gemini 3 Pro",
        family="Gemini 3",
        architecture="Mixture of Experts (MoE)",
        parameter_count="unknown",
        training_cutoff="2025-06",
        context_window=2_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_audio=True,
        supports_video=True,
        supports_reasoning=True,
        latency_tier="medium",
        throughput_tier="medium",
        strengths=[
            "Best-in-class multimodal understanding",
            "Massive 2M context window",
            "Excellent for agentic and vibe-coding",
            "State-of-the-art reasoning",
            "Video understanding",
        ],
        weaknesses=["Higher cost", "Newer model, less battle-tested"],
        best_for=[
            "very long documents",
            "video analysis",
            "multimodal tasks",
            "agentic workflows",
            "code understanding",
        ],
        avoid_for=["simple tasks", "cost-sensitive high-volume"],
        input_cost_per_1m=7.0,
        output_cost_per_1m=21.0,
        carbon_factor=4.0,
        status="preview",
    ),
    # Gemini 2.5 Family
    ModelCard(
        provider="google",
        model_id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        family="Gemini 2.5",
        architecture="Mixture of Experts (MoE)",
        parameter_count="unknown",
        training_cutoff="2025-03",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_audio=True,
        supports_video=True,
        supports_reasoning=True,
        latency_tier="medium",
        throughput_tier="medium",
        strengths=[
            "State-of-the-art thinking model",
            "Excellent for complex reasoning",
            "1M context window",
            "Great for STEM, code, math",
        ],
        weaknesses=["Slower than Flash", "Higher cost"],
        best_for=["complex reasoning", "long document analysis", "code understanding", "STEM"],
        avoid_for=["simple tasks", "latency-sensitive apps"],
        input_cost_per_1m=2.5,
        output_cost_per_1m=15.0,
        carbon_factor=3.5,
        status="stable",
    ),
    ModelCard(
        provider="google",
        model_id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        family="Gemini 2.5",
        architecture="Mixture of Experts (MoE, optimized)",
        parameter_count="unknown",
        training_cutoff="2025-03",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        supports_audio=True,
        supports_video=True,
        supports_reasoning=True,
        latency_tier="fast",
        throughput_tier="high",
        strengths=[
            "Best price-performance ratio",
            "Fast with thinking capabilities",
            "1M context window",
            "Good for agentic tasks",
        ],
        weaknesses=["Not as capable as Pro on hardest problems"],
        best_for=["large-scale processing", "low-latency apps", "agentic use cases", "general-purpose"],
        avoid_for=["mission-critical complex reasoning"],
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
        carbon_factor=1.5,
        status="stable",
    ),
    ModelCard(
        provider="google",
        model_id="gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash-Lite",
        family="Gemini 2.5",
        architecture="Mixture of Experts (MoE, highly optimized)",
        parameter_count="unknown",
        training_cutoff="2025-03",
        context_window=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        latency_tier="ultra-fast",
        throughput_tier="high",
        strengths=["Ultra-fast", "Most cost-efficient Gemini", "High throughput"],
        weaknesses=["Less capable than Flash"],
        best_for=["high-volume batch processing", "simple tasks", "real-time apps"],
        avoid_for=["complex reasoning"],
        input_cost_per_1m=0.075,
        output_cost_per_1m=0.30,
        carbon_factor=0.8,
        status="stable",
    ),
    # Gemini 2.0 Family
    ModelCard(
        provider="google",
        model_id="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        family="Gemini 2.0",
        architecture="Mixture of Experts (MoE)",
        parameter_count="unknown",
        training_cutoff="2024-10",
        context_window=1_000_000,
        max_output_tokens=8_192,
        supports_vision=True,
        supports_audio=True,
        latency_tier="fast",
        throughput_tier="high",
        strengths=["Reliable workhorse", "1M context", "Good multimodal"],
        weaknesses=["Superseded by 2.5 Flash"],
        best_for=["general tasks", "document processing"],
        avoid_for=["tasks needing latest capabilities"],
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.40,
        carbon_factor=1.5,
        status="stable",
        notes="Consider 2.5 Flash for new projects",
    ),
    ModelCard(
        provider="google",
        model_id="gemini-2.0-flash-lite",
        display_name="Gemini 2.0 Flash-Lite",
        family="Gemini 2.0",
        architecture="Mixture of Experts (MoE, optimized)",
        parameter_count="unknown",
        training_cutoff="2024-10",
        context_window=1_000_000,
        max_output_tokens=8_192,
        supports_vision=True,
        latency_tier="ultra-fast",
        throughput_tier="high",
        strengths=["Very fast", "Very cheap", "1M context"],
        weaknesses=["Superseded by 2.5 Flash-Lite"],
        best_for=["simple tasks", "high-volume"],
        avoid_for=["complex tasks"],
        input_cost_per_1m=0.075,
        output_cost_per_1m=0.30,
        carbon_factor=0.8,
        status="stable",
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
