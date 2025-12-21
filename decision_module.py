"""
AI-Gauge Decision Module - LangGraph 3-Agent System

This module is designed to be called by an IDE plugin that INTERCEPTS LLM API calls
before they execute. The plugin automatically extracts all metadata from the call itself.

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │                    IDE Plugin (VS Code Extension)                │
  │                                                                  │
  │  Intercepts:  client.responses.create(model="<any-model>", ...) │
  │  Extracts:    model, prompt, system_prompt, tools, context      │
  │  No user input required - fully automatic                        │
  └─────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    LangGraph Decision Pipeline                   │
  │                                                                  │
  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
  │   │   Agent 1    │───▶│   Agent 2    │───▶│   Agent 3    │      │
  │   │  Metadata    │    │  Analyzer    │    │   Reporter   │      │
  │   │  Extractor   │    │  (Validate)  │    │              │      │
  │   └──────────────┘    └──────────────┘    └──────────────┘      │
  │                                                                  │
  └─────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  Output (only if model is INAPPROPRIATE for the task):          │
  │                                                                  │
  │  "Your task is simple text editing. A frontier model is overkill│
  │   GPT-4o-mini can handle this because [reasoning].               │
  │   Switch to save 94% cost and reduce carbon footprint."         │
  └─────────────────────────────────────────────────────────────────┘

Agents:
  1. Metadata Extractor: Extracts raw metadata ONLY (no analysis, no recommendations)
  2. Analyzer: Uses LOCAL FINE-TUNED PHI-3.5 MODEL to assess if model choice is appropriate. 
               Only suggests alternatives IF the model is overkill/inappropriate.
  3. Reporter: Generates human-readable explanation with reasoning.
"""

import os
import json
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Import local inference module for fine-tuned model
try:
    from local_inference import (
        analyze_with_local_model,
        is_local_model_available,
        USE_LOCAL_MODEL
    )
    LOCAL_INFERENCE_AVAILABLE = True
except ImportError:
    LOCAL_INFERENCE_AVAILABLE = False
    USE_LOCAL_MODEL = False

# LangGraph imports
from langgraph.graph import StateGraph, END

# Import our model cards
from model_cards import (
    MODEL_CARDS,
    list_models_by_provider,
    get_model_card,
    TIER_RANKINGS,
    get_tier_rank,
    is_model_overkill_by_tier
)

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """
    Shared state passed between all agents in the pipeline.
    
    The IDE plugin populates the input fields by intercepting the LLM API call.
    Users don't need to provide any metadata manually.
    """
    # Input (auto-extracted by IDE plugin from the intercepted API call)
    original_model: str          # The model ID from the API call
    original_prompt: str         # The user prompt/content
    system_prompt: str           # System prompt if any
    context: Dict[str, Any]      # Any context object passed
    tools: List[str]             # Tools/functions being used
    reasoning_effort: str        # Reasoning effort level (none, low, medium, high, max)
    model_config: Dict[str, Any] # Model configuration (temperature, max_tokens, etc.)
    conversation_history: List[Dict[str, str]]  # Previous conversation turns
    
    # Agent 1 output: Comprehensive metadata extraction (NO analysis, NO recommendations)
    metadata: Dict[str, Any]
    
    # Agent 2 output: Analysis results
    is_model_appropriate: bool           # Key decision: is the model a good fit?
    appropriateness_reasoning: str       # Why it is/isn't appropriate
    current_model_info: Dict[str, Any]
    alternative_models: List[Dict[str, Any]]  # Multiple alternatives if inappropriate
    carbon_analysis: Dict[str, Any]
    
    # Agent 3 output: Review and Report
    recommendation: Dict[str, Any]       # Contains multiple recommended models
    human_readable_summary: str          # Detailed report with all alternatives
    review_verdict: str                  # APPROPRIATE, OVERKILL, or UNDERPOWERED
    
    # Debug/trace
    messages: List[Dict[str, Any]]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ≈ 1 token for English text)."""
    return max(1, len(text) // 4)


def calculate_carbon_cost(
    input_tokens: int,
    output_tokens: int,
    model_card: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate carbon cost using research-backed methodology.
    
    Based on:
    - CodeCarbon (mlco2.github.io) methodology
    - "Power Hungry Processing" paper by Luccioni et al. (2024)
    
    Formula:
        CO2eq (g) = Energy (kWh) × Carbon Intensity (gCO2/kWh) × PUE
    
    Where:
        Energy = (GPU Power (W) × Inference Time (s)) / 3600 / 1000
        Inference Time ≈ Total Tokens / Throughput (tokens/sec)
    
    Since providers don't publish exact hardware specs, we estimate using:
    - Model size category → Estimated GPU power draw
    - Latency tier → Estimated throughput
    - Provider defaults for data center carbon intensity
    
    Returns a dict with breakdown for transparency.
    """
    total_tokens = input_tokens + output_tokens
    
    # Provider-specific carbon intensity estimates (gCO2/kWh)
    # Based on their renewable energy commitments and data center locations
    PROVIDER_CARBON_INTENSITY = {
        'openai': 200,      # Azure datacenters, partial renewables
        'anthropic': 180,   # GCP/AWS mix, good renewable coverage
        'google': 150,      # Strong renewable commitments
        'default': 400      # Global grid average
    }
    
    # Estimated GPU power by model tier (Watts)
    # Frontier models use more powerful/more GPUs
    TIER_GPU_POWER = {
        'frontier': 700,    # H100 clusters, high utilization
        'premium': 500,     # A100 or H100
        'standard': 350,    # A100 or smaller
        'budget': 200       # Smaller GPUs, optimized inference
    }
    
    # Estimated throughput by latency tier (tokens/sec)
    LATENCY_THROUGHPUT = {
        'ultra-fast': 200,
        'fast': 100,
        'medium': 50,
        'slow': 20
    }
    
    # Default PUE (Power Usage Effectiveness) for cloud data centers
    # Typical range: 1.1 (best) to 1.6 (average)
    DEFAULT_PUE = 1.2
    
    # Get model characteristics
    provider = model_card.get('provider', 'default')
    latency_tier = model_card.get('latency_tier', 'medium')
    carbon_factor = model_card.get('carbon_factor', 0.0)
    
    # Determine model tier from carbon_factor or family
    if carbon_factor >= 8.0:
        model_tier = 'frontier'
    elif carbon_factor >= 4.0:
        model_tier = 'premium'
    elif carbon_factor >= 1.0:
        model_tier = 'standard'
    else:
        model_tier = 'budget'
    
    # If carbon_factor is 0, we don't have data
    if carbon_factor <= 0:
        return {
            'co2_grams': None,
            'energy_kwh': None,
            'methodology': 'unavailable',
            'note': 'Carbon data not available for this model'
        }
    
    # Calculate
    carbon_intensity = PROVIDER_CARBON_INTENSITY.get(provider, PROVIDER_CARBON_INTENSITY['default'])
    gpu_power = TIER_GPU_POWER.get(model_tier, 350)
    throughput = LATENCY_THROUGHPUT.get(latency_tier, 50)
    
    # Inference time in seconds
    inference_time_sec = total_tokens / throughput
    
    # Energy in kWh: (Watts × seconds) / 3600 / 1000
    energy_kwh = (gpu_power * inference_time_sec) / 3600 / 1000
    
    # Apply PUE (data center overhead)
    energy_kwh_with_pue = energy_kwh * DEFAULT_PUE
    
    # CO2 in grams
    co2_grams = energy_kwh_with_pue * carbon_intensity
    
    return {
        'co2_grams': round(co2_grams, 6),
        'energy_kwh': round(energy_kwh_with_pue, 9),
        'inference_time_sec': round(inference_time_sec, 3),
        'gpu_power_watts': gpu_power,
        'carbon_intensity_gco2_kwh': carbon_intensity,
        'pue': DEFAULT_PUE,
        'model_tier': model_tier,
        'methodology': 'estimated',
        'note': 'Based on CodeCarbon methodology and Luccioni et al. (2024)'
    }


def model_card_to_dict(card) -> Dict[str, Any]:
    """Convert a ModelCard dataclass to a dictionary for easier processing."""
    if card is None:
        return None
    if isinstance(card, dict):
        return card
    # It's a dataclass - convert to dict
    from dataclasses import asdict
    return asdict(card)


# Latency estimates by tier (milliseconds for first token, seconds for full response)
LATENCY_ESTIMATES = {
    # First token latency (ms), Full response time (sec per 100 tokens)
    'ultra-fast': {'first_token_ms': 100, 'per_100_tokens_sec': 0.3},
    'fast': {'first_token_ms': 200, 'per_100_tokens_sec': 0.5},
    'medium': {'first_token_ms': 400, 'per_100_tokens_sec': 0.8},
    'slow': {'first_token_ms': 800, 'per_100_tokens_sec': 1.5},
    'unknown': {'first_token_ms': 500, 'per_100_tokens_sec': 1.0},
}


def estimate_latency(model_info: Dict[str, Any], output_tokens: int = 100) -> Dict[str, Any]:
    """
    Estimate latency for a model based on its tier and expected output tokens.
    
    Returns:
        Dict with estimated first token latency, full response time, and tier.
    """
    latency_tier = model_info.get('latency_tier', 'unknown')
    estimates = LATENCY_ESTIMATES.get(latency_tier, LATENCY_ESTIMATES['unknown'])
    
    first_token_ms = estimates['first_token_ms']
    per_100_tokens_sec = estimates['per_100_tokens_sec']
    
    # Calculate full response time
    full_response_sec = per_100_tokens_sec * (output_tokens / 100)
    total_time_sec = (first_token_ms / 1000) + full_response_sec
    
    return {
        'latency_tier': latency_tier,
        'first_token_ms': first_token_ms,
        'full_response_sec': round(full_response_sec, 2),
        'total_time_sec': round(total_time_sec, 2),
    }


def extract_metadata_only(
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    context: Optional[Dict[str, Any]] = None,
    tools: Optional[List[str]] = None,
    reasoning_effort: Optional[str] = None,
    model_config: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Agent 1's helper: Extract RAW metadata from the intercepted API call.
    
    This function is PURELY for data extraction - NO analysis, NO recommendations,
    NO hardcoded rules for task classification. All intelligence is left to Agent 2.
    
    Returns raw factual data that Agent 2 will interpret using AI.
    """
    # Tool information (raw extraction, no categorization)
    tool_names = tools if tools else []
    
    # Context information (raw metrics)
    context_depth = calculate_context_depth(context) if context else 0
    context_size_bytes = len(str(context)) if context else 0
    
    # Conversation history (raw metrics)
    conversation_turns = len(conversation_history) if conversation_history else 0
    conversation_tokens = 0
    if conversation_history:
        for turn in conversation_history:
            conversation_tokens += estimate_tokens(turn.get('content', ''))
    
    # Model config (raw extraction)
    temperature = model_config.get('temperature', 0.7) if model_config else 0.7
    max_tokens_requested = model_config.get('max_tokens', 0) if model_config else 0
    top_p = model_config.get('top_p', 1.0) if model_config else 1.0
    
    # Reasoning effort (raw value)
    reasoning_level = reasoning_effort if reasoning_effort else "none"
    
    # Estimate total input tokens
    total_input_tokens = estimate_tokens((system_prompt or '') + (user_prompt or ''))
    total_input_tokens += conversation_tokens
    if context:
        total_input_tokens += estimate_tokens(str(context))
    
    # Return RAW metadata - Agent 2 will analyze this with AI
    return {
        # Core identifiers
        'model_requested': model_id,
        
        # Raw prompt data (no interpretation)
        'system_prompt': system_prompt or '',
        'system_prompt_length': len(system_prompt) if system_prompt else 0,
        'user_prompt': user_prompt or '',
        'user_prompt_length': len(user_prompt) if user_prompt else 0,
        
        # Raw tool data (Agent 2 will interpret)
        'tools': tool_names,
        'tool_count': len(tool_names),
        
        # Raw context data (Agent 2 will interpret)
        'context': context,
        'context_keys': list(context.keys()) if context else [],
        'context_depth': context_depth,
        'context_size_bytes': context_size_bytes,
        
        # Raw conversation data
        'conversation_history': conversation_history or [],
        'conversation_turns': conversation_turns,
        'conversation_tokens': conversation_tokens,
        
        # Raw reasoning config
        'reasoning_effort': reasoning_level,
        
        # Raw model config
        'temperature': temperature,
        'max_tokens_requested': max_tokens_requested,
        'top_p': top_p,
        
        # Raw token metrics
        'input_tokens_estimated': total_input_tokens,
    }


def calculate_context_depth(obj: Any, current_depth: int = 0) -> int:
    """Calculate the nesting depth of a context object."""
    if current_depth > 10:  # Prevent infinite recursion
        return current_depth
    
    if isinstance(obj, dict):
        if not obj:
            return current_depth
        return max(calculate_context_depth(v, current_depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return current_depth
        return max(calculate_context_depth(item, current_depth + 1) for item in obj[:10])  # Limit list check
    else:
        return current_depth


def analyze_task_appropriateness(
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    model_card: Dict[str, Any],
    metadata: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    tools: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Agent 2's helper: Use LOCAL FINE-TUNED PHI-3.5 MODEL to analyze if the model choice is APPROPRIATE.
    
    This is the ONLY place where we make judgments about model fit.
    The key output is `is_appropriate` - if True, we don't suggest alternatives.
    
    Returns structured analysis with reasoning.
    """
    
    # Build context from RAW metadata - NO pre-interpretation
    # Agent 2 (Local Phi-3.5 Model) will do ALL the intelligent analysis
    
    tool_info = f"""
TOOLS PROVIDED: {metadata.get('tool_count', 0)} tools
Tool Names: {', '.join(metadata.get('tools', [])) or 'none'}
"""

    context_info = f"""
CONTEXT DATA:
Context Keys: {metadata.get('context_keys', [])}
Context Depth: {metadata.get('context_depth', 0)} levels
Context Size: {metadata.get('context_size_bytes', 0)} bytes
"""

    conversation_info = f"""
CONVERSATION:
Turns: {metadata.get('conversation_turns', 0)}
History Tokens: {metadata.get('conversation_tokens', 0)}
"""

    model_config_info = f"""
MODEL CONFIGURATION:
Temperature: {metadata.get('temperature', 'N/A')}
Max Tokens Requested: {metadata.get('max_tokens_requested', 'N/A')}
Top P: {metadata.get('top_p', 'N/A')}
Reasoning Effort: {metadata.get('reasoning_effort', 'none')}
"""
    
    analysis_prompt = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    LLM API CALL EFFICIENCY ANALYZER                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  YOUR TASK: Analyze this API call and determine if the model is APPROPRIATE ║
║                                                                              ║
║  YOU ARE THE INTELLIGENCE - analyze the prompts and metadata to determine:  ║
║  1. What task is being requested?                                            ║
║  2. How complex is this task REALLY?                                         ║
║  3. Is the chosen model appropriate, overkill, or underpowered?             ║
║                                                                              ║
║  Do NOT rely on simple keyword matching - understand the ACTUAL task!        ║
╚══════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════
FULL SYSTEM PROMPT (Analyze this carefully!)
═══════════════════════════════════════════════════════════════════════════════
{metadata.get('system_prompt', '(none)')[:3000]}

═══════════════════════════════════════════════════════════════════════════════
FULL USER PROMPT (This is what the user actually wants!)
═══════════════════════════════════════════════════════════════════════════════
{metadata.get('user_prompt', '')[:3000]}

═══════════════════════════════════════════════════════════════════════════════
MODEL BEING USED
═══════════════════════════════════════════════════════════════════════════════
Model ID: {model_id}
Provider: {model_card.get('provider', 'unknown')}
Cost: ${model_card.get('input_cost_per_1m', 0):.2f} / ${model_card.get('output_cost_per_1m', 0):.2f} per MTok (in/out)
Model Tier: {model_card.get('family', 'unknown')}
Supports Reasoning Mode: {'YES' if model_card.get('supports_reasoning') else 'NO'}

═══════════════════════════════════════════════════════════════════════════════
RAW METADATA (Interpret these - do not just check keywords!)
═══════════════════════════════════════════════════════════════════════════════
{tool_info}
{context_info}
{conversation_info}
{model_config_info}

Total Input Tokens (estimated): {metadata.get('input_tokens_estimated', 0)}

═══════════════════════════════════════════════════════════════════════════════
YOUR ANALYSIS GUIDELINES
═══════════════════════════════════════════════════════════════════════════════
COMPLEXITY ASSESSMENT (Use your judgment!):

TRIVIAL - Examples:
  • "fix typo in this word" → even if prompt mentions "code"
  • "translate 'hello' to Spanish"
  • "what is 2+2"
  • "capitalize this sentence"

SIMPLE - Examples:
  • Short email drafts, brief summaries
  • Simple explanations without deep reasoning
  • Basic list generation, formatting tasks

MODERATE - Examples:
  • Write a function with proper error handling
  • Explain a concept with examples
  • Debug a specific error, write documentation

COMPLEX - Examples:
  • System architecture design
  • Multi-step code refactoring
  • Security audit of code
  • Complex data analysis

EXPERT - Examples:
  • Novel algorithm design with mathematical proofs
  • Distributed systems consensus protocols
  • Cryptographic implementation
  • Cutting-edge research problems

═══════════════════════════════════════════════════════════════════════════════
RESPONSE FORMAT (JSON ONLY - No markdown!)
═══════════════════════════════════════════════════════════════════════════════
{{
  "task_summary": "One clear sentence describing the ACTUAL task",
  "task_category": "text_generation|code_generation|analysis|summarization|translation|classification|extraction|creative_writing|question_answering|agentic_workflow|multimodal|other",
  "actual_complexity": "trivial|simple|moderate|complex|expert",
  "complexity_reasoning": "2-3 sentences explaining your assessment based on ACTUAL task difficulty",
  "requires_vision": false,
  "requires_audio": false,
  "requires_extended_reasoning": false,
  "requires_long_context": false,
  "requires_tool_use": false,
  "is_agentic_task": false,
  "minimum_capable_tier": "budget|standard|premium|frontier",
  "is_model_appropriate": true/false,
  "appropriateness_reasoning": "Clear explanation with specific justification",
  "estimated_output_tokens": 100
}}

RETURN ONLY THE JSON OBJECT. NO MARKDOWN. NO EXPLANATION."""

    # ===========================================================================
    # USE LOCAL FINE-TUNED MODEL ONLY (No GPT-5.2 fallback)
    # ===========================================================================
    if not LOCAL_INFERENCE_AVAILABLE:
        raise RuntimeError("Local inference module not available. Install with: pip install llama-cpp-python")
    
    if not is_local_model_available():
        raise RuntimeError(
            "Local AI-Gauge model not found. Ensure ai-gauge-q4_k_m.gguf exists in training_data/models/"
        )
    
    print("🔄 Using local fine-tuned AI-Gauge model for analysis...")
    try:
        result = analyze_with_local_model(
            system_prompt=metadata.get('system_prompt', ''),
            user_prompt=metadata.get('user_prompt', ''),
            model_id=model_id,
            metadata=metadata
        )
        print("✅ Local model analysis complete")
        return result
    except Exception as e:
        # Return safe defaults on error (no external API fallback)
        print(f"⚠️  Local model inference failed: {e}")
        return {
            "task_summary": "Unable to analyze - local model error",
            "task_category": "other",
            "actual_complexity": "moderate",
            "complexity_reasoning": f"Local analysis failed: {str(e)[:50]}",
            "requires_vision": False,
            "requires_audio": False,
            "requires_extended_reasoning": False,
            "requires_long_context": False,
            "requires_tool_use": False,
            "is_agentic_task": False,
            "minimum_capable_tier": "standard",
            "is_model_appropriate": True,  # Conservative default
            "appropriateness_reasoning": "Unable to assess - defaulting to appropriate",
            "estimated_output_tokens": 200
        }


# =============================================================================
# TIER COMPARISON LOGIC
# =============================================================================

# TIER_RANKINGS is imported from model_cards.py

def get_model_tier(model_id: str) -> str:
    """Get the tier of a model by its ID from model_cards.py."""
    # First, try to get from model_cards
    card = get_model_card(model_id)
    if card:
        return card.tier
    
    # Fuzzy matching for common patterns (fallback for unlisted models)
    model_lower = model_id.lower()
    
    # Frontier indicators
    if any(x in model_lower for x in ['opus', 'gemini-3', 'o3']):
        return 'frontier'
    if 'gpt-5' in model_lower and 'mini' not in model_lower and 'nano' not in model_lower:
        return 'frontier'
    
    # Premium indicators
    if any(x in model_lower for x in ['o4-mini', 'gemini-2.5-pro', 'pro']):
        return 'premium'
    
    # Budget indicators
    if any(x in model_lower for x in ['mini', 'nano', 'haiku', 'flash-lite', 'lite']):
        return 'budget'
    
    # Standard is the default
    return 'standard'


def is_model_overkill(current_model_id: str, minimum_tier: str) -> bool:
    """
    Determine if the current model is overkill for the task.
    
    Returns True if current model tier is higher than minimum required tier.
    Uses tier info from model_cards.py where available.
    """
    current_tier = get_model_tier(current_model_id)
    return is_model_overkill_by_tier(current_tier, minimum_tier)


def get_alternative_models(
    minimum_tier: str,
    current_model_id: str,
    requires_vision: bool = False,
    requires_reasoning: bool = False
) -> List[str]:
    """
    Get alternative model IDs that meet the minimum tier requirements.
    Only returns models that are DIFFERENT from the current one.
    Uses tier info from model_cards.py.
    """
    from model_cards import MODEL_CARDS
    
    # Get all models at or below the minimum tier
    candidates = []
    for card in MODEL_CARDS:
        if card.model_id == current_model_id:
            continue
        # Only include models at or below the required tier
        if get_tier_rank(card.tier) <= get_tier_rank(minimum_tier):
            if requires_vision and not card.supports_vision:
                continue
            if requires_reasoning and not card.supports_reasoning:
                continue
            # Prefer models in the same tier as minimum
            candidates.append((card.model_id, card.tier))
    
    # Sort by tier rank (prefer exact tier match), then by cost
    def sort_key(item):
        model_id, tier = item
        tier_diff = abs(get_tier_rank(tier) - get_tier_rank(minimum_tier))
        card = get_model_card(model_id)
        cost = card.input_cost_per_1m + card.output_cost_per_1m if card else float('inf')
        return (tier_diff, cost)
    
    candidates.sort(key=sort_key)
    return [model_id for model_id, _ in candidates]


def generate_recommendation_reasoning(
    task_summary: str,
    current_model: str,
    recommended_model: str,
    recommended_card: Dict[str, Any],
    appropriateness_reasoning: str,
    cost_savings_percent: float
) -> str:
    """
    Generate a concise explanation of WHY the recommended model suits the task
    and why it's sufficient compared to the original choice.
    
    Uses the local fine-tuned Phi-3.5 model for inference.
    """
    # Build a simple template-based reasoning (no external API call)
    # The local model handles appropriateness analysis in analyze_task_appropriateness()
    # This function just generates the human-readable summary
    
    best_for = recommended_card.get('best_for', ['general tasks'])
    best_for_str = best_for[0] if best_for else 'general tasks'
    
    if cost_savings_percent > 0:
        return (
            f"{recommended_model} is well-suited for this task because it excels at {best_for_str}. "
            f"{current_model} is more powerful than needed for this complexity level, "
            f"resulting in unnecessary cost ({cost_savings_percent:.0f}% savings possible) and carbon emissions."
        )
    else:
        return (
            f"{recommended_model} can handle this task because it's designed for {best_for_str}. "
            f"{current_model} is more powerful than needed, resulting in unnecessary cost and carbon emissions."
        )


# ============================================================================
# AGENT 1: METADATA EXTRACTOR (Pure extraction, NO analysis)
# ============================================================================

def metadata_extractor_agent(state: AgentState) -> AgentState:
    """
    Agent 1: Metadata Extractor
    
    Responsibilities:
    - Extract comprehensive metadata from the intercepted API call
    - Analyze tool usage, context complexity, reasoning requirements
    - Estimate token counts and detect task signals
    - NO analysis, NO recommendations, NO model assessment
    
    This agent structures rich input data for Agent 2.
    """
    print("\n🔍 [Agent 1: Metadata Extractor] Extracting comprehensive call metadata...")
    
    original_model = state.get('original_model', 'unknown')
    original_prompt = state.get('original_prompt', '')
    system_prompt = state.get('system_prompt', '')
    context = state.get('context', {})
    tools = state.get('tools', [])
    reasoning_effort = state.get('reasoning_effort', None)
    model_config = state.get('model_config', {})
    conversation_history = state.get('conversation_history', [])
    
    # Comprehensive metadata extraction (no LLM call, no analysis)
    metadata = extract_metadata_only(
        system_prompt=system_prompt,
        user_prompt=original_prompt,
        model_id=original_model,
        context=context,
        tools=tools,
        reasoning_effort=reasoning_effort,
        model_config=model_config,
        conversation_history=conversation_history
    )
    
    # Print raw metadata summary
    has_system = bool(metadata.get('system_prompt', ''))
    tool_list = metadata.get('tools', [])
    print(f"   ├─ Model: {original_model}")
    print(f"   ├─ Input tokens (est): {metadata['input_tokens_estimated']}")
    print(f"   ├─ System prompt: {'✓' if has_system else '✗'} ({metadata['system_prompt_length']} chars)")
    print(f"   ├─ User prompt: {metadata['user_prompt_length']} chars")
    print(f"   ├─ Context: depth={metadata['context_depth']}, size={metadata['context_size_bytes']} bytes")
    print(f"   ├─ Tools: {metadata['tool_count']} ({', '.join(tool_list[:3]) or 'none'})")
    print(f"   ├─ Reasoning effort: {metadata['reasoning_effort']}")
    print(f"   └─ Conversation: {metadata['conversation_turns']} turns, {metadata['conversation_tokens']} tokens")
    
    return {
        **state,
        'metadata': metadata,
        'messages': state.get('messages', []) + [{
            'role': 'agent',
            'agent': 'metadata_extractor',
            'content': f"Extracted raw metadata: {metadata['tool_count']} tools, {metadata['context_depth']} context depth"
        }]
    }


# ============================================================================
# AGENT 2: ANALYZER (Validates model appropriateness, conditionally suggests)
# ============================================================================

def analyzer_agent(state: AgentState) -> AgentState:
    """
    Agent 2: Analyzer
    
    Responsibilities:
    - Use LOCAL FINE-TUNED PHI-3.5 MODEL to analyze if the model choice is APPROPRIATE
    - ONLY suggest alternatives if the model is NOT appropriate
    - Calculate costs and carbon for comparison
    
    Key principle: Don't force recommendations. Validate first.
    """
    print("\n📊 [Agent 2: Analyzer] Assessing model appropriateness with local Phi-3.5 model...")
    
    metadata = state.get('metadata', {})
    original_model = metadata.get('model_requested', 'unknown')
    original_prompt = state.get('original_prompt', '')
    system_prompt = state.get('system_prompt', '')
    context = state.get('context', {})
    tools = state.get('tools', [])
    
    # Get current model info
    raw_model_info = get_model_card(original_model)
    current_model_info = model_card_to_dict(raw_model_info)
    if not current_model_info:
        current_model_info = {
            'model_id': original_model,
            'provider': 'unknown',
            'display_name': original_model,
            'carbon_factor': 0.0,
            'input_cost_per_1m': 0.0,
            'output_cost_per_1m': 0.0,
            'latency_tier': 'medium',
        }
    
    print(f"   ├─ Current model: {current_model_info.get('display_name', original_model)}")
    print(f"   │   └─ Cost: ${current_model_info.get('input_cost_per_1m', 0):.2f}/${current_model_info.get('output_cost_per_1m', 0):.2f} per MTok")
    
    # Use local fine-tuned Phi-3.5 model to analyze appropriateness
    print("   ├─ Invoking local Phi-3.5 model to analyze task and model fit...")
    analysis = analyze_task_appropriateness(
        system_prompt=system_prompt,
        user_prompt=original_prompt,
        model_id=original_model,
        model_card=current_model_info,
        metadata=metadata,  # Pass the comprehensive metadata
        context=context,
        tools=tools
    )
    
    is_appropriate = analysis.get('is_model_appropriate', True)
    appropriateness_reasoning = analysis.get('appropriateness_reasoning', '')
    minimum_tier = analysis.get('minimum_capable_tier', 'standard')
    estimated_output = analysis.get('estimated_output_tokens', 200)
    actual_complexity = analysis.get('actual_complexity', 'moderate').lower()
    
    # ==========================================================================
    # HEURISTIC ADJUSTMENTS
    # The fine-tuned model provides good base analysis but sometimes needs
    # refinement for edge cases. These adjustments are based on test results.
    # ==========================================================================
    
    # 1. Tool use requires at least standard tier (not budget)
    #    Tools add complexity regardless of the underlying task
    if metadata.get('tool_count', 0) > 0 and minimum_tier == 'budget':
        minimum_tier = 'standard'
    
    # 2. Complex tasks should require frontier tier
    #    The model sometimes underestimates complexity for reasoning-heavy tasks
    if actual_complexity == 'complex' and minimum_tier in ('budget', 'standard'):
        minimum_tier = 'frontier'
    
    # 3. Agentic tasks require frontier tier
    if analysis.get('is_agentic_task', False) and minimum_tier != 'frontier':
        minimum_tier = 'frontier'
    
    # 4. Extended reasoning explicitly requested requires at least premium tier
    if analysis.get('requires_extended_reasoning', False) and minimum_tier in ('budget', 'standard'):
        minimum_tier = 'premium'
    
    # 5. Logic puzzles with expert difficulty require frontier tier
    #    The model sometimes underestimates complexity for multi-constraint logic problems
    context_difficulty = str(metadata.get('context', {}).get('difficulty', '')).lower()
    context_type = str(metadata.get('context', {}).get('type', '')).lower()
    if context_difficulty == 'expert' and 'puzzle' in context_type:
        minimum_tier = 'frontier'
    
    # ==========================================================================
    # END HEURISTIC ADJUSTMENTS
    # ==========================================================================
    
    # TIER-BASED OVERKILL DETECTION
    # The local model outputs BOTH is_model_appropriate AND minimum_capable_tier.
    # We use the model's minimum_capable_tier as the authoritative source for tier comparison.
    # If current model tier > minimum required tier, it's OVERKILL.
    # This is not "overriding" the model - it's ensuring consistency with the model's OWN tier assessment.
    model_overkill = is_model_overkill(original_model, minimum_tier)
    current_tier = get_model_tier(original_model)
    
    if model_overkill:
        # Model's minimum_capable_tier says this task needs a lower tier
        # Override is_appropriate to ensure overkill detection works correctly
        is_appropriate = False
        if not appropriateness_reasoning or 'appropriate' in appropriateness_reasoning.lower():
            appropriateness_reasoning = f"Model is OVERKILL: Using {current_tier} tier model for a task that only requires {minimum_tier} tier. This wastes compute resources and increases costs unnecessarily."
    else:
        # If tier matches or current is lower, the model is appropriate
        # This ensures we don't flag UNDERPOWERED when tier matches exactly
        if not is_appropriate and get_tier_rank(current_tier) >= get_tier_rank(minimum_tier):
            is_appropriate = True
            appropriateness_reasoning = f"Model tier ({current_tier}) meets or exceeds the minimum required tier ({minimum_tier}) for this {actual_complexity} task."
    
    print(f"   ├─ Task: {analysis.get('task_summary', 'unknown')[:50]}...")
    print(f"   ├─ Complexity: {actual_complexity.upper()}")
    print(f"   ├─ Minimum tier needed: {minimum_tier}")
    print(f"   ├─ Current model tier: {current_tier}")
    print(f"   ├─ Model appropriate: {'✅ YES' if is_appropriate else '❌ NO (OVERKILL)' if model_overkill else '❌ NO'}")
    
    # Calculate carbon for current model
    input_tokens = metadata.get('input_tokens_estimated', 100)
    current_carbon = calculate_carbon_cost(input_tokens, estimated_output, current_model_info)
    
    # Calculate latency for current model
    current_latency = estimate_latency(current_model_info, estimated_output)
    
    # Calculate costs
    current_input_cost = (input_tokens / 1_000_000) * current_model_info.get('input_cost_per_1m', 0)
    current_output_cost = (estimated_output / 1_000_000) * current_model_info.get('output_cost_per_1m', 0)
    current_total_cost = current_input_cost + current_output_cost
    
    print(f"   ├─ Estimated latency: {current_latency['total_time_sec']}s ({current_latency['latency_tier']} tier)")
    
    # Only look for alternatives if model is NOT appropriate
    alternatives = []
    if not is_appropriate:
        print(f"   ├─ Searching for appropriate alternatives in tier: {minimum_tier}")
        alt_ids = get_alternative_models(
            minimum_tier=minimum_tier,
            current_model_id=original_model,
            requires_vision=analysis.get('requires_vision', False),
            requires_reasoning=analysis.get('requires_extended_reasoning', False)
        )
        
        for model_id in alt_ids[:5]:  # Limit to top 5
            raw_info = get_model_card(model_id)
            model_info = model_card_to_dict(raw_info)
            if not model_info:
                continue
            
            # Calculate costs
            alt_input_cost = (input_tokens / 1_000_000) * model_info.get('input_cost_per_1m', 0)
            alt_output_cost = (estimated_output / 1_000_000) * model_info.get('output_cost_per_1m', 0)
            alt_total_cost = alt_input_cost + alt_output_cost
            alt_carbon = calculate_carbon_cost(input_tokens, estimated_output, model_info)
            
            # Calculate latency for alternative
            alt_latency = estimate_latency(model_info, estimated_output)
            latency_savings_sec = current_latency['total_time_sec'] - alt_latency['total_time_sec']
            latency_savings_percent = (latency_savings_sec / current_latency['total_time_sec'] * 100) if current_latency['total_time_sec'] > 0 else 0
            
            cost_savings = current_total_cost - alt_total_cost
            cost_savings_percent = (cost_savings / current_total_cost * 100) if current_total_cost > 0 else 0
            
            # Carbon savings
            if current_carbon.get('co2_grams') and alt_carbon.get('co2_grams'):
                carbon_savings = current_carbon['co2_grams'] - alt_carbon['co2_grams']
                carbon_savings_percent = (carbon_savings / current_carbon['co2_grams'] * 100) if current_carbon['co2_grams'] > 0 else 0
            else:
                carbon_savings = None
                carbon_savings_percent = None
            
            alternatives.append({
                'model_id': model_id,
                'name': model_info.get('display_name', model_id),
                'provider': model_info.get('provider', 'unknown'),
                'total_cost': alt_total_cost,
                'cost_savings': cost_savings,
                'cost_savings_percent': cost_savings_percent,
                'carbon': alt_carbon,
                'carbon_savings': carbon_savings,
                'carbon_savings_percent': carbon_savings_percent,
                'latency': alt_latency,
                'latency_savings_sec': latency_savings_sec,
                'latency_savings_percent': latency_savings_percent,
                'best_for': model_info.get('best_for', []),
            })
        
        # Sort by cost savings
        alternatives.sort(key=lambda x: x['cost_savings_percent'], reverse=True)
        print(f"   └─ Found {len(alternatives)} suitable alternatives")
    else:
        print(f"   └─ No alternatives needed - model is appropriate")
    
    # Build carbon analysis summary
    carbon_analysis = {
        'current_model_cost': current_total_cost,
        'current_model_carbon': current_carbon,
        'current_model_latency': current_latency,
        'task_analysis': {
            'summary': analysis.get('task_summary', ''),
            'category': analysis.get('task_category', 'other'),
            'complexity': analysis.get('actual_complexity', 'moderate'),
            'complexity_reasoning': analysis.get('complexity_reasoning', ''),
            'minimum_tier': minimum_tier,
        },
        'input_tokens': input_tokens,
        'estimated_output_tokens': estimated_output,
    }
    
    return {
        **state,
        'is_model_appropriate': is_appropriate,
        'appropriateness_reasoning': appropriateness_reasoning,
        'current_model_info': current_model_info,
        'alternative_models': alternatives,
        'carbon_analysis': carbon_analysis,
        'messages': state.get('messages', []) + [{
            'role': 'agent',
            'agent': 'analyzer',
            'content': f"Analysis: appropriate={is_appropriate}, alternatives={len(alternatives)}"
        }]
    }


# ============================================================================
# AGENT 3: REVIEWER & REPORT GENERATOR (Review findings, generate comprehensive report)
# ============================================================================

def reporter_agent(state: AgentState) -> AgentState:
    """
    Agent 3: Reviewer & Report Generator
    
    Responsibilities:
    - Review the analysis from Agent 2
    - Generate a comprehensive human-readable report
    - If recommending a switch, provide MULTIPLE alternatives with GENERIC reasoning
    - Each alternative should be justified with reasoning applicable to similar models
    - Provide clear verdict: APPROPRIATE, OVERKILL, or UNDERPOWERED
    """
    print("\n📋 [Agent 3: Reviewer & Report Generator] Reviewing analysis and generating report...")
    
    is_appropriate = state.get('is_model_appropriate', True)
    appropriateness_reasoning = state.get('appropriateness_reasoning', '')
    current_model = state.get('current_model_info', {})
    alternatives = state.get('alternative_models', [])
    carbon_analysis = state.get('carbon_analysis', {})
    task_analysis = carbon_analysis.get('task_analysis', {})
    metadata = state.get('metadata', {})
    
    # Get current and minimum tier for overkill detection
    current_model_id = current_model.get('model_id', '')
    minimum_tier = task_analysis.get('minimum_tier', 'standard')
    
    # Determine verdict - check for overkill using tier comparison
    if is_appropriate:
        review_verdict = "APPROPRIATE"
    elif is_model_overkill(current_model_id, minimum_tier) or "overkill" in appropriateness_reasoning.lower():
        review_verdict = "OVERKILL"
    else:
        review_verdict = "UNDERPOWERED"
    
    # Format carbon display
    current_carbon = carbon_analysis.get('current_model_carbon', {})
    current_carbon_display = f"{current_carbon.get('co2_grams', 0):.4f}g CO₂" if current_carbon.get('co2_grams') else "unknown"
    
    if is_appropriate:
        # Model is appropriate - no recommendation needed
        recommendation = {
            'switch_recommended': False,
            'verdict': review_verdict,
            'reason': 'Model choice is appropriate for this task',
            'current_model': current_model.get('display_name', 'Unknown'),
            'recommended_alternatives': [],
            'generic_reasoning': 'The selected model tier matches the task complexity.',
            'confidence': 'high'
        }
        
        summary = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         🌱 AI-GAUGE ANALYSIS REPORT                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  VERDICT: ✅ MODEL CHOICE IS APPROPRIATE                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│ 📊 TASK ANALYSIS                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ Task:          {task_analysis.get('summary', 'Unknown task')[:55]:<55} │
│ Complexity:    {task_analysis.get('complexity', 'unknown').upper():<55} │
│ Category:      {task_analysis.get('category', 'unknown'):<55} │
│ Min Tier:      {task_analysis.get('minimum_tier', 'unknown'):<55} │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ 📍 CURRENT MODEL                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ Model:         {current_model.get('display_name', 'Unknown'):<55} │
│ Provider:      {current_model.get('provider', 'Unknown'):<55} │
│ Cost:          ${carbon_analysis.get('current_model_cost', 0):.6f} per request{' ':<36} │
│ Carbon:        {current_carbon_display:<55} │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ 💡 ASSESSMENT                                                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│ {appropriateness_reasoning[:72]:<72} │
│                                                                              │
│ Your model choice is well-suited for this task. No changes recommended.     │
└──────────────────────────────────────────────────────────────────────────────┘
"""
        print(f"   ├─ Verdict: {review_verdict}")
        print(f"   └─ Recommendation: KEEP current model")
        
    elif alternatives:
        # Model is NOT appropriate and we have alternatives
        # Build list of recommended alternatives with GENERIC reasoning
        recommended_alternatives = []
        alternatives_summary = ""
        
        # Generic reasoning that applies to all models in the recommended tier
        minimum_tier = task_analysis.get('minimum_tier', 'budget')
        generic_reasoning = generate_generic_tier_reasoning(
            minimum_tier=minimum_tier,
            task_summary=task_analysis.get('summary', 'this task'),
            task_category=task_analysis.get('category', 'general'),
            complexity=task_analysis.get('complexity', 'moderate')
        )
        
        # Process up to 4 alternatives
        for i, alt in enumerate(alternatives[:4]):
            raw_alt_card = get_model_card(alt['model_id'])
            alt_card = model_card_to_dict(raw_alt_card) or {}
            
            carbon_savings_display = f"{alt.get('carbon_savings_percent', 0):.1f}%" if alt.get('carbon_savings_percent') else "N/A"
            alt_carbon_display = f"{alt['carbon']['co2_grams']:.4f}g" if alt.get('carbon', {}).get('co2_grams') else "N/A"
            
            recommended_alternatives.append({
                'model_id': alt['model_id'],
                'name': alt['name'],
                'provider': alt['provider'],
                'cost': alt['total_cost'],
                'cost_savings_percent': alt['cost_savings_percent'],
                'carbon_savings_percent': alt.get('carbon_savings_percent'),
                'best_for': alt_card.get('best_for', [])[:3]
            })
            
            alternatives_summary += f"""
│ Option {i+1}: {alt['name']:<20} ({alt['provider']})                        │
│   Cost:    ${alt['total_cost']:.6f}  |  Savings: {alt['cost_savings_percent']:.1f}%  |  Carbon: {carbon_savings_display:<8} │"""

        recommendation = {
            'switch_recommended': True,
            'verdict': review_verdict,
            'current_model': current_model.get('display_name', 'Unknown'),
            'recommended_alternatives': recommended_alternatives,
            'primary_recommendation': recommended_alternatives[0] if recommended_alternatives else None,
            'generic_reasoning': generic_reasoning,
            'confidence': 'high' if alternatives[0]['cost_savings_percent'] > 50 else 'medium',
            'alternatives_count': len(alternatives)
        }
        
        summary = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      🌱 AI-GAUGE RECOMMENDATION REPORT                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  VERDICT: ⚠️  MODEL CHOICE IS {review_verdict:<47} ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│ 📊 TASK ANALYSIS                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ Task:          {task_analysis.get('summary', 'Unknown task')[:55]:<55} │
│ Complexity:    {task_analysis.get('complexity', 'unknown').upper():<55} │
│ Category:      {task_analysis.get('category', 'unknown'):<55} │
│ Min Tier:      {task_analysis.get('minimum_tier', 'unknown'):<55} │
│                                                                              │
│ Reasoning:     {task_analysis.get('complexity_reasoning', '')[:55]:<55} │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ 📍 CURRENT MODEL (not recommended)                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ Model:         {current_model.get('display_name', 'Unknown'):<55} │
│ Provider:      {current_model.get('provider', 'Unknown'):<55} │
│ Cost:          ${carbon_analysis.get('current_model_cost', 0):.6f} per request{' ':<36} │
│ Carbon:        {current_carbon_display:<55} │
│                                                                              │
│ Assessment:    {appropriateness_reasoning[:55]:<55} │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ 🔄 RECOMMENDED ALTERNATIVES                                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│ Any of the following models can handle this task effectively:                │
│{alternatives_summary}
│                                                                              │
│ ─────────────────────────────────────────────────────────────────────────── │
│                                                                              │
│ 💡 WHY THESE MODELS ARE SUFFICIENT (Generic Reasoning):                      │
│                                                                              │
│ {generic_reasoning[:70]:<70} │
│ {generic_reasoning[70:140] if len(generic_reasoning) > 70 else '':<70} │
│ {generic_reasoning[140:210] if len(generic_reasoning) > 140 else '':<70} │
│                                                                              │
│ All models in the {minimum_tier} tier share these capabilities and can      │
│ successfully complete this type of task with similar quality.                │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ 📈 POTENTIAL IMPACT                                                           │
├──────────────────────────────────────────────────────────────────────────────┤
│ Cost Savings:    Up to {alternatives[0]['cost_savings_percent']:.1f}% per request{' ':<39} │
│ Carbon Savings:  Up to {alternatives[0].get('carbon_savings_percent') or 0:.1f}% reduction{' ':<38} │
│ Latency Savings: Up to {alternatives[0].get('latency_savings_percent') or 0:.1f}% faster (~{alternatives[0].get('latency_savings_sec') or 0:.2f}s){' ':<18} │
└──────────────────────────────────────────────────────────────────────────────┘
"""
        print(f"   ├─ Verdict: {review_verdict}")
        print(f"   ├─ Found {len(alternatives)} suitable alternatives")
        for i, alt in enumerate(alternatives[:3]):
            print(f"   │   {i+1}. {alt['name']} ({alt['provider']}) - saves {alt['cost_savings_percent']:.1f}%")
        print(f"   └─ Generic reasoning: {generic_reasoning[:60]}...")
        
    else:
        # Model is NOT appropriate but no alternatives found
        recommendation = {
            'switch_recommended': False,
            'verdict': review_verdict,
            'reason': 'Model may be overkill but no suitable alternatives found in our registry',
            'current_model': current_model.get('display_name', 'Unknown'),
            'recommended_alternatives': [],
            'generic_reasoning': 'Consider using smaller models from the same provider or exploring other providers.',
            'confidence': 'low'
        }
        
        summary = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      🌱 AI-GAUGE ANALYSIS REPORT                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  VERDICT: ⚠️  ANALYSIS INCONCLUSIVE                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│ Assessment:                                                                  │
│ {appropriateness_reasoning[:70]:<70} │
│                                                                              │
│ However, we couldn't find suitable alternatives in our model registry.       │
│ Consider reviewing your model choice manually or exploring:                  │
│   • Claude Haiku 3.5 (Anthropic) - fast, cost-effective                     │
│   • GPT-4o-mini (OpenAI) - balanced performance                             │
│   • Gemini 2.0 Flash Lite (Google) - budget-friendly                        │
└──────────────────────────────────────────────────────────────────────────────┘
"""
        print(f"   ├─ Verdict: {review_verdict}")
        print(f"   └─ Note: No alternatives found in registry")
    
    return {
        **state,
        'recommendation': recommendation,
        'human_readable_summary': summary,
        'review_verdict': review_verdict,
        'messages': state.get('messages', []) + [{
            'role': 'agent',
            'agent': 'reviewer',
            'content': f"Review complete: verdict={review_verdict}, alternatives={len(recommendation.get('recommended_alternatives', []))}"
        }]
    }


def generate_generic_tier_reasoning(
    minimum_tier: str,
    task_summary: str,
    task_category: str,
    complexity: str
) -> str:
    """
    Generate generic reasoning that applies to ALL models in the recommended tier.
    This ensures the reasoning is not specific to one model but explains why
    any model at that tier level is sufficient.
    """
    tier_capabilities = {
        'budget': {
            'description': 'Budget-tier models (like Claude Haiku 3.5, GPT-4o-mini, Gemini Flash Lite)',
            'capabilities': [
                'handle simple text transformations and formatting',
                'answer straightforward questions with factual accuracy',
                'perform basic code generation and simple debugging',
                'execute quick summarization and extraction tasks'
            ],
            'reasoning': 'These lightweight models excel at routine tasks with low latency and minimal cost. They have sufficient context understanding for simple instructions and can produce quality outputs for tasks that don\'t require deep reasoning or complex analysis.'
        },
        'standard': {
            'description': 'Standard-tier models (like GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Flash)',
            'capabilities': [
                'write production-quality code with error handling',
                'perform multi-step analysis and reasoning',
                'generate detailed documentation and reports',
                'handle moderate complexity problem-solving'
            ],
            'reasoning': 'Standard models offer the ideal balance of capability and cost for most professional tasks. They can handle nuanced instructions, maintain context over longer conversations, and produce outputs that meet professional standards without the premium cost of frontier models.'
        },
        'premium': {
            'description': 'Premium-tier models (like GPT-4.1, Claude Sonnet 4.5, Gemini 2.5 Pro)',
            'capabilities': [
                'architect complex systems and solutions',
                'perform deep analysis with multiple considerations',
                'handle specialized domain knowledge tasks',
                'execute multi-step reasoning chains'
            ],
            'reasoning': 'Premium models are designed for complex professional work requiring sophisticated reasoning. They excel at tasks that need careful consideration of trade-offs, integration of multiple knowledge domains, and production of high-stakes outputs.'
        },
        'frontier': {
            'description': 'Frontier-tier models (like GPT-5, Claude Opus 4.5, o3)',
            'capabilities': [
                'tackle novel research problems',
                'perform expert-level reasoning and analysis',
                'handle cutting-edge technical challenges',
                'generate innovative solutions to complex problems'
            ],
            'reasoning': 'Frontier models represent the state-of-the-art in AI capability. They are essential for tasks requiring breakthrough thinking, expert-level knowledge synthesis, or handling unprecedented complexity that challenges even experienced professionals.'
        }
    }
    
    tier_info = tier_capabilities.get(minimum_tier, tier_capabilities['standard'])
    
    return f"{tier_info['description']} are well-suited for this {complexity} {task_category} task. {tier_info['reasoning']}"


# ============================================================================
# LANGGRAPH WORKFLOW
# ============================================================================

def create_decision_graph() -> StateGraph:
    """
    Create the LangGraph workflow with 3 agents.
    
    Flow: metadata_extractor -> analyzer -> reviewer -> END
    
    Agent 1 (Metadata Extractor): Comprehensive metadata extraction from intercepted call
        - Extracts tools, context, reasoning effort, model config, conversation history
        - Analyzes task signals (code, analysis, creative, reasoning indicators)
        - NO analysis, NO recommendations
    
    Agent 2 (Analyzer): Assess model appropriateness using LOCAL FINE-TUNED PHI-3.5 MODEL
        - Uses local Phi-3.5 model to analyze task complexity
        - Determines if model is APPROPRIATE, OVERKILL, or UNDERPOWERED
        - Finds alternatives only if model is inappropriate
    
    Agent 3 (Reviewer & Report Generator): Generate comprehensive report
        - Reviews analysis from Agent 2
        - Provides MULTIPLE model recommendations with GENERIC reasoning
        - Reasoning applies to all models in the recommended tier
        - Generates formatted human-readable report
    """
    workflow = StateGraph(AgentState)
    
    # Add agent nodes
    workflow.add_node("metadata_extractor", metadata_extractor_agent)
    workflow.add_node("analyzer", analyzer_agent)
    workflow.add_node("reviewer", reporter_agent)  # Renamed conceptually but keeping function name
    
    # Linear flow
    workflow.set_entry_point("metadata_extractor")
    workflow.add_edge("metadata_extractor", "analyzer")
    workflow.add_edge("analyzer", "reviewer")
    workflow.add_edge("reviewer", END)
    
    return workflow.compile()


# ============================================================================
# PUBLIC API (Designed for IDE Plugin Integration)
# ============================================================================

def analyze_llm_call(
    model: str,
    prompt: str,
    system_prompt: str = "",
    context: Optional[Dict[str, Any]] = None,
    tools: Optional[List[str]] = None,
    reasoning_effort: Optional[str] = None,
    model_config: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Main entry point for the decision module.
    
    DESIGNED FOR IDE PLUGIN INTEGRATION:
    The IDE plugin intercepts an LLM API call and extracts these parameters
    automatically from the call itself. Users don't need to provide anything.
    
    Example interception:
        # Developer writes:
        client.responses.create(
            model="<any-frontier-model>",
            input=[
                {"role": "system", "content": "You are..."},
                {"role": "user", "content": "Rewrite this..."}
            ],
            tools=[{"type": "function", "function": {"name": "search"}}],
            temperature=0.3
        )
        
        # Plugin extracts:
        analyze_llm_call(
            model="<intercepted-model>",
            prompt="Rewrite this...",
            system_prompt="You are...",
            tools=["search"],
            model_config={"temperature": 0.3}
        )
    
    Args:
        model: The model ID from the intercepted API call
        prompt: The user prompt/content
        system_prompt: System prompt if present
        context: Any context object passed to the call
        tools: List of tool/function names if using function calling
        reasoning_effort: Reasoning effort level (none, low, medium, high, max)
        model_config: Model configuration (temperature, max_tokens, top_p, etc.)
        conversation_history: Previous conversation turns for context
    
    Returns:
        Dict containing:
        - is_appropriate: Boolean - is the model choice appropriate?
        - verdict: APPROPRIATE, OVERKILL, or UNDERPOWERED
        - recommendation: Multiple recommended alternatives with generic reasoning
        - carbon_analysis: Detailed carbon/cost breakdown
        - alternatives: List of alternative models with savings
        - summary: Human-readable report
        - metadata: Extracted comprehensive metadata
    """
    graph = create_decision_graph()
    
    initial_state: AgentState = {
        'original_model': model,
        'original_prompt': prompt,
        'system_prompt': system_prompt,
        'context': context or {},
        'tools': tools or [],
        'reasoning_effort': reasoning_effort or 'none',
        'model_config': model_config or {},
        'conversation_history': conversation_history or [],
        'metadata': {},
        'is_model_appropriate': True,
        'appropriateness_reasoning': '',
        'current_model_info': {},
        'alternative_models': [],
        'carbon_analysis': {},
        'recommendation': {},
        'human_readable_summary': '',
        'review_verdict': '',
        'messages': []
    }
    
    print("\n" + "="*70)
    print("🌱 AI-GAUGE: Analyzing intercepted LLM call...")
    print("="*70)
    
    final_state = graph.invoke(initial_state)
    
    return {
        'is_appropriate': final_state.get('is_model_appropriate', True),
        'verdict': final_state.get('review_verdict', 'UNKNOWN'),
        'recommendation': final_state.get('recommendation', {}),
        'carbon_analysis': final_state.get('carbon_analysis', {}),
        'alternatives': final_state.get('alternative_models', []),
        'summary': final_state.get('human_readable_summary', ''),
        'metadata': final_state.get('metadata', {}),
        'messages': final_state.get('messages', [])
    }


# Keep old function name for backward compatibility
def analyze_llm_request(*args, **kwargs):
    """Deprecated: Use analyze_llm_call instead."""
    return analyze_llm_call(*args, **kwargs)


# ============================================================================
# CLI / DEMO
# ============================================================================
# NOTE: CLI/DEMO code has been moved to test_decision_module.py
# Run: python test_decision_module.py
# This file now serves as the core decision module library only.
