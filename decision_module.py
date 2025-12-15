"""
AI-Gauge Decision Module - LangGraph 3-Agent System

This module is designed to be called by an IDE plugin that INTERCEPTS LLM API calls
before they execute. The plugin automatically extracts all metadata from the call itself.

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │                    IDE Plugin (VS Code Extension)                │
  │                                                                  │
  │  Intercepts:  client.responses.create(model="gpt-5.2", ...)     │
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
  │  "Your task is simple text editing. GPT-5.2 is overkill.        │
  │   GPT-4o-mini can handle this because [reasoning].               │
  │   Switch to save 94% cost and reduce carbon footprint."         │
  └─────────────────────────────────────────────────────────────────┘

Agents:
  1. Metadata Extractor: Extracts raw metadata ONLY (no analysis, no recommendations)
  2. Analyzer: Uses GPT-5.2 to assess if model choice is appropriate. 
               Only suggests alternatives IF the model is overkill/inappropriate.
  3. Reporter: Generates human-readable explanation with reasoning.
"""

import os
import json
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

# LangGraph imports
from langgraph.graph import StateGraph, END

# Import our model cards
from model_cards import MODEL_CARDS, list_models_by_provider, get_model_card

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
    
    # Agent 1 output: Pure metadata extraction (NO analysis, NO recommendations)
    metadata: Dict[str, Any]
    
    # Agent 2 output: Analysis results
    is_model_appropriate: bool           # Key decision: is the model a good fit?
    appropriateness_reasoning: str       # Why it is/isn't appropriate
    current_model_info: Dict[str, Any]
    alternative_models: List[Dict[str, Any]]  # Only populated if inappropriate
    carbon_analysis: Dict[str, Any]
    
    # Agent 3 output: Final report
    recommendation: Dict[str, Any]
    human_readable_summary: str
    
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


def extract_metadata_only(
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    context: Optional[Dict[str, Any]] = None,
    tools: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Agent 1's helper: Extract pure metadata from the intercepted API call.
    
    This function does NOT:
    - Make recommendations
    - Assess if the model is appropriate
    - Suggest alternatives
    
    It ONLY extracts factual metadata about what the task IS.
    """
    return {
        'model_requested': model_id,
        'has_system_prompt': bool(system_prompt),
        'has_context': bool(context),
        'has_tools': bool(tools),
        'tool_count': len(tools) if tools else 0,
        'system_prompt_length': len(system_prompt) if system_prompt else 0,
        'user_prompt_length': len(user_prompt) if user_prompt else 0,
        'context_keys': list(context.keys()) if context else [],
        'input_tokens_estimated': estimate_tokens((system_prompt or '') + (user_prompt or '')),
    }


def analyze_task_appropriateness(
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    model_card: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    tools: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Agent 2's helper: Use GPT-5.2 to analyze if the model choice is APPROPRIATE.
    
    This is the ONLY place where we make judgments about model fit.
    The key output is `is_appropriate` - if True, we don't suggest alternatives.
    
    Returns structured analysis with reasoning.
    """
    analysis_prompt = f"""You are an expert at assessing LLM API usage efficiency.

Analyze this LLM API call and determine if the chosen model is APPROPRIATE for the task.

## System Prompt
{system_prompt[:2000] if system_prompt else "(none)"}

## User Prompt  
{user_prompt[:2000]}

## Model Being Used
{model_id}
- Provider: {model_card.get('provider', 'unknown')}
- Cost: ${model_card.get('input_cost_per_1m', 0):.2f}/${model_card.get('output_cost_per_1m', 0):.2f} per MTok
- Tier: {model_card.get('family', 'unknown')}

## Context/Tools
Context keys: {list(context.keys()) if context else 'none'}
Tools: {', '.join(tools) if tools else 'none'}

---

Analyze the ACTUAL task complexity (not how it appears) and return JSON:

{{
  "task_summary": "One sentence describing what this task actually does",
  "task_category": "text_generation|code_generation|analysis|summarization|translation|classification|extraction|creative_writing|question_answering|agentic_workflow|multimodal|other",
  "actual_complexity": "trivial|simple|moderate|complex|expert",
  "complexity_reasoning": "Why you assessed this complexity level (2-3 sentences)",
  "requires_vision": false,
  "requires_audio": false,
  "requires_extended_reasoning": false,
  "requires_long_context": false,
  "minimum_capable_tier": "budget|standard|premium|frontier - the MINIMUM tier that can handle this task well",
  "is_model_appropriate": true/false,
  "appropriateness_reasoning": "Explain whether the model is appropriate, overkill, or underpowered. Be specific about WHY.",
  "estimated_output_tokens": 100
}}

CRITICAL: Only mark is_model_appropriate=false if the model is significantly OVERKILL for the task.
A trivial task using a frontier model = inappropriate (overkill).
A complex task using a budget model = inappropriate (underpowered).
A moderate task using a standard model = appropriate.

Return ONLY the JSON object."""

    try:
        response = client.responses.create(
            model="gpt-5.2",
            input=analysis_prompt,
            max_output_tokens=600
        )
        
        response_text = response.output_text.strip()
        # Remove markdown code fences if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            if response_text.startswith("json"):
                response_text = response_text[4:].strip()
        
        return json.loads(response_text)
    except Exception as e:
        return {
            "task_summary": "Unable to analyze",
            "task_category": "other",
            "actual_complexity": "moderate",
            "complexity_reasoning": f"Analysis failed: {str(e)[:50]}",
            "requires_vision": False,
            "requires_audio": False,
            "requires_extended_reasoning": False,
            "requires_long_context": False,
            "minimum_capable_tier": "standard",
            "is_model_appropriate": True,  # Default to appropriate if analysis fails
            "appropriateness_reasoning": "Unable to assess - defaulting to appropriate",
            "estimated_output_tokens": 200
        }


def get_alternative_models(
    minimum_tier: str,
    current_model_id: str,
    requires_vision: bool = False,
    requires_reasoning: bool = False
) -> List[str]:
    """
    Get alternative model IDs that meet the minimum tier requirements.
    Only returns models that are DIFFERENT from the current one.
    """
    tier_models = {
        'budget': [
            'gpt-4o-mini', 'gpt-4.1-nano',
            'claude-haiku-4-5-20251001',
            'gemini-2.0-flash-lite'
        ],
        'standard': [
            'gpt-4o', 'gpt-4.1-mini',
            'claude-sonnet-4-5-20250929',
            'gemini-2.5-flash', 'gemini-2.0-flash'
        ],
        'premium': [
            'gpt-4.1', 'o4-mini',
            'claude-sonnet-4-5-20250929',
            'gemini-2.5-pro'
        ],
        'frontier': [
            'gpt-5', 'gpt-5.2', 'o3',
            'claude-opus-4-5-20251101',
            'gemini-3-pro-preview'
        ]
    }
    
    candidates = tier_models.get(minimum_tier, tier_models['standard'])
    
    # Filter out current model and check capabilities
    result = []
    for model_id in candidates:
        if model_id == current_model_id:
            continue
        card = get_model_card(model_id)
        if card:
            card_dict = model_card_to_dict(card)
            if requires_vision and not card_dict.get('supports_vision', False):
                continue
            if requires_reasoning and not card_dict.get('supports_reasoning', False):
                continue
            result.append(model_id)
    
    return result


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
    """
    try:
        prompt = f"""In 2-3 sentences, explain why {recommended_model} is sufficient for this task and why {current_model} is overkill.

Task: {task_summary}
Why original is overkill: {appropriateness_reasoning}
Recommended model strengths: {', '.join(recommended_card.get('best_for', ['general tasks'])[:3])}
Cost savings: {cost_savings_percent:.0f}%

Be specific and practical. Focus on why the simpler model CAN handle this task well."""

        response = client.responses.create(
            model="gpt-5.2",
            input=prompt,
            max_output_tokens=150
        )
        return response.output_text.strip()
    except Exception:
        return f"{recommended_model} can handle this task because it's designed for {recommended_card.get('best_for', ['general tasks'])[0] if recommended_card.get('best_for') else 'general tasks'}. {current_model} is more powerful than needed, resulting in unnecessary cost and carbon emissions."


# ============================================================================
# AGENT 1: METADATA EXTRACTOR (Pure extraction, NO analysis)
# ============================================================================

def metadata_extractor_agent(state: AgentState) -> AgentState:
    """
    Agent 1: Metadata Extractor
    
    Responsibilities:
    - Extract raw metadata from the intercepted API call
    - Estimate token counts
    - NO analysis, NO recommendations, NO model assessment
    
    This agent simply structures the input data for Agent 2.
    """
    print("\n🔍 [Agent 1: Metadata Extractor] Extracting call metadata...")
    
    original_model = state.get('original_model', 'unknown')
    original_prompt = state.get('original_prompt', '')
    system_prompt = state.get('system_prompt', '')
    context = state.get('context', {})
    tools = state.get('tools', [])
    
    # Pure metadata extraction (no LLM call, no analysis)
    metadata = extract_metadata_only(
        system_prompt=system_prompt,
        user_prompt=original_prompt,
        model_id=original_model,
        context=context,
        tools=tools
    )
    
    print(f"   ├─ Model: {original_model}")
    print(f"   ├─ Input tokens (est): {metadata['input_tokens_estimated']}")
    print(f"   ├─ Has system prompt: {metadata['has_system_prompt']}")
    print(f"   ├─ Has context: {metadata['has_context']}")
    print(f"   └─ Tools: {metadata['tool_count']}")
    
    return {
        **state,
        'metadata': metadata,
        'messages': state.get('messages', []) + [{
            'role': 'agent',
            'agent': 'metadata_extractor',
            'content': f"Extracted metadata: {json.dumps(metadata)}"
        }]
    }


# ============================================================================
# AGENT 2: ANALYZER (Validates model appropriateness, conditionally suggests)
# ============================================================================

def analyzer_agent(state: AgentState) -> AgentState:
    """
    Agent 2: Analyzer
    
    Responsibilities:
    - Use GPT-5.2 to analyze if the model choice is APPROPRIATE
    - ONLY suggest alternatives if the model is NOT appropriate
    - Calculate costs and carbon for comparison
    
    Key principle: Don't force recommendations. Validate first.
    """
    print("\n📊 [Agent 2: Analyzer] Assessing model appropriateness with GPT-5.2...")
    
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
    
    # Use GPT-5.2 to analyze appropriateness (this is where the LLM reasoning happens)
    print("   ├─ Invoking GPT-5.2 to analyze task and model fit...")
    analysis = analyze_task_appropriateness(
        system_prompt=system_prompt,
        user_prompt=original_prompt,
        model_id=original_model,
        model_card=current_model_info,
        context=context,
        tools=tools
    )
    
    is_appropriate = analysis.get('is_model_appropriate', True)
    appropriateness_reasoning = analysis.get('appropriateness_reasoning', '')
    minimum_tier = analysis.get('minimum_capable_tier', 'standard')
    estimated_output = analysis.get('estimated_output_tokens', 200)
    
    print(f"   ├─ Task: {analysis.get('task_summary', 'unknown')[:50]}...")
    print(f"   ├─ Complexity: {analysis.get('actual_complexity', 'unknown').upper()}")
    print(f"   ├─ Minimum tier needed: {minimum_tier}")
    print(f"   ├─ Model appropriate: {'✅ YES' if is_appropriate else '❌ NO'}")
    
    # Calculate carbon for current model
    input_tokens = metadata.get('input_tokens_estimated', 100)
    current_carbon = calculate_carbon_cost(input_tokens, estimated_output, current_model_info)
    
    # Calculate costs
    current_input_cost = (input_tokens / 1_000_000) * current_model_info.get('input_cost_per_1m', 0)
    current_output_cost = (estimated_output / 1_000_000) * current_model_info.get('output_cost_per_1m', 0)
    current_total_cost = current_input_cost + current_output_cost
    
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
# AGENT 3: REPORTER (Generates human-readable output with reasoning)
# ============================================================================

def reporter_agent(state: AgentState) -> AgentState:
    """
    Agent 3: Reporter
    
    Responsibilities:
    - Generate human-readable summary
    - If recommending a switch, explain WHY the alternative suits the task
    - Provide clear reasoning for the recommendation (or lack thereof)
    """
    print("\n✅ [Agent 3: Reporter] Generating report...")
    
    is_appropriate = state.get('is_model_appropriate', True)
    appropriateness_reasoning = state.get('appropriateness_reasoning', '')
    current_model = state.get('current_model_info', {})
    alternatives = state.get('alternative_models', [])
    carbon_analysis = state.get('carbon_analysis', {})
    task_analysis = carbon_analysis.get('task_analysis', {})
    
    # Format carbon display
    current_carbon = carbon_analysis.get('current_model_carbon', {})
    current_carbon_display = f"{current_carbon.get('co2_grams', 0):.4f}g CO₂" if current_carbon.get('co2_grams') else "unknown"
    
    if is_appropriate:
        # Model is appropriate - no recommendation needed
        recommendation = {
            'switch_recommended': False,
            'reason': 'Model choice is appropriate for this task',
            'current_model': current_model.get('display_name', 'Unknown'),
            'confidence': 'high'
        }
        
        summary = f"""
🌱 AI-GAUGE ANALYSIS
════════════════════════════════════════════════════════════════

✅ MODEL CHOICE: APPROPRIATE

📊 TASK ANALYSIS
   Task:         {task_analysis.get('summary', 'Unknown task')}
   Complexity:   {task_analysis.get('complexity', 'unknown').upper()}
   Category:     {task_analysis.get('category', 'unknown')}
   
📍 YOUR MODEL
   Model:        {current_model.get('display_name', 'Unknown')} ({current_model.get('provider', '?')})
   Cost:         ${carbon_analysis.get('current_model_cost', 0):.6f} for this request
   Carbon:       {current_carbon_display}
   
💡 ASSESSMENT
   {appropriateness_reasoning}
   
   Your model choice is well-suited for this task. No changes recommended.
   
════════════════════════════════════════════════════════════════
"""
        print(f"   ├─ Recommendation: KEEP current model")
        print(f"   └─ Reason: Model is appropriate for the task")
        
    elif alternatives:
        # Model is NOT appropriate and we have alternatives
        best_alt = alternatives[0]
        
        # Get the alternative's model card for reasoning
        raw_alt_card = get_model_card(best_alt['model_id'])
        alt_card = model_card_to_dict(raw_alt_card) or {}
        
        # Generate specific reasoning for why this alternative suits the task
        recommendation_reasoning = generate_recommendation_reasoning(
            task_summary=task_analysis.get('summary', 'this task'),
            current_model=current_model.get('display_name', 'current model'),
            recommended_model=best_alt['name'],
            recommended_card=alt_card,
            appropriateness_reasoning=appropriateness_reasoning,
            cost_savings_percent=best_alt['cost_savings_percent']
        )
        
        # Format carbon savings
        carbon_savings_display = "unknown"
        if best_alt.get('carbon_savings_percent') is not None:
            carbon_savings_display = f"{best_alt['carbon_savings_percent']:.1f}%"
        
        alt_carbon_display = "unknown"
        if best_alt.get('carbon', {}).get('co2_grams') is not None:
            alt_carbon_display = f"{best_alt['carbon']['co2_grams']:.4f}g CO₂"
        
        recommendation = {
            'switch_recommended': True,
            'recommended_model': best_alt['model_id'],
            'recommended_model_name': best_alt['name'],
            'current_model': current_model.get('display_name', 'Unknown'),
            'cost_savings_percent': best_alt['cost_savings_percent'],
            'cost_savings_usd': best_alt['cost_savings'],
            'carbon_savings_percent': best_alt.get('carbon_savings_percent'),
            'reasoning': recommendation_reasoning,
            'confidence': 'high' if best_alt['cost_savings_percent'] > 50 else 'medium',
            'alternatives_count': len(alternatives)
        }
        
        summary = f"""
🌱 AI-GAUGE RECOMMENDATION
════════════════════════════════════════════════════════════════

⚠️  MODEL CHOICE: OVERKILL

📊 TASK ANALYSIS
   Task:         {task_analysis.get('summary', 'Unknown task')}
   Complexity:   {task_analysis.get('complexity', 'unknown').upper()}
   Category:     {task_analysis.get('category', 'unknown')}
   Reasoning:    {task_analysis.get('complexity_reasoning', '')[:100]}...
   
📍 CURRENT MODEL (overkill)
   Model:        {current_model.get('display_name', 'Unknown')} ({current_model.get('provider', '?')})
   Cost:         ${carbon_analysis.get('current_model_cost', 0):.6f}
   Carbon:       {current_carbon_display}
   
🔄 RECOMMENDED ALTERNATIVE
   Model:        {best_alt['name']} ({best_alt['provider']})
   Cost:         ${best_alt['total_cost']:.6f}
   Carbon:       {alt_carbon_display}
   
💰 SAVINGS
   Cost:         {best_alt['cost_savings_percent']:.1f}% (${best_alt['cost_savings']:.6f})
   Carbon:       {carbon_savings_display}
   
💡 WHY THIS MODEL IS SUFFICIENT
   {recommendation_reasoning}
   
════════════════════════════════════════════════════════════════
"""
        print(f"   ├─ Recommendation: SWITCH to {best_alt['name']}")
        print(f"   ├─ Cost savings: {best_alt['cost_savings_percent']:.1f}%")
        print(f"   └─ Reason: {recommendation_reasoning[:60]}...")
        
    else:
        # Model is NOT appropriate but no alternatives found
        recommendation = {
            'switch_recommended': False,
            'reason': 'Model may be overkill but no suitable alternatives found',
            'current_model': current_model.get('display_name', 'Unknown'),
            'confidence': 'low'
        }
        
        summary = f"""
🌱 AI-GAUGE ANALYSIS
════════════════════════════════════════════════════════════════

⚠️  ANALYSIS INCONCLUSIVE

   {appropriateness_reasoning}
   
   However, we couldn't find suitable alternatives in our model registry.
   Consider reviewing your model choice manually.
   
════════════════════════════════════════════════════════════════
"""
        print(f"   ├─ Recommendation: NONE (no alternatives found)")
        print(f"   └─ Note: {appropriateness_reasoning[:50]}...")
    
    return {
        **state,
        'recommendation': recommendation,
        'human_readable_summary': summary,
        'messages': state.get('messages', []) + [{
            'role': 'agent',
            'agent': 'reporter',
            'content': f"Final: switch_recommended={recommendation.get('switch_recommended', False)}"
        }]
    }


# ============================================================================
# LANGGRAPH WORKFLOW
# ============================================================================

def create_decision_graph() -> StateGraph:
    """
    Create the LangGraph workflow with 3 agents.
    
    Flow: metadata_extractor -> analyzer -> reporter -> END
    
    Agent 1 (Extractor): Pure metadata extraction, no analysis
    Agent 2 (Analyzer): Assess appropriateness, conditionally find alternatives
    Agent 3 (Reporter): Generate human-readable output with reasoning
    """
    workflow = StateGraph(AgentState)
    
    # Add agent nodes (renamed for clarity)
    workflow.add_node("metadata_extractor", metadata_extractor_agent)
    workflow.add_node("analyzer", analyzer_agent)
    workflow.add_node("reporter", reporter_agent)
    
    # Linear flow
    workflow.set_entry_point("metadata_extractor")
    workflow.add_edge("metadata_extractor", "analyzer")
    workflow.add_edge("analyzer", "reporter")
    workflow.add_edge("reporter", END)
    
    return workflow.compile()


# ============================================================================
# PUBLIC API (Designed for IDE Plugin Integration)
# ============================================================================

def analyze_llm_call(
    model: str,
    prompt: str,
    system_prompt: str = "",
    context: Optional[Dict[str, Any]] = None,
    tools: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Main entry point for the decision module.
    
    DESIGNED FOR IDE PLUGIN INTEGRATION:
    The IDE plugin intercepts an LLM API call and extracts these parameters
    automatically from the call itself. Users don't need to provide anything.
    
    Example interception:
        # Developer writes:
        client.responses.create(
            model="gpt-5.2",
            input=[
                {"role": "system", "content": "You are..."},
                {"role": "user", "content": "Rewrite this..."}
            ]
        )
        
        # Plugin extracts:
        analyze_llm_call(
            model="gpt-5.2",
            prompt="Rewrite this...",
            system_prompt="You are..."
        )
    
    Args:
        model: The model ID from the intercepted API call
        prompt: The user prompt/content
        system_prompt: System prompt if present
        context: Any context object passed to the call
        tools: List of tool/function names if using function calling
    
    Returns:
        Dict containing:
        - recommendation: Whether to switch and to what
        - carbon_analysis: Detailed carbon/cost breakdown
        - summary: Human-readable explanation
    """
    graph = create_decision_graph()
    
    initial_state: AgentState = {
        'original_model': model,
        'original_prompt': prompt,
        'system_prompt': system_prompt,
        'context': context or {},
        'tools': tools or [],
        'metadata': {},
        'is_model_appropriate': True,
        'appropriateness_reasoning': '',
        'current_model_info': {},
        'alternative_models': [],
        'carbon_analysis': {},
        'recommendation': {},
        'human_readable_summary': '',
        'messages': []
    }
    
    print("\n" + "="*70)
    print("🌱 AI-GAUGE: Analyzing intercepted LLM call...")
    print("="*70)
    
    final_state = graph.invoke(initial_state)
    
    return {
        'is_appropriate': final_state.get('is_model_appropriate', True),
        'recommendation': final_state.get('recommendation', {}),
        'carbon_analysis': final_state.get('carbon_analysis', {}),
        'alternatives': final_state.get('alternative_models', []),
        'summary': final_state.get('human_readable_summary', ''),
        'messages': final_state.get('messages', [])
    }


# Keep old function name for backward compatibility
def analyze_llm_request(*args, **kwargs):
    """Deprecated: Use analyze_llm_call instead."""
    return analyze_llm_call(*args, **kwargs)


# ============================================================================
# CLI / DEMO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🌿"*35)
    print("       AI-GAUGE DECISION MODULE - Demo")
    print("       (Simulating IDE Plugin Interception)")
    print("🌿"*35)
    
    # This simulates an IDE plugin intercepting a developer's LLM call
    # The plugin automatically extracts all parameters from the API call
    
    # SCENARIO: Developer uses GPT-5.2 for what looks complex but is trivial
    # The elaborate system prompt and context make it APPEAR sophisticated,
    # but the actual task is just rewriting one sentence.
    
    demo_system_prompt = """
You are an elite AI communications specialist at a Fortune 500 developer tools company.
Your expertise spans technical writing, UX copy, brand voice consistency, and conversion optimization.
Apply the following frameworks:
- AIDA (Attention, Interest, Desire, Action) for engagement
- Plain language principles (Flesch-Kincaid Grade 8 or below)
- Developer empathy mapping for audience resonance
"""
    
    demo_prompt = """
Rewrite the following onboarding tooltip to improve clarity and user engagement:

ORIGINAL: "This extension helps you track LLM usage for greener prompts and lower costs."

Apply all brand guidelines, audience insights, and engagement frameworks from the context.
Deliver a single, refined sentence optimized for the specified constraints.

Return exactly ONE sentence. No explanation, no alternatives, no preamble.
"""
    
    demo_context = {
        "company": {"name": "AI Gauge", "industry": "Developer Tools"},
        "constraints": {"max_words": 30, "avoid": ["hype", "buzzwords"]},
    }
    
    print("\n📡 IDE Plugin intercepted API call:")
    print(f"   model='gpt-5.2'")
    print(f"   prompt='{demo_prompt[:50]}...'")
    print(f"   system_prompt='{demo_system_prompt[:40]}...'")
    
    # Run the 3-agent analysis
    result = analyze_llm_call(
        model="gpt-5.2",
        prompt=demo_prompt,
        system_prompt=demo_system_prompt,
        context=demo_context
    )
    
    # Print the human-readable summary
    print(result['summary'])
    
    # Print structured output for integration
    print("\n📋 Structured Output (for IDE integration):")
    print("-" * 40)
    output = {
        'is_appropriate': result['is_appropriate'],
        'recommendation': result['recommendation'],
        'carbon_analysis': {
            'current_cost': result['carbon_analysis'].get('current_model_cost'),
            'current_carbon': result['carbon_analysis'].get('current_model_carbon'),
            'task_complexity': result['carbon_analysis'].get('task_analysis', {}).get('complexity'),
        }
    }
    print(json.dumps(output, indent=2, default=str))
