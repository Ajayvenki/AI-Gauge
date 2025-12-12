"""
AI-Gauge Decision Module - LangGraph 3-Agent System

This module intercepts metadata from developer's LLM calls and provides
intelligent recommendations for more carbon-efficient model choices.

Architecture:
  ┌─────────────────┐
  │  Developer's    │
  │  main.py        │──────► Metadata (model, prompt, task)
  └─────────────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    LangGraph Decision Pipeline                   │
  │                                                                  │
  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
  │   │   Agent 1    │───▶│   Agent 2    │───▶│   Agent 3    │      │
  │   │  Metadata    │    │  Researcher  │    │   Reviewer   │      │
  │   │  Collector   │    │              │    │              │      │
  │   └──────────────┘    └──────────────┘    └──────────────┘      │
  │                                                                  │
  └─────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────┐
  │ Recommendation  │  "For this trivial task, use gpt-4o-mini 
  │ Output          │   instead of gpt-5. Save 94% carbon."
  └─────────────────┘

Agents:
  1. Metadata Collector: Extracts model, prompt, token estimates, task type
  2. Researcher: Queries model_cards, calculates carbon, suggests alternatives
  3. Reviewer: Validates recommendations, adds human-readable insights
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
    """
    # Input from developer's code
    original_model: str
    original_prompt: str
    task_description: str
    system_prompt: str  # NEW: System prompt if any
    context: Dict[str, Any]  # NEW: Context object if any
    tools: List[str]  # NEW: Tools being used
    
    # Agent 1 output: Rich Metadata (LLM-analyzed)
    metadata: Dict[str, Any]
    
    # Agent 2 output: Research findings
    current_model_info: Dict[str, Any]
    alternative_models: List[Dict[str, Any]]
    carbon_analysis: Dict[str, Any]
    gpt_analysis: Optional[str]  # NEW: GPT-5.2 analysis
    
    # Agent 3 output: Final recommendation
    recommendation: Dict[str, Any]
    human_readable_summary: str
    
    # Conversation messages for debugging (simple list, no reducer)
    messages: List[Dict[str, Any]]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ≈ 1 token)."""
    return max(1, len(text) // 4)


def calculate_carbon_cost(tokens: int, carbon_factor: float) -> Optional[float]:
    """
    Calculate carbon cost in grams CO2.
    Returns None if carbon_factor is unknown (0 or negative).
    
    Formula: Tokens × Carbon Factor × Energy per Token × Grid Intensity
    Simplified: tokens * carbon_factor * 0.0001 (grams CO2)
    """
    if tokens <= 0 or carbon_factor <= 0:
        return None  # Unknown carbon data
    return tokens * carbon_factor * 0.0001


def model_card_to_dict(card) -> Dict[str, Any]:
    """Convert a ModelCard dataclass to a dictionary for easier processing."""
    if card is None:
        return None
    if isinstance(card, dict):
        return card
    # It's a dataclass - convert to dict
    from dataclasses import asdict
    return asdict(card)


def extract_task_metadata_with_llm(
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    context: Optional[Dict[str, Any]] = None,
    tools: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Use GPT-5.2 to analyze the task and extract rich metadata.
    This replaces the keyword-based classification with intelligent analysis.
    
    Returns structured metadata about:
    - Task intention and goal
    - Estimated complexity (with reasoning)
    - Required capabilities (vision, reasoning, long context, etc.)
    - Accuracy/quality requirements
    - Estimated tokens needed
    - Whether the chosen model is appropriate
    """
    analysis_prompt = f"""Analyze this LLM API call and extract structured metadata.

## System Prompt
{system_prompt[:2000] if system_prompt else "(none)"}

## User Prompt
{user_prompt[:2000]}

## Model Selected by Developer
{model_id}

## Context/Tools Provided
{json.dumps(context, indent=2)[:1000] if context else "(none)"}
Tools: {', '.join(tools) if tools else "(none)"}

---

Analyze the above and return a JSON object with these fields:
{{
  "task_intention": "One sentence describing what the task is trying to accomplish",
  "task_category": "One of: text_generation, code_generation, analysis, summarization, translation, classification, extraction, creative_writing, question_answering, agentic_workflow, multimodal, other",
  "actual_complexity": "One of: trivial, simple, moderate, complex, expert - based on what the task ACTUALLY requires, not how it appears",
  "complexity_reasoning": "Brief explanation of why you classified it at this complexity level",
  "requires_vision": true/false,
  "requires_audio": true/false,
  "requires_reasoning": true/false - whether extended thinking/chain-of-thought is needed,
  "requires_long_context": true/false - whether >32k tokens input is likely needed,
  "estimated_output_tokens": number - rough estimate of tokens needed for good output,
  "accuracy_requirement": "One of: low, medium, high, critical - how important is correctness",
  "latency_sensitivity": "One of: real_time, fast, relaxed, batch - how time-sensitive",
  "model_appropriate": true/false - is {model_id} a good fit for this task,
  "model_assessment": "Brief assessment of whether the chosen model is overkill, appropriate, or underpowered",
  "recommended_tier": "One of: budget, standard, premium, frontier - minimum tier needed"
}}

Return ONLY the JSON object, no other text."""

    try:
        response = client.responses.create(
            model="gpt-5.2",
            input=analysis_prompt,
            max_output_tokens=800
        )
        
        # Parse JSON from response
        response_text = response.output_text.strip()
        # Remove markdown code fences if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        metadata = json.loads(response_text)
        return metadata
    except Exception as e:
        # Fallback to basic extraction if LLM fails
        return {
            "task_intention": "unknown",
            "task_category": "other",
            "actual_complexity": "moderate",
            "complexity_reasoning": f"LLM analysis failed: {str(e)[:50]}",
            "requires_vision": False,
            "requires_audio": False,
            "requires_reasoning": False,
            "requires_long_context": False,
            "estimated_output_tokens": 500,
            "accuracy_requirement": "medium",
            "latency_sensitivity": "relaxed",
            "model_appropriate": True,
            "model_assessment": "Unable to assess",
            "recommended_tier": "standard"
        }


def get_models_for_tier(tier: str, requires_vision: bool = False, requires_reasoning: bool = False) -> List[str]:
    """
    Return model IDs appropriate for the given tier.
    Tiers: budget, standard, premium, frontier
    """
    all_models = {
        'budget': [
            'gpt-4o-mini', 'gpt-4.1-nano', 'gpt-3.5-turbo',
            'claude-haiku-4-5-20251001', 'claude-3-5-haiku-20241022',
            'gemini-2.0-flash-lite', 'gemini-2.5-flash-lite'
        ],
        'standard': [
            'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1',
            'claude-sonnet-4-5-20250929', 'claude-3-5-sonnet-20241022',
            'gemini-2.5-flash', 'gemini-2.0-flash'
        ],
        'premium': [
            'gpt-5-mini', 'gpt-5-nano', 'gpt-4.1', 'o4-mini',
            'claude-sonnet-4-5-20250929',
            'gemini-2.5-pro'
        ],
        'frontier': [
            'gpt-5', 'gpt-5.2', 'gpt-5.2-pro', 'o3',
            'claude-opus-4-5-20251101',
            'gemini-3-pro-preview'
        ]
    }
    
    # Get models for this tier and below (frontier can use any)
    tier_order = ['budget', 'standard', 'premium', 'frontier']
    tier_idx = tier_order.index(tier) if tier in tier_order else 1
    
    candidates = []
    for i in range(tier_idx + 1):
        candidates.extend(all_models.get(tier_order[i], []))
    
    # Filter by capabilities if needed
    filtered = []
    for model_id in candidates:
        card = get_model_card(model_id)
        if card:
            card_dict = model_card_to_dict(card)
            if requires_vision and not card_dict.get('supports_vision', False):
                continue
            if requires_reasoning and not card_dict.get('supports_reasoning', False):
                continue
            filtered.append(model_id)
    
    return filtered if filtered else candidates


# ============================================================================
# AGENT 1: METADATA COLLECTOR
# ============================================================================

def metadata_collector_agent(state: AgentState) -> AgentState:
    """
    Agent 1: Metadata Collector
    
    Responsibilities:
    - Extract model, system prompt, user prompt
    - Use GPT-5.2 to analyze task intention and requirements
    - Build rich structured metadata (NOT keyword-based)
    """
    print("\n🔍 [Agent 1: Metadata Collector] Analyzing request with GPT-5.2...")
    
    original_model = state.get('original_model', 'unknown')
    original_prompt = state.get('original_prompt', '')
    task_description = state.get('task_description', '')
    system_prompt = state.get('system_prompt', '')
    context = state.get('context', {})
    tools = state.get('tools', [])
    
    # Estimate tokens
    input_tokens = estimate_tokens(original_prompt + system_prompt)
    
    # Use LLM to analyze task (replaces keyword-based classification)
    print("   ├─ Invoking GPT-5.2 for intelligent task analysis...")
    llm_analysis = extract_task_metadata_with_llm(
        system_prompt=system_prompt,
        user_prompt=original_prompt,
        model_id=original_model,
        context=context,
        tools=tools
    )
    
    # Build comprehensive metadata
    metadata = {
        # Basic info
        'model_requested': original_model,
        'input_tokens': input_tokens,
        'estimated_output_tokens': llm_analysis.get('estimated_output_tokens', 500),
        'total_tokens': input_tokens + llm_analysis.get('estimated_output_tokens', 500),
        
        # LLM-analyzed task details
        'task_intention': llm_analysis.get('task_intention', 'unknown'),
        'task_category': llm_analysis.get('task_category', 'other'),
        'actual_complexity': llm_analysis.get('actual_complexity', 'moderate'),
        'complexity_reasoning': llm_analysis.get('complexity_reasoning', ''),
        
        # Required capabilities
        'requires_vision': llm_analysis.get('requires_vision', False),
        'requires_audio': llm_analysis.get('requires_audio', False),
        'requires_reasoning': llm_analysis.get('requires_reasoning', False),
        'requires_long_context': llm_analysis.get('requires_long_context', False),
        
        # Quality/performance requirements
        'accuracy_requirement': llm_analysis.get('accuracy_requirement', 'medium'),
        'latency_sensitivity': llm_analysis.get('latency_sensitivity', 'relaxed'),
        
        # Model fit assessment
        'model_appropriate': llm_analysis.get('model_appropriate', True),
        'model_assessment': llm_analysis.get('model_assessment', ''),
        'recommended_tier': llm_analysis.get('recommended_tier', 'standard'),
        
        # Additional context
        'has_system_prompt': bool(system_prompt),
        'has_context': bool(context),
        'has_tools': bool(tools),
        'prompt_length_chars': len(original_prompt),
    }
    
    print(f"   ├─ Model: {original_model}")
    print(f"   ├─ Task: {metadata['task_intention'][:60]}...")
    print(f"   ├─ Actual Complexity: {metadata['actual_complexity'].upper()}")
    print(f"   ├─ Category: {metadata['task_category']}")
    print(f"   ├─ Model Assessment: {metadata['model_assessment'][:50]}...")
    print(f"   └─ Recommended Tier: {metadata['recommended_tier']}")
    
    return {
        **state,
        'metadata': metadata,
        'messages': state.get('messages', []) + [{
            'role': 'agent',
            'agent': 'metadata_collector',
            'content': f"Extracted metadata: {json.dumps(metadata, indent=2)}"
        }]
    }


# ============================================================================
# AGENT 2: RESEARCHER
# ============================================================================

def researcher_agent(state: AgentState) -> AgentState:
    """
    Agent 2: Researcher
    
    Responsibilities:
    - Look up current model in model_cards
    - Find suitable alternatives based on extracted metadata (not labels)
    - Calculate costs for each option
    - Use GPT-5.2 to provide intelligent analysis
    """
    print("\n📚 [Agent 2: Researcher] Researching alternatives...")
    
    metadata = state.get('metadata', {})
    original_model = metadata.get('model_requested', 'unknown')
    recommended_tier = metadata.get('recommended_tier', 'standard')
    requires_vision = metadata.get('requires_vision', False)
    requires_reasoning = metadata.get('requires_reasoning', False)
    total_tokens = metadata.get('total_tokens', 100)
    
    # Get current model info
    raw_model_info = get_model_card(original_model)
    current_model_info = model_card_to_dict(raw_model_info)
    if not current_model_info:
        current_model_info = {
            'model_id': original_model,
            'provider': 'unknown',
            'display_name': original_model,
            'carbon_factor': 0.0,  # Unknown
            'input_cost_per_1m': 0.0,
            'output_cost_per_1m': 0.0,
        }
    
    print(f"   ├─ Current model: {current_model_info.get('display_name', original_model)}")
    print(f"   │   └─ Cost: ${current_model_info.get('input_cost_per_1m', 0)}/{current_model_info.get('output_cost_per_1m', 0)} per MTok")
    
    # Get recommended models based on tier and requirements
    recommended_ids = get_models_for_tier(
        recommended_tier, 
        requires_vision=requires_vision,
        requires_reasoning=requires_reasoning
    )
    
    # Calculate costs for current model
    current_input_cost = (total_tokens / 1_000_000) * current_model_info.get('input_cost_per_1m', 0)
    current_output_cost = (metadata.get('estimated_output_tokens', 500) / 1_000_000) * current_model_info.get('output_cost_per_1m', 0)
    current_total_cost = current_input_cost + current_output_cost
    current_carbon = calculate_carbon_cost(total_tokens, current_model_info.get('carbon_factor', 0.0))
    
    # Build alternative models list with cost calculations
    alternatives = []
    
    for model_id in recommended_ids:
        if model_id == original_model:
            continue
            
        raw_info = get_model_card(model_id)
        model_info = model_card_to_dict(raw_info)
        if not model_info:
            continue
            
        # Calculate costs
        alt_input_cost = (total_tokens / 1_000_000) * model_info.get('input_cost_per_1m', 0)
        alt_output_cost = (metadata.get('estimated_output_tokens', 500) / 1_000_000) * model_info.get('output_cost_per_1m', 0)
        alt_total_cost = alt_input_cost + alt_output_cost
        alt_carbon = calculate_carbon_cost(total_tokens, model_info.get('carbon_factor', 0.0))
        
        # Calculate savings
        cost_savings = current_total_cost - alt_total_cost
        cost_savings_percent = (cost_savings / current_total_cost * 100) if current_total_cost > 0 else 0
        
        # Carbon savings (may be None if unknown)
        if current_carbon is not None and alt_carbon is not None:
            carbon_savings = current_carbon - alt_carbon
            carbon_savings_percent = (carbon_savings / current_carbon * 100) if current_carbon > 0 else 0
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
            'carbon_cost': alt_carbon,
            'carbon_savings': carbon_savings,
            'carbon_savings_percent': carbon_savings_percent,
            'input_cost_per_1m': model_info.get('input_cost_per_1m', 0),
            'output_cost_per_1m': model_info.get('output_cost_per_1m', 0),
        })
    
    # Sort by cost savings (highest first)
    alternatives.sort(key=lambda x: x['cost_savings_percent'], reverse=True)
    
    print(f"   ├─ Recommended tier: {recommended_tier}")
    print(f"   ├─ Found {len(alternatives)} alternatives")
    if alternatives:
        best = alternatives[0]
        print(f"   └─ Best option: {best['name']} (saves {best['cost_savings_percent']:.1f}% cost)")
    
    # Analysis summary
    carbon_analysis = {
        'current_model_cost': current_total_cost,
        'current_model_carbon': current_carbon,  # May be None
        'best_alternative_cost': alternatives[0]['total_cost'] if alternatives else current_total_cost,
        'max_cost_savings_percent': alternatives[0]['cost_savings_percent'] if alternatives else 0,
        'carbon_data_available': current_carbon is not None,
        'total_alternatives_analyzed': len(alternatives)
    }
    
    # Use GPT-5.2 for deeper analysis
    gpt_analysis = None
    if alternatives and carbon_analysis['max_cost_savings_percent'] > 10:
        try:
            analysis_prompt = f"""
A developer is using {original_model} for this task:
- Intention: {metadata.get('task_intention', 'unknown')}
- Actual Complexity: {metadata.get('actual_complexity', 'unknown')}
- Assessment: {metadata.get('model_assessment', 'unknown')}

Top alternative: {alternatives[0]['name']}
Cost savings: {carbon_analysis['max_cost_savings_percent']:.1f}%

In 2-3 sentences, explain why switching models makes sense and any caveats.
"""
            response = client.responses.create(
                model="gpt-5.2",
                input=analysis_prompt,
                max_output_tokens=150
            )
            gpt_analysis = response.output_text
            print(f"   └─ GPT-5.2 Analysis: {gpt_analysis[:80]}...")
        except Exception as e:
            gpt_analysis = f"(Analysis unavailable: {str(e)[:50]})"
    
    return {
        **state,
        'current_model_info': current_model_info,
        'alternative_models': alternatives[:5],  # Top 5 alternatives
        'carbon_analysis': carbon_analysis,
        'gpt_analysis': gpt_analysis,
        'messages': state.get('messages', []) + [{
            'role': 'agent',
            'agent': 'researcher',
            'content': f"Research complete. Found {len(alternatives)} alternatives. GPT insight: {gpt_analysis}"
        }]
    }


# ============================================================================
# AGENT 3: REVIEWER
# ============================================================================

def reviewer_agent(state: AgentState) -> AgentState:
    """
    Agent 3: Reviewer
    
    Responsibilities:
    - Validate researcher's recommendations
    - Generate human-readable summary
    - Add practical insights based on metadata
    - Finalize recommendation
    """
    print("\n✅ [Agent 3: Reviewer] Validating recommendation...")
    
    metadata = state.get('metadata', {})
    current_model = state.get('current_model_info', {})
    alternatives = state.get('alternative_models', [])
    carbon_analysis = state.get('carbon_analysis', {})
    gpt_analysis = state.get('gpt_analysis', '')
    
    complexity = metadata.get('actual_complexity', 'moderate')
    task_intention = metadata.get('task_intention', 'unknown task')
    model_assessment = metadata.get('model_assessment', '')
    
    # Select best recommendation
    if alternatives:
        best_alt = alternatives[0]
        
        # Validate the recommendation
        is_valid = True
        warnings = []
        
        # Check if the original model was already appropriate
        if metadata.get('model_appropriate', True) and best_alt['cost_savings_percent'] < 20:
            warnings.append("ℹ️ Original model choice was reasonable for this task")
        
        # Check for minimal savings
        if best_alt['cost_savings_percent'] < 5:
            warnings.append("ℹ️ Savings are minimal - current model choice is acceptable")
            is_valid = False
        
        # Check accuracy requirements
        if metadata.get('accuracy_requirement') == 'critical':
            warnings.append("⚠️ High accuracy required - consider sticking with more capable model")
        
        recommendation = {
            'switch_recommended': is_valid and best_alt['cost_savings_percent'] > 10,
            'recommended_model': best_alt['model_id'],
            'recommended_model_name': best_alt['name'],
            'current_model': metadata.get('model_requested', 'unknown'),
            'cost_savings_percent': best_alt['cost_savings_percent'],
            'cost_savings_usd': best_alt['cost_savings'],
            'carbon_savings_percent': best_alt.get('carbon_savings_percent'),  # May be None
            'confidence': 'high' if best_alt['cost_savings_percent'] > 50 else 'medium' if best_alt['cost_savings_percent'] > 20 else 'low',
            'warnings': warnings,
            'alternatives_considered': len(alternatives)
        }
        
        # Format carbon savings for display
        carbon_display = "unknown (carbon data not available)"
        if best_alt.get('carbon_savings_percent') is not None:
            carbon_display = f"{best_alt['carbon_savings_percent']:.1f}%"
        
        current_carbon_display = "unknown"
        if carbon_analysis.get('current_model_carbon') is not None:
            current_carbon_display = f"{carbon_analysis['current_model_carbon']:.6f}g CO₂"
        
        alt_carbon_display = "unknown"
        if best_alt.get('carbon_cost') is not None:
            alt_carbon_display = f"{best_alt['carbon_cost']:.6f}g CO₂"
        
        # Generate human-readable summary
        if recommendation['switch_recommended']:
            summary = f"""
🌱 AI-GAUGE RECOMMENDATION
════════════════════════════════════════════════════════════════

📊 TASK ANALYSIS (by GPT-5.2)
   Intention:       {task_intention}
   Complexity:      {complexity.upper()}
   Category:        {metadata.get('task_category', 'unknown')}
   Assessment:      {model_assessment}
   
📍 CURRENT MODEL
   Model:           {current_model.get('display_name', 'Unknown')} ({current_model.get('provider', '?')})
   Cost:            ${carbon_analysis.get('current_model_cost', 0):.6f} for this request
   Carbon:          {current_carbon_display}
   
🔄 RECOMMENDATION: SWITCH MODEL
   Suggested:       {best_alt['name']} ({best_alt['provider']})
   Cost:            ${best_alt['total_cost']:.6f}
   Carbon:          {alt_carbon_display}
   
💰 SAVINGS
   Cost Savings:    {best_alt['cost_savings_percent']:.1f}% (${best_alt['cost_savings']:.6f})
   Carbon Savings:  {carbon_display}
   
💡 GPT-5.2 INSIGHT
   {gpt_analysis if gpt_analysis else 'N/A'}
   
{"".join(['   ' + w + chr(10) for w in warnings])}
════════════════════════════════════════════════════════════════
"""
        else:
            summary = f"""
🌱 AI-GAUGE RECOMMENDATION
════════════════════════════════════════════════════════════════

📊 TASK ANALYSIS (by GPT-5.2)
   Intention:       {task_intention}
   Complexity:      {complexity.upper()}
   Assessment:      {model_assessment}
   
✅ RECOMMENDATION: KEEP CURRENT MODEL
   Your model choice ({current_model.get('display_name', 'Unknown')}) is appropriate.
   {"".join(['   ' + w + chr(10) for w in warnings])}
   
════════════════════════════════════════════════════════════════
"""
    else:
        recommendation = {
            'switch_recommended': False,
            'reason': 'No suitable alternatives found',
            'current_model': metadata.get('model_requested', 'unknown'),
            'confidence': 'low'
        }
        summary = "⚠️ AI-GAUGE: Unable to find alternatives for your model."
    
    print(f"   ├─ Switch recommended: {recommendation.get('switch_recommended', False)}")
    print(f"   ├─ Confidence: {recommendation.get('confidence', 'N/A')}")
    savings = recommendation.get('cost_savings_percent', 0)
    print(f"   └─ Savings: {savings:.1f}%" if savings else "   └─ Savings: N/A")
    print(f"   └─ Savings: {recommendation.get('carbon_savings_percent', 0):.1f}%")
    
    return {
        **state,
        'recommendation': recommendation,
        'human_readable_summary': summary,
        'messages': state.get('messages', []) + [{
            'role': 'agent',
            'agent': 'reviewer',
            'content': f"Final recommendation: {json.dumps(recommendation, indent=2)}"
        }]
    }


# ============================================================================
# LANGGRAPH WORKFLOW
# ============================================================================

def create_decision_graph() -> StateGraph:
    """
    Create the LangGraph workflow with 3 agents.
    
    Flow: metadata_collector -> researcher -> reviewer -> END
    """
    # Initialize graph with state schema
    workflow = StateGraph(AgentState)
    
    # Add agent nodes
    workflow.add_node("metadata_collector", metadata_collector_agent)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("reviewer", reviewer_agent)
    
    # Define edges (linear flow for now)
    workflow.set_entry_point("metadata_collector")
    workflow.add_edge("metadata_collector", "researcher")
    workflow.add_edge("researcher", "reviewer")
    workflow.add_edge("reviewer", END)
    
    # Compile the graph
    return workflow.compile()


# ============================================================================
# PUBLIC API
# ============================================================================

def analyze_llm_request(
    model: str,
    prompt: str,
    task_description: str = "",
    system_prompt: str = "",
    context: Optional[Dict[str, Any]] = None,
    tools: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Main entry point for the decision module.
    
    Analyzes an LLM request and provides cost/carbon-efficient recommendations.
    Uses GPT-5.2 for intelligent task analysis (not keyword matching).
    
    Args:
        model: The model ID the developer intends to use (e.g., "gpt-5")
        prompt: The user prompt being sent to the LLM
        task_description: Optional description of what the task is for
        system_prompt: Optional system prompt if any
        context: Optional context object being passed
        tools: Optional list of tools being used
    
    Returns:
        Dict containing recommendation, cost/carbon analysis, and human-readable summary
    """
    # Create the graph
    graph = create_decision_graph()
    
    # Initialize state
    initial_state: AgentState = {
        'original_model': model,
        'original_prompt': prompt,
        'task_description': task_description or prompt[:100],
        'system_prompt': system_prompt,
        'context': context or {},
        'tools': tools or [],
        'metadata': {},
        'current_model_info': {},
        'alternative_models': [],
        'carbon_analysis': {},
        'gpt_analysis': None,
        'recommendation': {},
        'human_readable_summary': '',
        'messages': []
    }
    
    # Run the graph
    print("\n" + "="*70)
    print("🌱 AI-GAUGE: Analyzing your LLM request with GPT-5.2...")
    print("="*70)
    
    final_state = graph.invoke(initial_state)
    
    return {
        'metadata': final_state.get('metadata', {}),
        'recommendation': final_state.get('recommendation', {}),
        'carbon_analysis': final_state.get('carbon_analysis', {}),
        'alternatives': final_state.get('alternative_models', []),
        'gpt_analysis': final_state.get('gpt_analysis', ''),
        'summary': final_state.get('human_readable_summary', ''),
        'agent_messages': final_state.get('messages', [])
    }


# ============================================================================
# CLI / DEMO
# ============================================================================

if __name__ == "__main__":
    # Demo: Simulate analyzing the developer's main.py request
    print("\n" + "🌿"*35)
    print("       AI-GAUGE DECISION MODULE - LangGraph Demo")
    print("🌿"*35)
    
    # This simulates the "bad example" from main.py
    # Developer uses GPT-5.2 for what looks complex but is actually trivial
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
    
    # Run the 3-agent analysis
    result = analyze_llm_request(
        model="gpt-5.2",
        prompt=demo_prompt,
        system_prompt=demo_system_prompt,
        context=demo_context,
        task_description="Copy-editing a tooltip sentence"
    )
    
    # Print the human-readable summary
    print(result['summary'])
    
    # Print JSON output for integration
    print("\n📋 JSON Output (for integration):")
    print("-" * 40)
    print(json.dumps({
        'recommendation': result['recommendation'],
        'carbon_analysis': result['carbon_analysis'],
        'metadata_summary': {
            'task_intention': result['metadata'].get('task_intention'),
            'actual_complexity': result['metadata'].get('actual_complexity'),
            'model_assessment': result['metadata'].get('model_assessment'),
        }
    }, indent=2, default=str))
