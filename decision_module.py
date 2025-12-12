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
    
    # Agent 1 output: Metadata
    metadata: Dict[str, Any]
    
    # Agent 2 output: Research findings
    current_model_info: Dict[str, Any]
    alternative_models: List[Dict[str, Any]]
    carbon_analysis: Dict[str, Any]
    
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


def calculate_carbon_cost(tokens: int, carbon_factor: float) -> float:
    """
    Calculate carbon cost in grams CO2.
    Formula: Tokens × Carbon Factor × Energy per Token × Grid Intensity
    
    Simplified: tokens * carbon_factor * 0.0001 (grams CO2)
    """
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


def classify_task_complexity(prompt: str, task_description: str) -> str:
    """
    Classify task complexity based on prompt characteristics.
    Returns: 'trivial', 'simple', 'moderate', 'complex', 'expert'
    """
    combined = (prompt + " " + task_description).lower()
    
    # Trivial: Simple formatting, basic edits, greetings
    trivial_keywords = ['fix typo', 'capitalize', 'hello', 'hi', 'thanks', 
                        'format', 'spell check', 'grammar', 'punctuation']
    
    # Simple: Basic Q&A, summarization, translation
    simple_keywords = ['summarize', 'translate', 'explain simply', 'list',
                       'what is', 'define', 'short answer']
    
    # Moderate: Analysis, comparison, structured output
    moderate_keywords = ['analyze', 'compare', 'evaluate', 'review',
                         'pros and cons', 'structured', 'json output']
    
    # Complex: Multi-step reasoning, code generation, research
    complex_keywords = ['implement', 'code', 'algorithm', 'research',
                        'multi-step', 'reasoning', 'architecture', 'design']
    
    # Expert: Advanced reasoning, novel solutions, specialized domains
    expert_keywords = ['novel', 'cutting-edge', 'phd-level', 'prove',
                       'mathematical proof', 'optimize algorithm', 'agentic']
    
    if any(kw in combined for kw in trivial_keywords):
        return 'trivial'
    elif any(kw in combined for kw in expert_keywords):
        return 'expert'
    elif any(kw in combined for kw in complex_keywords):
        return 'complex'
    elif any(kw in combined for kw in moderate_keywords):
        return 'moderate'
    elif any(kw in combined for kw in simple_keywords):
        return 'simple'
    else:
        # Default to moderate for unknown patterns
        return 'moderate'


def get_recommended_models_for_complexity(complexity: str) -> List[str]:
    """
    Return list of appropriate model IDs for given complexity level.
    """
    recommendations = {
        'trivial': ['gpt-4o-mini', 'gpt-4.1-mini', 'claude-3.5-haiku', 'gemini-2.0-flash'],
        'simple': ['gpt-4o-mini', 'gpt-4.1-nano', 'claude-3.5-haiku', 'gemini-2.5-flash'],
        'moderate': ['gpt-4o', 'gpt-4.1', 'claude-sonnet-4', 'gemini-2.5-pro'],
        'complex': ['gpt-4.1', 'gpt-5-mini', 'claude-sonnet-4', 'gemini-2.5-pro'],
        'expert': ['gpt-5', 'gpt-5.2', 'claude-opus-4', 'gemini-3-pro', 'o3', 'o4-mini']
    }
    return recommendations.get(complexity, recommendations['moderate'])


# ============================================================================
# AGENT 1: METADATA COLLECTOR
# ============================================================================

def metadata_collector_agent(state: AgentState) -> AgentState:
    """
    Agent 1: Metadata Collector
    
    Responsibilities:
    - Extract model being used
    - Estimate input/output tokens
    - Classify task type and complexity
    - Build structured metadata object
    """
    print("\n🔍 [Agent 1: Metadata Collector] Analyzing request...")
    
    original_model = state.get('original_model', 'unknown')
    original_prompt = state.get('original_prompt', '')
    task_description = state.get('task_description', '')
    
    # Estimate tokens
    input_tokens = estimate_tokens(original_prompt)
    # Assume output is roughly 50% of input for estimation
    estimated_output_tokens = max(50, input_tokens // 2)
    
    # Classify complexity
    complexity = classify_task_complexity(original_prompt, task_description)
    
    # Build metadata
    metadata = {
        'model_requested': original_model,
        'input_tokens': input_tokens,
        'estimated_output_tokens': estimated_output_tokens,
        'total_tokens': input_tokens + estimated_output_tokens,
        'task_complexity': complexity,
        'prompt_length_chars': len(original_prompt),
        'has_code': 'def ' in original_prompt or 'function' in original_prompt,
        'has_json_request': 'json' in original_prompt.lower(),
        'language_detected': 'english',  # Simplified
    }
    
    print(f"   ├─ Model: {original_model}")
    print(f"   ├─ Tokens: ~{metadata['total_tokens']} (in: {input_tokens}, out: ~{estimated_output_tokens})")
    print(f"   └─ Complexity: {complexity.upper()}")
    
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
    - Find suitable alternatives based on task complexity
    - Calculate carbon costs for each option
    - Use GPT to provide intelligent analysis (optional)
    """
    print("\n📚 [Agent 2: Researcher] Researching alternatives...")
    
    metadata = state.get('metadata', {})
    original_model = metadata.get('model_requested', 'unknown')
    complexity = metadata.get('task_complexity', 'moderate')
    total_tokens = metadata.get('total_tokens', 100)
    
    # Get current model info (convert dataclass to dict)
    raw_model_info = get_model_card(original_model)
    current_model_info = model_card_to_dict(raw_model_info)
    if not current_model_info:
        # Create placeholder for unknown model
        current_model_info = {
            'model_id': original_model,
            'provider': 'unknown',
            'display_name': original_model,
            'carbon_factor': 1.5,  # Assume high
            'best_for': ['unknown'],
            'strengths': ['unknown'],
            'weaknesses': ['unknown']
        }
    
    print(f"   ├─ Current model: {current_model_info.get('display_name', original_model)}")
    print(f"   │   └─ Carbon factor: {current_model_info.get('carbon_factor', 'N/A')}")
    
    # Get recommended models for this complexity
    recommended_ids = get_recommended_models_for_complexity(complexity)
    
    # Build alternative models list with carbon calculations
    alternatives = []
    current_carbon = calculate_carbon_cost(total_tokens, current_model_info.get('carbon_factor', 1.0))
    
    for model_id in recommended_ids:
        raw_info = get_model_card(model_id)
        model_info = model_card_to_dict(raw_info)
        if model_info and model_id != original_model:
            alt_carbon = calculate_carbon_cost(total_tokens, model_info.get('carbon_factor', 1.0))
            carbon_savings = current_carbon - alt_carbon
            savings_percent = (carbon_savings / current_carbon * 100) if current_carbon > 0 else 0
            
            best_for = model_info.get('best_for', [])
            alternatives.append({
                'model_id': model_id,
                'name': model_info.get('display_name', model_id),
                'provider': model_info.get('provider', 'unknown'),
                'carbon_factor': model_info.get('carbon_factor', 1.0),
                'carbon_cost': alt_carbon,
                'carbon_savings': carbon_savings,
                'savings_percent': savings_percent,
                'best_for': best_for,
                'suitable_for_task': complexity in best_for or 
                                    any(complexity in bf for bf in best_for)
            })
    
    # Sort by carbon savings (highest first)
    alternatives.sort(key=lambda x: x['savings_percent'], reverse=True)
    
    print(f"   ├─ Found {len(alternatives)} alternatives for '{complexity}' tasks")
    if alternatives:
        best = alternatives[0]
        print(f"   └─ Best option: {best['name']} (saves {best['savings_percent']:.1f}% carbon)")
    
    # Carbon analysis summary
    carbon_analysis = {
        'current_model_carbon': current_carbon,
        'best_alternative_carbon': alternatives[0]['carbon_cost'] if alternatives else current_carbon,
        'max_savings_percent': alternatives[0]['savings_percent'] if alternatives else 0,
        'total_alternatives_analyzed': len(alternatives)
    }
    
    # Optional: Use GPT-5.2 for deeper analysis (if API available)
    gpt_analysis = None
    try:
        # Only call API if we have significant savings potential
        if carbon_analysis['max_savings_percent'] > 20:
            analysis_prompt = f"""
            A developer is using {original_model} for a {complexity} task.
            The task involves: {state.get('task_description', 'unknown task')}
            
            Top alternative: {alternatives[0]['name'] if alternatives else 'none'}
            Carbon savings: {carbon_analysis['max_savings_percent']:.1f}%
            
            In 2-3 sentences, explain why switching models makes sense for this use case.
            """
            
            response = client.responses.create(
                model="gpt-4o-mini",  # Use mini for meta-analysis to save carbon!
                input=analysis_prompt,
                max_output_tokens=150
            )
            gpt_analysis = response.output_text
            print(f"   └─ GPT Analysis: {gpt_analysis[:80]}...")
    except Exception as e:
        gpt_analysis = f"(Analysis unavailable: {str(e)[:50]})"
    
    return {
        **state,
        'current_model_info': current_model_info,
        'alternative_models': alternatives[:5],  # Top 5 alternatives
        'carbon_analysis': carbon_analysis,
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
    - Add practical insights and warnings
    - Finalize recommendation
    """
    print("\n✅ [Agent 3: Reviewer] Validating recommendation...")
    
    metadata = state.get('metadata', {})
    current_model = state.get('current_model_info', {})
    alternatives = state.get('alternative_models', [])
    carbon_analysis = state.get('carbon_analysis', {})
    complexity = metadata.get('task_complexity', 'moderate')
    
    # Select best recommendation
    if alternatives:
        best_alt = alternatives[0]
        
        # Validate the recommendation makes sense
        is_valid = True
        warnings = []
        
        # Check if we're recommending a model that's TOO weak
        if complexity in ['complex', 'expert'] and best_alt['carbon_factor'] < 0.3:
            warnings.append("⚠️ Recommended model may be underpowered for this complex task")
            is_valid = False
        
        # Check for minimal savings (not worth switching)
        if best_alt['savings_percent'] < 5:
            warnings.append("ℹ️ Savings are minimal - current model choice is reasonable")
        
        recommendation = {
            'switch_recommended': is_valid and best_alt['savings_percent'] > 10,
            'recommended_model': best_alt['model_id'],
            'recommended_model_name': best_alt['name'],
            'current_model': metadata.get('model_requested', 'unknown'),
            'carbon_savings_percent': best_alt['savings_percent'],
            'carbon_savings_grams': best_alt['carbon_savings'],
            'confidence': 'high' if best_alt['savings_percent'] > 50 else 'medium' if best_alt['savings_percent'] > 20 else 'low',
            'warnings': warnings,
            'alternatives_considered': len(alternatives)
        }
        
        # Generate human-readable summary
        if recommendation['switch_recommended']:
            summary = f"""
🌱 AI-GAUGE RECOMMENDATION
════════════════════════════════════════════════════════════════

📊 ANALYSIS
   Task Complexity: {complexity.upper()}
   Current Model:   {current_model.get('display_name', 'Unknown')} ({current_model.get('provider', '?')})
   
🔄 RECOMMENDATION: SWITCH MODEL
   Suggested:       {best_alt['name']} ({best_alt['provider']})
   
💚 CARBON IMPACT
   Current Cost:    {carbon_analysis['current_model_carbon']:.4f}g CO₂
   New Cost:        {best_alt['carbon_cost']:.4f}g CO₂
   You Save:        {best_alt['savings_percent']:.1f}% carbon emissions
   
💡 WHY SWITCH?
   Your task is classified as '{complexity}' - it doesn't require
   the full power of {current_model.get('display_name', 'your current model')}.
   {best_alt['name']} handles {complexity} tasks efficiently with
   significantly lower environmental impact.
   
{"".join(['   ' + w + chr(10) for w in warnings])}
════════════════════════════════════════════════════════════════
"""
        else:
            summary = f"""
🌱 AI-GAUGE RECOMMENDATION
════════════════════════════════════════════════════════════════

📊 ANALYSIS
   Task Complexity: {complexity.upper()}
   Current Model:   {current_model.get('display_name', 'Unknown')}
   
✅ RECOMMENDATION: KEEP CURRENT MODEL
   Your model choice is appropriate for this {complexity} task.
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
    task_description: str = ""
) -> Dict[str, Any]:
    """
    Main entry point for the decision module.
    
    Analyzes an LLM request and provides carbon-efficient recommendations.
    
    Args:
        model: The model ID the developer intends to use (e.g., "gpt-5")
        prompt: The actual prompt being sent to the LLM
        task_description: Optional description of what the task is for
    
    Returns:
        Dict containing recommendation, carbon analysis, and human-readable summary
    """
    # Create the graph
    graph = create_decision_graph()
    
    # Initialize state
    initial_state: AgentState = {
        'original_model': model,
        'original_prompt': prompt,
        'task_description': task_description or prompt[:100],
        'metadata': {},
        'current_model_info': {},
        'alternative_models': [],
        'carbon_analysis': {},
        'recommendation': {},
        'human_readable_summary': '',
        'messages': []
    }
    
    # Run the graph
    print("\n" + "="*70)
    print("🌱 AI-GAUGE: Analyzing your LLM request...")
    print("="*70)
    
    final_state = graph.invoke(initial_state)
    
    return {
        'metadata': final_state.get('metadata', {}),
        'recommendation': final_state.get('recommendation', {}),
        'carbon_analysis': final_state.get('carbon_analysis', {}),
        'alternatives': final_state.get('alternative_models', []),
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
    
    # This simulates intercepting the developer's oversized model request
    demo_model = "gpt-5"
    demo_prompt = """
    Please review the following enterprise documentation for quality and compliance:
    
    Meeting Notes - Q3 Planning
    - Discussed roadmap
    - Action items assigned
    - Next meeting Tuesday
    
    Ensure professional tone and fix any grammatical issues.
    """
    demo_task = "Copy-editing meeting notes (trivial formatting task)"
    
    # Run the 3-agent analysis
    result = analyze_llm_request(
        model=demo_model,
        prompt=demo_prompt,
        task_description=demo_task
    )
    
    # Print the human-readable summary
    print(result['summary'])
    
    # Print JSON output for integration
    print("\n📋 JSON Output (for integration):")
    print("-" * 40)
    print(json.dumps({
        'recommendation': result['recommendation'],
        'carbon_analysis': result['carbon_analysis']
    }, indent=2))
