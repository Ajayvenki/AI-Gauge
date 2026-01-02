"""
AI-Gauge Local Inference Module

Uses the fine-tuned Phi-3.5 model for local inference.
Supports multiple backends:
  - Ollama (recommended): Easy setup, manages model lifecycle
  - llama-cpp-python: Direct GGUF loading, for advanced users
"""

import json
import os
import re
import requests
from pathlib import Path
from typing import Dict, Any, Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

# Backend selection: 'ollama' (recommended), 'llama_cpp' (local), 'huggingface' (cloud)
INFERENCE_BACKEND = os.getenv('AI_GAUGE_BACKEND', 'ollama')

# Ollama configuration (recommended - easiest for users)
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('AI_GAUGE_MODEL', 'ai-gauge')

# Path to the fine-tuned GGUF model (for llama_cpp backend)
# MODEL_PATH = Path(__file__).parent / "training_data" / "models" / "ai-gauge-q4_k_m.gguf"

# Try to import llama_cpp for fallback
# try:
#     from llama_cpp import Llama
#     LLAMA_CPP_AVAILABLE = True
# except ImportError:
#     LLAMA_CPP_AVAILABLE = False
#     Llama = None

# Model instance for llama_cpp (lazy loaded)
# _model_instance: Optional["Llama"] = None


def is_ollama_available() -> bool:
    """Check if Ollama is running and the ai-gauge model is available."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return any(m.get('name', '').startswith(OLLAMA_MODEL) for m in models)
        return False
    except:
        return False


def get_ollama_response(prompt: str) -> Optional[str]:
    """Get a response from Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,  # Changed from 0.1 to 0.0 for more deterministic responses
                    "top_p": 0.9,
                    "num_predict": 512,
                }
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get('response', '')
        return None
    except Exception as e:
        print(f"⚠️  Ollama request failed: {e}")
        return None



# def get_model() -> Optional["Llama"]:
#     """
#     Get or load the fine-tuned model instance (llama_cpp backend).
#     Uses lazy loading to only load the model when first needed.
#     """
#     global _model_instance
#     
#     if not LLAMA_CPP_AVAILABLE:
#         print("⚠️  llama_cpp not available. Install with: pip install llama-cpp-python")
#         return None
#     
#     if _model_instance is None:
#         if not MODEL_PATH.exists():
#             # Silent fail - model not found
#             return None
#         
#         # Suppress llama.cpp verbose output during loading
#         import sys
#         old_stderr = sys.stderr
#         sys.stderr = open(os.devnull, 'w')
#         try:
#             _model_instance = Llama(
#                 model_path=str(MODEL_PATH),
#                 n_ctx=4096,           # Context window
#                 n_threads=4,          # CPU threads
#                 n_gpu_layers=-1,      # Use all GPU layers (Metal on Mac)
#                 verbose=False
#             )
#         finally:
#             sys.stderr.close()
#             sys.stderr = old_stderr
#     
#     return _model_instance


def build_analysis_prompt(
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    metadata: Dict[str, Any]
) -> str:
    """Build the analysis prompt for the model."""
    input_cost = metadata.get('input_cost', 'Unknown')
    output_cost = metadata.get('output_cost', 'Unknown')
    provider = metadata.get('provider', 'unknown')
    
    context_str = ""
    if metadata.get('context'):
        context_str = f"\nContext: {json.dumps(metadata.get('context'))}"
    
    return f"""Analyze this LLM API call and determine if the model choice is appropriate:

Model: {model_id}
Provider: {provider}
Input Cost: {input_cost}
Output Cost: {output_cost}

System Prompt: {system_prompt[:1500] if system_prompt else '(none)'}
User Prompt: {user_prompt[:1500]}{context_str}

Return a JSON object with your analysis including:
- task_analysis: Brief description of what the task requires
- actual_complexity: One of [trivial, simple, moderate, complex, expert]
- complexity_reasoning: Why this complexity level
- minimum_capable_tier: One of [budget, standard, premium, frontier]
- is_model_appropriate: true/false
- appropriateness_reasoning: Explain why appropriate or overkill
- estimated_output_tokens: Estimated tokens needed
- requires_extended_reasoning: true/false
- is_agentic_task: true/false
- has_tool_use: true/false"""


def analyze_with_local_model(
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze an LLM API call using the fine-tuned model.
    
    Supports Ollama backend only (simplified from multiple backends).
    
    Args:
        system_prompt: The system prompt from the LLM call
        user_prompt: The user prompt from the LLM call
        model_id: The model being used (e.g., 'gpt-5.2')
        metadata: Additional metadata about the call
    
    Returns:
        Structured analysis matching the expected output format.
    """
    # Build the instruction prompt
    instruction = build_analysis_prompt(system_prompt, user_prompt, model_id, metadata)
    
    # Try Ollama only (simplified - no llama_cpp fallback)
    if INFERENCE_BACKEND == 'ollama' or is_ollama_available():
        response_text = get_ollama_response(instruction)
        if response_text:
            result = parse_model_response(response_text)
            return result
    
    # No fallback available - return default analysis
    return get_fallback_analysis(metadata)


def parse_model_response(response_text: str) -> Dict[str, Any]:
    """
    Parse the model's response to extract JSON.
    Handles various formats the model might produce, including malformed JSON.
    """
    result = None
    
    # Try direct JSON parse first
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code block
    if result is None and "```" in response_text:
        parts = response_text.split("```")
        for part in parts:
            clean = part.strip()
            if clean.startswith("json"):
                clean = clean[4:].strip()
            try:
                result = json.loads(clean)
                break
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                try:
                    # Remove newlines from string values
                    fixed = re.sub(r'(".*?)(?:\n|\r\n?)(")', r'\1 \2', clean)
                    # Fix trailing commas
                    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
                    result = json.loads(fixed)
                    break
                except:
                    continue
    
    # Try to extract key values using regex as last resort
    if result is None:
        try:
            # Extract key JSON fields using regex
            complexity_match = re.search(r'"actual_complexity"\s*:\s*"([^"]*)"', response_text)
            tier_match = re.search(r'"minimum_capable_tier"\s*:\s*"([^"]*)"', response_text)
            appropriate_match = re.search(r'"is_model_appropriate"\s*:\s*(true|false)', response_text)
            
            if complexity_match and tier_match and appropriate_match:
                result = {
                    "actual_complexity": complexity_match.group(1),
                    "minimum_capable_tier": tier_match.group(1),
                    "is_model_appropriate": appropriate_match.group(1).lower() == 'true',
                    "task_analysis": "Extracted from malformed JSON",
                    "complexity_reasoning": "Extracted from malformed JSON",
                    "appropriateness_reasoning": "Extracted from malformed JSON",
                    "estimated_output_tokens": 200,
                    "requires_extended_reasoning": False,
                    "is_agentic_task": False,
                    "has_tool_use": False
                }
        except:
            pass
    
    # Fall back to default if parsing failed
    if result is None:
        return get_fallback_analysis({})
    
    # Normalize the result to match expected format
    return normalize_analysis_result(result)


def normalize_analysis_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the model's output to match the expected format.
    Maps alternative complexity values to standard ones.
    """
    # Complexity normalization mapping
    complexity_mapping = {
        "low": "trivial",
        "very low": "trivial",
        "easy": "trivial",
        "basic": "simple",
        "medium": "moderate",
        "mid": "moderate",
        "high": "complex",
        "very high": "expert",
        "advanced": "expert"
    }
    
    # Tier normalization mapping
    tier_mapping = {
        "basic": "budget",
        "low": "budget",
        "mid": "standard",
        "medium": "standard",
        "high": "premium",
        "advanced": "frontier",
        "top": "frontier"
    }
    
    # Normalize complexity
    actual_complexity = result.get("actual_complexity", "moderate")
    if isinstance(actual_complexity, str):
        actual_complexity_lower = actual_complexity.lower()
        if actual_complexity_lower in complexity_mapping:
            result["actual_complexity"] = complexity_mapping[actual_complexity_lower]
        else:
            # Keep original if it's already valid
            valid_values = ["trivial", "simple", "moderate", "complex", "expert"]
            if actual_complexity_lower not in valid_values:
                result["actual_complexity"] = "moderate"  # Default
            else:
                result["actual_complexity"] = actual_complexity_lower
    
    # Normalize tier
    min_tier = result.get("minimum_capable_tier", "standard")
    if isinstance(min_tier, str):
        min_tier_lower = min_tier.lower()
        if min_tier_lower in tier_mapping:
            result["minimum_capable_tier"] = tier_mapping[min_tier_lower]
        else:
            valid_tiers = ["budget", "standard", "premium", "frontier"]
            if min_tier_lower not in valid_tiers:
                result["minimum_capable_tier"] = "standard"  # Default
            else:
                result["minimum_capable_tier"] = min_tier_lower
    
    # Ensure boolean fields have correct types
    bool_fields = [
        "requires_vision", "requires_audio", "requires_extended_reasoning",
        "requires_long_context", "requires_tool_use", "is_agentic_task",
        "is_model_appropriate"
    ]
    for field in bool_fields:
        if field in result:
            val = result[field]
            if isinstance(val, str):
                result[field] = val.lower() in ["true", "yes", "1"]
            elif not isinstance(val, bool):
                result[field] = bool(val)
    
    # Ensure estimated_output_tokens is always an integer
    if "estimated_output_tokens" in result:
        try:
            result["estimated_output_tokens"] = int(result["estimated_output_tokens"])
        except (ValueError, TypeError):
            result["estimated_output_tokens"] = 200
    
    # Ensure required fields have defaults
    defaults = {
        "task_summary": "Task analysis",
        "task_category": "other",
        "actual_complexity": "moderate",
        "complexity_reasoning": "Analysis completed",
        "requires_vision": False,
        "requires_audio": False,
        "requires_extended_reasoning": False,
        "requires_long_context": False,
        "requires_tool_use": False,
        "is_agentic_task": False,
        "minimum_capable_tier": "standard",
        "is_model_appropriate": True,
        "appropriateness_reasoning": "Analysis completed",
        "estimated_output_tokens": 200
    }
    
    for key, default_value in defaults.items():
        if key not in result:
            result[key] = default_value
    
    return result


def get_fallback_analysis(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a default/fallback analysis when the model is unavailable or fails.
    Uses conservative defaults (assumes model is appropriate).
    """
    return {
        "task_summary": "Unable to analyze - using defaults",
        "task_category": "other",
        "actual_complexity": "moderate",
        "complexity_reasoning": "Analysis unavailable - defaulting to moderate complexity",
        "requires_vision": False,
        "requires_audio": False,
        "requires_extended_reasoning": False,
        "requires_long_context": metadata.get('input_tokens_estimated', 0) > 8000,
        "requires_tool_use": metadata.get('tool_count', 0) > 0,
        "is_agentic_task": metadata.get('tool_count', 0) >= 3,
        "minimum_capable_tier": "standard",
        "is_model_appropriate": True,  # Conservative: don't recommend changes if unsure
        "appropriateness_reasoning": "Unable to assess - assuming appropriate",
        "estimated_output_tokens": 200
    }


def is_local_model_available() -> bool:
    """Check if the local model is available via Ollama only."""
    return is_ollama_available()


def get_model_info() -> Dict[str, Any]:
    """Get information about the local model and backends."""
    ollama_ready = is_ollama_available()
    # llama_cpp_ready = LLAMA_CPP_AVAILABLE and MODEL_PATH.exists()
    
    return {
        "available": ollama_ready,  # or llama_cpp_ready,
        "backend": "ollama" if ollama_ready else "none",  # ("llama_cpp" if llama_cpp_ready else "none"),
        "ollama": {
            "available": ollama_ready,
            "url": OLLAMA_URL,
            "model": OLLAMA_MODEL
        },
        # "llama_cpp": {
        #     "available": llama_cpp_ready,
        #     "model_path": str(MODEL_PATH),
        #     "exists": MODEL_PATH.exists(),
        #     "installed": LLAMA_CPP_AVAILABLE
        # },
        "model_name": "AI-Gauge Fine-tuned Phi-3.5 (Q4_K_M)"
    }


# ============================================================================
# CONFIGURATION (Local-only inference, no external API fallback)
# ============================================================================

# This flag is kept for backward compatibility but local model is now mandatory
USE_LOCAL_MODEL = True  # Always True - GPT-5.2 fallback has been removed


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_local_model():
    """Quick test of the local model."""
    print("🧪 Testing local AI-Gauge model...")
    print(f"Model info: {get_model_info()}")
    
    if not is_local_model_available():
        print("❌ Local model not available")
        return False
    
    # Test 1: Trivial task with frontier model (should detect OVERKILL)
    print("\n" + "=" * 50)
    print("TEST 1: Trivial task with frontier model")
    print("Expected: is_model_appropriate=False, minimum_capable_tier=budget")
    print("=" * 50)
    
    test_result_1 = analyze_with_local_model(
        system_prompt="You fix typos.",
        user_prompt="Fix the typo in: 'teh quick brown fox'",
        model_id="gpt-5.2",
        metadata={
            "provider": "openai",
            "input_cost": "$1.75/1M tokens",
            "output_cost": "$14.0/1M tokens"
        }
    )
    
    print(f"\n📊 Test 1 Result:")
    print(f"  actual_complexity: {test_result_1.get('actual_complexity')}")
    print(f"  minimum_capable_tier: {test_result_1.get('minimum_capable_tier')}")
    print(f"  is_model_appropriate: {test_result_1.get('is_model_appropriate')}")
    test1_pass = (test_result_1.get('is_model_appropriate') == False and 
                  test_result_1.get('minimum_capable_tier') == 'budget')
    print(f"  {'✅ PASS' if test1_pass else '❌ FAIL'}")
    
    # Test 2: Expert task with frontier model (should be APPROPRIATE)
    print("\n" + "=" * 50)
    print("TEST 2: Expert logic puzzle with frontier model")
    print("Expected: is_model_appropriate=True, minimum_capable_tier=frontier")
    print("=" * 50)
    
    test_result_2 = analyze_with_local_model(
        system_prompt="Expert problem solver. Think step by step.",
        user_prompt="Solve Einstein's riddle: Five houses in different colors, different nationalities...",
        model_id="gpt-5.2",
        metadata={
            "provider": "openai",
            "input_cost": "$1.75/1M tokens",
            "output_cost": "$14.0/1M tokens",
            "context": {"type": "logic_puzzle", "difficulty": "expert"}
        }
    )
    
    print(f"\n📊 Test 2 Result:")
    print(f"  actual_complexity: {test_result_2.get('actual_complexity')}")
    print(f"  minimum_capable_tier: {test_result_2.get('minimum_capable_tier')}")
    print(f"  is_model_appropriate: {test_result_2.get('is_model_appropriate')}")
    test2_pass = (test_result_2.get('is_model_appropriate') == True and 
                  test_result_2.get('minimum_capable_tier') == 'frontier')
    print(f"  {'✅ PASS' if test2_pass else '❌ FAIL'}")
    
    print(f"\n{'🎉 ALL TESTS PASSED!' if test1_pass and test2_pass else '⚠️ SOME TESTS FAILED'}")
    return test1_pass and test2_pass


if __name__ == "__main__":
    test_local_model()
