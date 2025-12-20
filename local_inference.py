"""
AI-Gauge Local Inference Module

Uses the fine-tuned Phi-3.5 model (Q4_K_M GGUF) for local inference.
This module replaces the GPT-5.2 calls in the analyzer agent.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

# Try to import llama_cpp, fall back gracefully
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to the fine-tuned GGUF model
MODEL_PATH = Path(__file__).parent / "training_data" / "models" / "ai-gauge-q4_k_m.gguf"

# Model instance (lazy loaded)
_model_instance: Optional["Llama"] = None


def get_model() -> Optional["Llama"]:
    """
    Get or load the fine-tuned model instance.
    Uses lazy loading to only load the model when first needed.
    """
    global _model_instance
    
    if not LLAMA_CPP_AVAILABLE:
        print("⚠️  llama_cpp not available. Install with: pip install llama-cpp-python")
        return None
    
    if _model_instance is None:
        if not MODEL_PATH.exists():
            print(f"⚠️  Model not found at: {MODEL_PATH}")
            return None
        
        print(f"🔄 Loading fine-tuned AI-Gauge model from: {MODEL_PATH}")
        _model_instance = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=4096,           # Context window
            n_threads=4,          # CPU threads
            n_gpu_layers=-1,      # Use all GPU layers (Metal on Mac)
            verbose=False
        )
        print("✅ Model loaded successfully!")
    
    return _model_instance


def analyze_with_local_model(
    system_prompt: str,
    user_prompt: str,
    model_id: str,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze an LLM API call using the fine-tuned local model.
    
    This replaces the GPT-5.2 call in analyze_task_appropriateness().
    
    Returns structured analysis matching the expected output format.
    """
    model = get_model()
    
    if model is None:
        # Fall back to default response if model not available
        return get_fallback_analysis(metadata)
    
    # Build the instruction in the format the model was trained on
    tool_info = f"Tools: {metadata.get('tool_count', 0)} ({', '.join(metadata.get('tools', [])[:3]) if metadata.get('tools') else 'none'})"
    
    instruction = f"""Analyze this LLM API call for efficiency.

## System Prompt
{system_prompt[:1500] if system_prompt else '(none)'}

## User Prompt
{user_prompt[:1500]}

## Model Used
{model_id}

## Metadata
{tool_info}
Reasoning Effort: {metadata.get('reasoning_effort', 'none')}
Context Complexity: {metadata.get('context_complexity', 'none')}
Temperature: {metadata.get('temperature', 0.5)}

Return JSON with: task_summary, task_category, actual_complexity, minimum_capable_tier, is_model_appropriate, appropriateness_reasoning"""

    # Format for Phi-3.5 chat template
    prompt = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n"
    
    try:
        # Generate response
        output = model(
            prompt,
            max_tokens=512,
            temperature=0.1,
            stop=["<|end|>", "<|user|>", "</s>"],
            echo=False
        )
        
        response_text = output["choices"][0]["text"].strip()
        
        # Parse JSON from response
        result = parse_model_response(response_text)
        return result
        
    except Exception as e:
        print(f"⚠️  Local model inference failed: {e}")
        return get_fallback_analysis(metadata)


def parse_model_response(response_text: str) -> Dict[str, Any]:
    """
    Parse the model's response to extract JSON.
    Handles various formats the model might produce.
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
                continue
    
    # Try to find JSON object in text
    if result is None:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                result = json.loads(response_text[start:end])
            except json.JSONDecodeError:
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
    """Check if the local model is available for use."""
    return LLAMA_CPP_AVAILABLE and MODEL_PATH.exists()


def get_model_info() -> Dict[str, Any]:
    """Get information about the local model."""
    return {
        "available": is_local_model_available(),
        "model_path": str(MODEL_PATH),
        "exists": MODEL_PATH.exists(),
        "llama_cpp_installed": LLAMA_CPP_AVAILABLE,
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
    
    test_result = analyze_with_local_model(
        system_prompt="You are a helpful assistant.",
        user_prompt="Fix the typo in: 'The quik brown fox'",
        model_id="gpt-5.2",
        metadata={
            "tool_count": 0,
            "tools": [],
            "reasoning_effort": "none",
            "context_complexity": "none",
            "temperature": 0.3
        }
    )
    
    print(f"\n📊 Test result:")
    print(json.dumps(test_result, indent=2))
    
    if test_result.get("actual_complexity") in ["trivial", "simple"]:
        print("\n✅ Model correctly identified trivial task!")
        return True
    else:
        print(f"\n⚠️  Model returned: {test_result.get('actual_complexity')}")
        return False


if __name__ == "__main__":
    test_local_model()
