#!/usr/bin/env python3
"""
AI-Gauge Inference Server

A lightweight Flask server that provides the AI-Gauge analysis endpoint
for the VS Code extension. Uses HuggingFace Inference API.

Run with: python inference_server.py
Default: http://localhost:8080
"""

import os
import sys
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_module import analyze_llm_call
from model_cards import get_model_card, MODEL_CARDS
from local_inference import get_model_info

app = Flask(__name__)
CORS(app)  # Enable CORS for VS Code extension

# Configuration
PORT = int(os.getenv('AI_GAUGE_PORT', 8080))
HOST = os.getenv('AI_GAUGE_HOST', '127.0.0.1')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint with backend status."""
    model_info = get_model_info()
    return jsonify({
        'status': 'ok',
        'version': '0.2.0',
        'backend': model_info.get('backend', 'unknown'),
        'model_available': model_info.get('available', False),
        'ollama': model_info.get('ollama', {}),
        # 'llama_cpp': model_info.get('llama_cpp', {})
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze an LLM call and return optimization recommendations.
    
    Expected payload:
    {
        "model_used": "gpt-5.2",
        "provider": "openai",
        "context": {
            "has_system_prompt": true,
            "has_tools": false,
            "tool_count": 0,
            "has_structured_output": false,
            "max_tokens": 1000,
            "temperature": 0.7
        },
        "code_snippet": "..."
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON payload provided'}), 400
        
        model_used = data.get('model_used', 'unknown')
        context = data.get('context', {})
        code_snippet = data.get('code_snippet', '')
        
        # Build tools list from context
        tools = []
        if context.get('has_tools'):
            tools = [f"tool_{i}" for i in range(context.get('tool_count', 1))]
        
        # Run analysis with local model
        result = analyze_llm_call(
            model=model_used,
            prompt=code_snippet[:500],  # Use code snippet as proxy for prompt
            system_prompt="",
            context={
                "type": "api_call",
                "has_structured_output": context.get('has_structured_output', False),
                "max_tokens": context.get('max_tokens'),
                "temperature": context.get('temperature'),
            },
            tools=tools,
            verbose=False
        )
        
        # Get model info
        model_card = get_model_card(model_used)
        current_cost = 0
        if model_card:
            current_cost = model_card.input_cost_per_1m + model_card.output_cost_per_1m
        
        # Get recommended alternative
        recommended_alt = None
        cost_savings = 0
        
        alternatives = result.get('alternatives', [])
        if alternatives and len(alternatives) > 0:
            alt = alternatives[0]
            alt_model_id = alt.get('model_id', alt.get('name', ''))
            alt_card = get_model_card(alt_model_id)
            if alt_card:
                alt_cost = alt_card.input_cost_per_1m + alt_card.output_cost_per_1m
                cost_savings = ((current_cost - alt_cost) / current_cost * 100) if current_cost > 0 else 0
                recommended_alt = {
                    'modelId': alt_card.model_id,
                    'provider': alt_card.provider,
                    'estimatedCostPer1k': alt_cost / 1000,
                    'latencyTier': alt_card.latency_tier
                }
        
        # Build response matching TypeScript interface
        response = {
            'verdict': result.get('verdict', 'APPROPRIATE'),
            'confidence': 0.9,  # High confidence for local model
            'currentModel': {
                'modelId': model_used,
                'provider': model_card.provider if model_card else 'unknown',
                'estimatedCostPer1k': current_cost / 1000 if current_cost else 0,
                'latencyTier': model_card.latency_tier if model_card else 'medium'
            },
            'recommendedAlternative': recommended_alt,
            'costSavingsPercent': cost_savings,
            'latencySavingsMs': 0,  # Not calculated
            'reasoning': result.get('summary', 'Analysis complete'),
            'carbonFactor': model_card.carbon_factor if model_card else 1.0,
            'metadata': result.get('metadata', {})
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'verdict': 'APPROPRIATE',  # Fallback to safe default
            'confidence': 0
        }), 500


@app.route('/models', methods=['GET'])
def list_models():
    """List all available models in the database."""
    models = []
    for card in MODEL_CARDS:
        models.append({
            'model_id': card.model_id,
            'display_name': card.display_name,
            'provider': card.provider,
            'tier': card.tier,
            'input_cost': card.input_cost_per_1m,
            'output_cost': card.output_cost_per_1m,
            'carbon_factor': card.carbon_factor
        })
    return jsonify({'models': models, 'count': len(models)})


@app.route('/models/<tier>', methods=['GET'])
def list_models_by_tier(tier: str):
    """List all models in a specific tier."""
    models = [
        {
            'model_id': card.model_id,
            'display_name': card.display_name,
            'provider': card.provider,
            'input_cost': card.input_cost_per_1m,
            'output_cost': card.output_cost_per_1m
        }
        for card in MODEL_CARDS
        if card.tier == tier and card.status == 'stable'
    ]
    return jsonify({'tier': tier, 'models': models, 'count': len(models)})


if __name__ == '__main__':
    model_info = get_model_info()
    
    print("=" * 60)
    print("🌱 AI-GAUGE INFERENCE SERVER")
    print("=" * 60)
    print("   Backend: HUGGINGFACE")
    print(f"   Model: {model_info.get('huggingface', {}).get('model_id', 'AJhuggingface/ai-gauge')}")
    print(f"   API Key Set: {'✅ Yes' if model_info.get('huggingface', {}).get('api_key_set') else '❌ No'}")
    print(f"\n   Endpoint: http://{HOST}:{PORT}")
    print(f"   Health: http://{HOST}:{PORT}/health")
    print(f"   Analyze: POST http://{HOST}:{PORT}/analyze")
    print("=" * 60)
    print("\n   Press Ctrl+C to stop\n")
    
    app.run(host=HOST, port=PORT, debug=False)
