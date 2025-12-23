"use strict";
/**
 * AI-Gauge Client
 *
 * Communicates with the local inference server or directly with Ollama
 * to analyze LLM calls and return optimization recommendations.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.AIGaugeClient = void 0;
class AIGaugeClient {
    constructor(config) {
        this.config = config;
    }
    /**
     * Update configuration (called when VS Code settings change)
     */
    updateConfig(config) {
        this.config = { ...this.config, ...config };
    }
    /**
     * Analyze a detected LLM call
     */
    async analyze(call) {
        try {
            // Priority: HuggingFace (cloud) > Ollama (local) > Server
            if (this.config.useHuggingFace && this.config.huggingFaceApiKey) {
                return await this.analyzeWithHuggingFace(call);
            }
            else if (this.config.useOllamaDirect) {
                return await this.analyzeWithOllama(call);
            }
            else {
                return await this.analyzeWithServer(call);
            }
        }
        catch (error) {
            console.warn('AI-Gauge analysis failed, using fallback:', error);
            return this.fallbackAnalysis(call);
        }
    }
    /**
     * Analyze using HuggingFace Inference API (simplest setup)
     */
    async analyzeWithHuggingFace(call) {
        const prompt = this.buildOllamaPrompt(call); // Same prompt format works
        const HF_INFERENCE_URL = 'https://router.huggingface.co';
        const response = await fetch(`${HF_INFERENCE_URL}/${this.config.huggingFaceModel}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.config.huggingFaceApiKey}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                inputs: `<|user|>\n${prompt}<|end|>\n<|assistant|>\n`,
                parameters: {
                    max_new_tokens: 512,
                    temperature: 0.1,
                    top_p: 0.9,
                    do_sample: true,
                    return_full_text: false
                }
            })
        });
        if (response.status === 503) {
            // Model is loading - wait and retry
            console.log('HuggingFace model loading, waiting...');
            await new Promise(resolve => setTimeout(resolve, 20000));
            return this.analyzeWithHuggingFace(call); // Retry once
        }
        if (!response.ok) {
            throw new Error(`HuggingFace returned ${response.status}`);
        }
        const result = await response.json();
        let responseText = '';
        if (Array.isArray(result) && result.length > 0) {
            responseText = result[0].generated_text || '';
        }
        else {
            responseText = result.generated_text || '';
        }
        return this.parseOllamaResponse(responseText, call); // Same parser works
    }
    /**
     * Analyze using the inference server (recommended)
     */
    async analyzeWithServer(call) {
        const payload = this.buildPayload(call);
        const response = await fetch(`${this.config.serverUrl}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }
        const result = await response.json();
        return this.parseResponse(result, call);
    }
    /**
     * Analyze by calling Ollama directly (no inference server needed)
     */
    async analyzeWithOllama(call) {
        const prompt = this.buildOllamaPrompt(call);
        const response = await fetch(`${this.config.ollamaUrl}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: this.config.ollamaModel,
                prompt: prompt,
                stream: false,
                options: {
                    temperature: 0.1,
                    num_predict: 512
                }
            })
        });
        if (!response.ok) {
            throw new Error(`Ollama returned ${response.status}`);
        }
        const result = await response.json();
        return this.parseOllamaResponse(result.response, call);
    }
    /**
     * Build prompt for direct Ollama analysis
     */
    buildOllamaPrompt(call) {
        return `Analyze this LLM API call and determine if the model choice is appropriate:

Model: ${call.modelId}
Provider: ${call.provider}
Has Tools: ${call.hasTools}
Tool Count: ${call.toolCount}
Has Structured Output: ${call.hasStructuredOutput}
Code Context: ${call.surroundingCode.substring(0, 500)}

Return a JSON object with:
- is_model_appropriate: true/false
- minimum_capable_tier: one of [budget, standard, premium, frontier]
- actual_complexity: one of [trivial, simple, moderate, complex, expert]
- appropriateness_reasoning: brief explanation`;
    }
    /**
     * Parse Ollama's raw response
     */
    parseOllamaResponse(responseText, call) {
        let analysis = {};
        // Try to extract JSON from response
        try {
            const jsonMatch = responseText.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                analysis = JSON.parse(jsonMatch[0]);
            }
        }
        catch {
            // Fall back to defaults
        }
        const isAppropriate = analysis.is_model_appropriate !== false;
        return {
            verdict: isAppropriate ? 'APPROPRIATE' : 'OVERKILL',
            confidence: 0.85,
            currentModel: {
                modelId: call.modelId,
                provider: call.provider,
                estimatedCostPer1k: this.estimateCost(call.modelId),
                latencyTier: this.estimateLatencyTier(call.modelId)
            },
            recommendedAlternative: !isAppropriate ? {
                modelId: this.suggestAlternative(call),
                provider: call.provider,
                estimatedCostPer1k: 0.15,
                latencyTier: 'fast'
            } : null,
            costSavingsPercent: !isAppropriate ? 80 : 0,
            latencySavingsMs: !isAppropriate ? 500 : 0,
            reasoning: analysis.appropriateness_reasoning || 'Analysis complete',
            lineNumber: call.lineNumber,
            rawCode: call.rawCallCode
        };
    }
    /**
     * Build the analysis payload in the format expected by the model
     */
    buildPayload(call) {
        return {
            model_used: call.modelId,
            provider: call.provider,
            context: {
                has_system_prompt: call.hasSystemPrompt,
                has_tools: call.hasTools,
                tool_count: call.toolCount,
                has_structured_output: call.hasStructuredOutput,
                max_tokens: call.estimatedMaxTokens,
                temperature: call.temperature,
            },
            code_snippet: call.surroundingCode
        };
    }
    /**
     * Parse the model's response into AnalysisResult
     */
    parseResponse(response, call) {
        return {
            verdict: response.verdict || 'APPROPRIATE',
            confidence: response.confidence || 0.5,
            currentModel: {
                modelId: call.modelId,
                provider: call.provider,
                estimatedCostPer1k: response.current_cost || this.estimateCost(call.modelId),
                latencyTier: response.current_latency_tier || 'medium'
            },
            recommendedAlternative: response.recommended_alternative ? {
                modelId: response.recommended_alternative.model_id,
                provider: response.recommended_alternative.provider,
                estimatedCostPer1k: response.recommended_alternative.cost,
                latencyTier: response.recommended_alternative.latency_tier
            } : null,
            costSavingsPercent: response.cost_savings_percent || 0,
            latencySavingsMs: response.latency_savings_ms || 0,
            reasoning: response.reasoning || '',
            lineNumber: call.lineNumber,
            rawCode: call.rawCallCode
        };
    }
    /**
     * Fallback rule-based analysis when server is unavailable
     */
    fallbackAnalysis(call) {
        const isOverkill = this.isLikelyOverkill(call);
        return {
            verdict: isOverkill ? 'OVERKILL' : 'APPROPRIATE',
            confidence: 0.6, // Lower confidence for rule-based
            currentModel: {
                modelId: call.modelId,
                provider: call.provider,
                estimatedCostPer1k: this.estimateCost(call.modelId),
                latencyTier: this.estimateLatencyTier(call.modelId)
            },
            recommendedAlternative: isOverkill ? {
                modelId: this.suggestAlternative(call),
                provider: call.provider,
                estimatedCostPer1k: 0.15, // Estimate for cheaper model
                latencyTier: 'fast'
            } : null,
            costSavingsPercent: isOverkill ? 80 : 0,
            latencySavingsMs: isOverkill ? 500 : 0,
            reasoning: isOverkill
                ? 'Task appears simple and may not require a frontier model'
                : 'Model selection appears appropriate for the task',
            lineNumber: call.lineNumber,
            rawCode: call.rawCallCode
        };
    }
    /**
     * Simple heuristic to detect likely overkill
     */
    isLikelyOverkill(call) {
        const frontierModels = [
            'gpt-4o', 'gpt-4-turbo', 'gpt-4',
            'claude-3-opus', 'claude-3-5-sonnet',
            'gemini-1.5-pro', 'gemini-ultra'
        ];
        const isFrontierModel = frontierModels.some(m => call.modelId.toLowerCase().includes(m.toLowerCase()));
        // Frontier model with no tools, no structured output = likely overkill
        const isSimpleTask = !call.hasTools && !call.hasStructuredOutput;
        return isFrontierModel && isSimpleTask;
    }
    /**
     * Suggest a cheaper alternative model
     */
    suggestAlternative(call) {
        const alternatives = {
            'gpt-4o': 'gpt-4o-mini',
            'gpt-4': 'gpt-3.5-turbo',
            'claude-3-opus': 'claude-3-haiku',
            'claude-3-5-sonnet': 'claude-3-haiku',
            'gemini-1.5-pro': 'gemini-1.5-flash',
        };
        for (const [frontier, alternative] of Object.entries(alternatives)) {
            if (call.modelId.toLowerCase().includes(frontier.toLowerCase())) {
                return alternative;
            }
        }
        return 'gpt-4o-mini'; // Default suggestion
    }
    /**
     * Estimate cost per 1k tokens for a model
     */
    estimateCost(modelId) {
        const costs = {
            'gpt-4o': 5.0,
            'gpt-4o-mini': 0.15,
            'gpt-4-turbo': 10.0,
            'gpt-3.5-turbo': 0.5,
            'claude-3-opus': 15.0,
            'claude-3-sonnet': 3.0,
            'claude-3-haiku': 0.25,
            'gemini-1.5-pro': 3.5,
            'gemini-1.5-flash': 0.075,
        };
        for (const [model, cost] of Object.entries(costs)) {
            if (modelId.toLowerCase().includes(model.toLowerCase())) {
                return cost;
            }
        }
        return 1.0; // Default
    }
    /**
     * Estimate latency tier for a model
     */
    estimateLatencyTier(modelId) {
        const slowModels = ['opus', 'gpt-4-turbo', 'gemini-ultra'];
        const fastModels = ['mini', 'flash', 'haiku', '3.5-turbo'];
        const modelLower = modelId.toLowerCase();
        if (slowModels.some(m => modelLower.includes(m)))
            return 'slow';
        if (fastModels.some(m => modelLower.includes(m)))
            return 'fast';
        return 'medium';
    }
}
exports.AIGaugeClient = AIGaugeClient;
//# sourceMappingURL=aiGaugeClient.js.map