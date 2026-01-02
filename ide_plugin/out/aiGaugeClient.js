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
            // Priority: Ollama (local) > Server
            console.log('AI-Gauge: Analyzing call with config:', this.config);
            if (this.config.useOllamaDirect) {
                console.log('AI-Gauge: Using Ollama direct analysis');
                return await this.analyzeWithOllama(call);
            }
            else {
                console.log('AI-Gauge: Using inference server analysis');
                return await this.analyzeWithServer(call);
            }
        }
        catch (error) {
            console.warn('AI-Gauge analysis failed, using fallback:', error);
            return this.fallbackAnalysis(call);
        }
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
        console.log('AI-Gauge: Ollama prompt:', prompt);
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
        console.log('AI-Gauge: Ollama response:', result.response);
        return this.parseOllamaResponse(result.response, call);
    }
    /**
     * Build prompt for direct Ollama analysis
     */
    buildOllamaPrompt(call) {
        return `You are an AI model analyzer. Analyze this LLM API call and determine if the model choice is appropriate.

TASK TO ANALYZE: ${call.surroundingCode}

MODEL BEING USED: ${call.modelId} (${call.provider})

INSTRUCTIONS:
- Read the actual task in the code above
- Determine if the model is appropriate for THIS SPECIFIC TASK
- Do NOT copy example responses
- Analyze the real complexity of the task described

Respond with ONLY a JSON object in this exact format:
{
  "is_model_appropriate": true_or_false_based_on_actual_task,
  "minimum_capable_tier": "budget|standard|premium|frontier",
  "actual_complexity": "trivial|simple|moderate|complex|expert", 
  "appropriateness_reasoning": "Your analysis of this specific task"
}`;
    }
    /**
     * Parse Ollama's raw response
     */
    parseOllamaResponse(responseText, call) {
        console.log('AI-Gauge: Parsing Ollama response:', responseText);
        let analysis = {};
        // Try to extract JSON from response
        try {
            const jsonMatch = responseText.match(/\{[\s\S]*?\}/);
            if (jsonMatch) {
                const cleanJson = jsonMatch[0]
                    .replace(/[\u0000-\u001F\u007F-\u009F]/g, '') // Remove control characters
                    .replace(/,(\s*[}\]])/g, '$1'); // Remove trailing commas
                analysis = JSON.parse(cleanJson);
                console.log('AI-Gauge: Parsed analysis:', analysis);
            }
        }
        catch (e) {
            console.log('AI-Gauge: JSON parse failed, trying keyword analysis:', e);
        }
        // Keyword-based fallback if JSON parsing failed
        if (!analysis.is_model_appropriate && analysis.is_model_appropriate !== false) {
            console.log('AI-Gauge: Using keyword-based analysis');
            const lowerResponse = responseText.toLowerCase();
            // Check for overkill indicators
            const overkillKeywords = ['overkill', 'too powerful', 'unnecessary', 'waste', 'expensive', 'simpler model', 'cheaper'];
            const appropriateKeywords = ['appropriate', 'suitable', 'necessary', 'complex', 'advanced'];
            const hasOverkill = overkillKeywords.some(k => lowerResponse.includes(k));
            const hasAppropriate = appropriateKeywords.some(k => lowerResponse.includes(k));
            if (hasOverkill && !hasAppropriate) {
                analysis.is_model_appropriate = false;
                analysis.appropriateness_reasoning = 'Keyword analysis detected overkill';
            }
            else if (hasAppropriate || lowerResponse.includes('appropriate')) {
                analysis.is_model_appropriate = true;
                analysis.appropriateness_reasoning = 'Keyword analysis detected appropriate';
            }
            else {
                // Default to overkill for frontier models if unclear
                analysis.is_model_appropriate = !this.isLikelyOverkill(call);
                analysis.appropriateness_reasoning = 'Default analysis based on model tier';
            }
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
            reasoning: analysis.appropriateness_reasoning || (isAppropriate ? 'Model appears appropriate for this task' : 'Model may be overkill for this task'),
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