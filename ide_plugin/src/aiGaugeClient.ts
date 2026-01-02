/**
 * AI-Gauge Client
 * 
 * Communicates with the local inference server or directly with Ollama
 * to analyze LLM calls and return optimization recommendations.
 */

import { DetectedLLMCall } from './llmCallDetector';

export interface AnalysisResult {
    verdict: 'OVERKILL' | 'APPROPRIATE' | 'UNDERPOWERED';
    confidence: number;
    
    currentModel: {
        modelId: string;
        provider: string;
        estimatedCostPer1k: number;
        latencyTier: string;
    };
    
    recommendedAlternative: {
        modelId: string;
        provider: string;
        estimatedCostPer1k: number;
        latencyTier: string;
    } | null;
    
    costSavingsPercent: number;
    latencySavingsMs: number;
    
    // Carbon emissions
    currentCarbonGrams: number;
    alternativeCarbonGrams: number | null;
    carbonSavingsPercent: number;
    
    reasoning: string;
    
    // Original call info for diagnostics
    lineNumber: number;
    rawCode: string;
}

export interface ClientConfig {
    serverUrl: string;
    useOllamaDirect: boolean;  // Kept for interface compatibility but always false
    ollamaUrl: string;  // Not used in server mode
    ollamaModel: string;  // Not used in server mode
}

export class AIGaugeClient {
    private config: ClientConfig;

    constructor(config: ClientConfig) {
        this.config = config;
    }

    /**
     * Update configuration (called when VS Code settings change)
     */
    updateConfig(config: Partial<ClientConfig>): void {
        this.config = { ...this.config, ...config };
    }

    /**
     * Analyze a detected LLM call
     */
    async analyze(call: DetectedLLMCall): Promise<AnalysisResult> {
        console.log('AI-Gauge: Analyzing call with server mode');
        return await this.analyzeWithServer(call);
    }

    /**
     * Analyze using the inference server (recommended)
     */
    private async analyzeWithServer(call: DetectedLLMCall): Promise<AnalysisResult> {
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
     * Build the analysis payload in the format expected by the model
     */
    private buildPayload(call: DetectedLLMCall): object {
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
    private parseResponse(response: any, call: DetectedLLMCall): AnalysisResult {
        // Server returns camelCase (currentModel) - handle both formats
        const serverCurrentModel = response.currentModel || {};
        const serverAlt = response.recommendedAlternative || response.recommended_alternative;
        
        // Get cost from server response, fallback to defaults
        const currentCost = Number(serverCurrentModel.estimatedCostPer1k) || 
                           Number(response.current_cost) || 
                           this.getDefaultCost(call.modelId);
        
        const carbonFactor = Number(response.carbonFactor) || 1.0;
        const carbonGrams = currentCost > 0 ? (carbonFactor * currentCost * 0.001) : 0.028;
        
        return {
            verdict: response.verdict || 'APPROPRIATE',
            confidence: Number(response.confidence) || 0.5,
            
            currentModel: {
                modelId: call.modelId,
                provider: serverCurrentModel.provider || call.provider || 'openai',
                estimatedCostPer1k: currentCost,
                latencyTier: serverCurrentModel.latencyTier || response.current_latency_tier || 'medium'
            },
            
            recommendedAlternative: serverAlt ? {
                modelId: serverAlt.modelId || serverAlt.model_id || 'unknown',
                provider: serverAlt.provider || 'unknown',
                estimatedCostPer1k: Number(serverAlt.estimatedCostPer1k) || Number(serverAlt.cost) || 0.1,
                latencyTier: serverAlt.latencyTier || serverAlt.latency_tier || 'fast'
            } : null,
            
            costSavingsPercent: Number(response.costSavingsPercent) || Number(response.cost_savings_percent) || 0,
            latencySavingsMs: Number(response.latencySavingsMs) || Number(response.latency_savings_ms) || 0,
            currentCarbonGrams: carbonGrams,
            alternativeCarbonGrams: serverAlt ? (carbonGrams * 0.5) : null,
            carbonSavingsPercent: serverAlt ? 50 : 0,
            
            reasoning: response.reasoning || response.summary || '',
            
            lineNumber: call.lineNumber,
            rawCode: call.rawCallCode
        };
    }

    /**
     * Get default cost estimate for a model (synchronous fallback)
     */
    private getDefaultCost(modelId: string): number {
        const costs: Record<string, number> = {
            'gpt-4': 30.0,
            'gpt-4o': 5.0,
            'gpt-4o-mini': 0.15,
            'gpt-3.5-turbo': 0.5,
            'claude-3-opus': 15.0,
            'claude-3-5-sonnet': 3.0,
            'claude-3-haiku': 0.25,
            'gemini-1.5-pro': 3.5,
            'gemini-1.5-flash': 0.35,
        };
        
        const lowerModel = modelId.toLowerCase();
        for (const [key, cost] of Object.entries(costs)) {
            if (lowerModel.includes(key.toLowerCase())) {
                return cost;
            }
        }
        return 1.0; // Default fallback
    }

    /**
     * Suggest a cheaper alternative by querying the backend
     */
    private async suggestAlternative(call: DetectedLLMCall): Promise<string> {
        try {
            const response = await fetch(`${this.config.serverUrl}/models/${encodeURIComponent(call.modelId)}/alternatives`);
            if (response.ok) {
                const data = await response.json();
                const alternatives = data.alternatives || [];
                if (alternatives.length > 0) {
                    return alternatives[0].model_id;
                }
            }
        } catch (error) {
            console.warn('Failed to get model alternatives from backend:', error);
        }
        
        // Fallback alternatives
        const alternatives: Record<string, string> = {
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
     * Estimate cost per 1k tokens by querying the backend
     */
    private async estimateCost(modelId: string): Promise<number> {
        try {
            const response = await fetch(`${this.config.serverUrl}/models/${encodeURIComponent(modelId)}/cost`);
            if (response.ok) {
                const data = await response.json();
                return data.estimated_cost_per_1k || 1.0;
            }
        } catch (error) {
            console.warn('Failed to get model cost from backend:', error);
        }
        return 1.0; // Fallback
    }

    /**
     * Estimate latency tier by querying the backend
     */
    private async estimateLatencyTier(modelId: string): Promise<string> {
        try {
            const response = await fetch(`${this.config.serverUrl}/models/${encodeURIComponent(modelId)}`);
            if (response.ok) {
                const data = await response.json();
                return data.latency_tier || 'medium';
            }
        } catch (error) {
            console.warn('Failed to get model latency from backend:', error);
        }
        return 'medium'; // Fallback
    }

    /**
     * Get the tier of a model by querying the backend
     */
    private async getModelTier(modelId: string): Promise<string> {
        try {
            const response = await fetch(`${this.config.serverUrl}/models/${encodeURIComponent(modelId)}/tier`);
            if (response.ok) {
                const data = await response.json();
                return data.tier || 'standard';
            }
        } catch (error) {
            console.warn('Failed to get model tier from backend:', error);
        }
        return 'standard'; // Fallback
    }

    /**
     * Calculate latency savings based on tier differences
     */
    private calculateLatencySavings(currentLatency: string, alternativeLatency: string): number {
        if (!alternativeLatency) return 0;
        
        const latencyValues = {
            'ultra-fast': 100,
            'fast': 200,
            'medium': 400,
            'slow': 1000
        };
        
        const current = latencyValues[currentLatency as keyof typeof latencyValues] || 400;
        const alternative = latencyValues[alternativeLatency as keyof typeof latencyValues] || 200;
        
        return Math.max(0, current - alternative);
    }

    /**
     * Check if model is overkill based on tier comparison
     */
    private isModelOverkill(currentModelTier: string, minimumRequiredTier: string): boolean {
        const tierRankings = {
            'budget': 1,
            'standard': 2, 
            'premium': 3,
            'frontier': 4
        };
        
        const currentRank = tierRankings[currentModelTier as keyof typeof tierRankings] || 2;
        const requiredRank = tierRankings[minimumRequiredTier as keyof typeof tierRankings] || 2;
        
        return currentRank > requiredRank;
    }
}
