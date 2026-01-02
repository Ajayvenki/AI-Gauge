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
    currentCarbonGrams: number;
    alternativeCarbonGrams: number | null;
    carbonSavingsPercent: number;
    reasoning: string;
    lineNumber: number;
    rawCode: string;
}
export interface ClientConfig {
    serverUrl: string;
    useOllamaDirect: boolean;
    ollamaUrl: string;
    ollamaModel: string;
}
export declare class AIGaugeClient {
    private config;
    constructor(config: ClientConfig);
    /**
     * Update configuration (called when VS Code settings change)
     */
    updateConfig(config: Partial<ClientConfig>): void;
    /**
     * Analyze a detected LLM call
     */
    analyze(call: DetectedLLMCall): Promise<AnalysisResult>;
    /**
     * Analyze using the inference server (recommended)
     */
    private analyzeWithServer;
    /**
     * Build the analysis payload in the format expected by the model
     */
    private buildPayload;
    /**
     * Parse the model's response into AnalysisResult
     */
    private parseResponse;
    /**
     * Suggest a cheaper alternative by querying the backend
     */
    private suggestAlternative;
    /**
     * Estimate cost per 1k tokens by querying the backend
     */
    private estimateCost;
    /**
     * Estimate latency tier by querying the backend
     */
    private estimateLatencyTier;
    /**
     * Get the tier of a model by querying the backend
     */
    private getModelTier;
    /**
     * Calculate latency savings based on tier differences
     */
    private calculateLatencySavings;
    /**
     * Check if model is overkill based on tier comparison
     */
    private isModelOverkill;
}
