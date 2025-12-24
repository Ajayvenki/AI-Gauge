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
     * Analyze by calling Ollama directly (no inference server needed)
     */
    private analyzeWithOllama;
    /**
     * Build prompt for direct Ollama analysis
     */
    private buildOllamaPrompt;
    /**
     * Parse Ollama's raw response
     */
    private parseOllamaResponse;
    /**
     * Build the analysis payload in the format expected by the model
     */
    private buildPayload;
    /**
     * Parse the model's response into AnalysisResult
     */
    private parseResponse;
    /**
     * Fallback rule-based analysis when server is unavailable
     */
    private fallbackAnalysis;
    /**
     * Simple heuristic to detect likely overkill
     */
    private isLikelyOverkill;
    /**
     * Suggest a cheaper alternative model
     */
    private suggestAlternative;
    /**
     * Estimate cost per 1k tokens for a model
     */
    private estimateCost;
    /**
     * Estimate latency tier for a model
     */
    private estimateLatencyTier;
}
