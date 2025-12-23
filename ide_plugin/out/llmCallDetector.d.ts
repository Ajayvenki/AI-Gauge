/**
 * LLM Call Detector
 *
 * Detects LLM API calls in source code using pattern matching and AST parsing.
 * Supports OpenAI, Anthropic, Google, and other major providers.
 */
import * as vscode from 'vscode';
export interface DetectedLLMCall {
    range: vscode.Range;
    lineNumber: number;
    provider: string;
    modelId: string;
    hasSystemPrompt: boolean;
    hasTools: boolean;
    toolCount: number;
    hasStructuredOutput: boolean;
    estimatedMaxTokens: number | null;
    temperature: number | null;
    surroundingCode: string;
    rawCallCode: string;
}
export declare class LLMCallDetector {
    /**
     * Detect all LLM API calls in a document
     */
    detectCalls(document: vscode.TextDocument): DetectedLLMCall[];
    /**
     * Extract metadata from a detected LLM call
     */
    private extractCallMetadata;
    /**
     * Get lines from document as string
     */
    private getLines;
    /**
     * Map VS Code language ID to our internal language type
     */
    private mapLanguage;
}
