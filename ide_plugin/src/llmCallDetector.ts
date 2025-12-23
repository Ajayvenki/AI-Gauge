/**
 * LLM Call Detector
 * 
 * Detects LLM API calls in source code using pattern matching and AST parsing.
 * Supports OpenAI, Anthropic, Google, and other major providers.
 */

import * as vscode from 'vscode';

export interface DetectedLLMCall {
    // Location in source
    range: vscode.Range;
    lineNumber: number;
    
    // Detected model info
    provider: string;           // 'openai', 'anthropic', 'google', etc.
    modelId: string;            // 'gpt-4o', 'claude-3-opus', etc.
    
    // Call metadata (extracted from code)
    hasSystemPrompt: boolean;
    hasTools: boolean;
    toolCount: number;
    hasStructuredOutput: boolean;
    estimatedMaxTokens: number | null;
    temperature: number | null;
    
    // Context
    surroundingCode: string;    // Code context for analysis
    rawCallCode: string;        // The actual API call code
}

// Patterns to detect LLM API calls
const LLM_PATTERNS = {
    openai: {
        // client.chat.completions.create(model="gpt-4o", ...)
        python: /(?:client|openai)\.chat\.completions\.create\s*\([^)]*model\s*=\s*["']([^"']+)["'][^)]*\)/g,
        typescript: /(?:client|openai)\.chat\.completions\.create\s*\(\s*\{[^}]*model:\s*["']([^"']+)["'][^}]*\}\s*\)/g,
    },
    anthropic: {
        // client.messages.create(model="claude-3-opus", ...)
        python: /(?:client|anthropic)\.messages\.create\s*\([^)]*model\s*=\s*["']([^"']+)["'][^)]*\)/g,
        typescript: /(?:client|anthropic)\.messages\.create\s*\(\s*\{[^}]*model:\s*["']([^"']+)["'][^}]*\}\s*\)/g,
    },
    google: {
        // genai.GenerativeModel("gemini-pro")
        python: /GenerativeModel\s*\(\s*["']([^"']+)["']/g,
        typescript: /getGenerativeModel\s*\(\s*\{[^}]*model:\s*["']([^"']+)["'][^}]*\}\s*\)/g,
    }
};

// Patterns to detect features
const FEATURE_PATTERNS = {
    systemPrompt: /system|role\s*[=:]\s*["']system["']/i,
    tools: /tools\s*[=:]\s*\[|functions\s*[=:]\s*\[|function_call|tool_choice/i,
    structuredOutput: /response_format|json_schema|structured_output/i,
    maxTokens: /max_tokens\s*[=:]\s*(\d+)|maxTokens\s*[=:]\s*(\d+)/,
    temperature: /temperature\s*[=:]\s*([\d.]+)/,
};

export class LLMCallDetector {
    
    /**
     * Detect all LLM API calls in a document
     */
    detectCalls(document: vscode.TextDocument): DetectedLLMCall[] {
        const text = document.getText();
        const language = this.mapLanguage(document.languageId);
        const calls: DetectedLLMCall[] = [];

        // Check each provider's patterns
        for (const [provider, patterns] of Object.entries(LLM_PATTERNS)) {
            const pattern = patterns[language as keyof typeof patterns];
            if (!pattern) continue;

            // Reset regex state
            pattern.lastIndex = 0;
            let match;

            while ((match = pattern.exec(text)) !== null) {
                const startPos = document.positionAt(match.index);
                const endPos = document.positionAt(match.index + match[0].length);
                
                // Get surrounding context (5 lines before and after)
                const contextStart = Math.max(0, startPos.line - 5);
                const contextEnd = Math.min(document.lineCount - 1, endPos.line + 5);
                const surroundingCode = this.getLines(document, contextStart, contextEnd);

                const call = this.extractCallMetadata(
                    match[0],
                    surroundingCode,
                    provider,
                    match[1], // model ID from capture group
                    new vscode.Range(startPos, endPos),
                    startPos.line + 1
                );
                
                calls.push(call);
            }
        }

        return calls;
    }

    /**
     * Extract metadata from a detected LLM call
     */
    private extractCallMetadata(
        rawCode: string,
        context: string,
        provider: string,
        modelId: string,
        range: vscode.Range,
        lineNumber: number
    ): DetectedLLMCall {
        // Count tools if present
        let toolCount = 0;
        const toolsMatch = context.match(/tools\s*[=:]\s*\[([^\]]*)\]/s);
        if (toolsMatch) {
            // Count function definitions in tools array
            toolCount = (toolsMatch[1].match(/["']name["']\s*[=:]/g) || []).length;
        }

        // Extract max_tokens
        let maxTokens: number | null = null;
        const maxTokensMatch = context.match(FEATURE_PATTERNS.maxTokens);
        if (maxTokensMatch) {
            maxTokens = parseInt(maxTokensMatch[1] || maxTokensMatch[2], 10);
        }

        // Extract temperature
        let temperature: number | null = null;
        const tempMatch = context.match(FEATURE_PATTERNS.temperature);
        if (tempMatch) {
            temperature = parseFloat(tempMatch[1]);
        }

        return {
            range,
            lineNumber,
            provider,
            modelId,
            hasSystemPrompt: FEATURE_PATTERNS.systemPrompt.test(context),
            hasTools: FEATURE_PATTERNS.tools.test(context),
            toolCount,
            hasStructuredOutput: FEATURE_PATTERNS.structuredOutput.test(context),
            estimatedMaxTokens: maxTokens,
            temperature,
            surroundingCode: context,
            rawCallCode: rawCode
        };
    }

    /**
     * Get lines from document as string
     */
    private getLines(document: vscode.TextDocument, start: number, end: number): string {
        const lines: string[] = [];
        for (let i = start; i <= end; i++) {
            lines.push(document.lineAt(i).text);
        }
        return lines.join('\n');
    }

    /**
     * Map VS Code language ID to our internal language type
     */
    private mapLanguage(languageId: string): string {
        switch (languageId) {
            case 'python':
                return 'python';
            case 'javascript':
            case 'typescript':
            case 'typescriptreact':
            case 'javascriptreact':
                return 'typescript';
            default:
                return 'unknown';
        }
    }
}
