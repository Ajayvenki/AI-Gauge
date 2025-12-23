"use strict";
/**
 * LLM Call Detector
 *
 * Detects LLM API calls in source code using pattern matching and AST parsing.
 * Supports OpenAI, Anthropic, Google, and other major providers.
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.LLMCallDetector = void 0;
const vscode = __importStar(require("vscode"));
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
class LLMCallDetector {
    /**
     * Detect all LLM API calls in a document
     */
    detectCalls(document) {
        const text = document.getText();
        const language = this.mapLanguage(document.languageId);
        const calls = [];
        // Check each provider's patterns
        for (const [provider, patterns] of Object.entries(LLM_PATTERNS)) {
            const pattern = patterns[language];
            if (!pattern)
                continue;
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
                const call = this.extractCallMetadata(match[0], surroundingCode, provider, match[1], // model ID from capture group
                new vscode.Range(startPos, endPos), startPos.line + 1);
                calls.push(call);
            }
        }
        return calls;
    }
    /**
     * Extract metadata from a detected LLM call
     */
    extractCallMetadata(rawCode, context, provider, modelId, range, lineNumber) {
        // Count tools if present
        let toolCount = 0;
        const toolsMatch = context.match(/tools\s*[=:]\s*\[([^\]]*)\]/s);
        if (toolsMatch) {
            // Count function definitions in tools array
            toolCount = (toolsMatch[1].match(/["']name["']\s*[=:]/g) || []).length;
        }
        // Extract max_tokens
        let maxTokens = null;
        const maxTokensMatch = context.match(FEATURE_PATTERNS.maxTokens);
        if (maxTokensMatch) {
            maxTokens = parseInt(maxTokensMatch[1] || maxTokensMatch[2], 10);
        }
        // Extract temperature
        let temperature = null;
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
    getLines(document, start, end) {
        const lines = [];
        for (let i = start; i <= end; i++) {
            lines.push(document.lineAt(i).text);
        }
        return lines.join('\n');
    }
    /**
     * Map VS Code language ID to our internal language type
     */
    mapLanguage(languageId) {
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
exports.LLMCallDetector = LLMCallDetector;
//# sourceMappingURL=llmCallDetector.js.map