"use strict";
/**
 * Diagnostics Provider
 *
 * Shows VS Code diagnostics (squiggly lines) for LLM calls that are identified as overkill.
 * Provides quick-fix code actions to apply recommendations.
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
exports.DiagnosticsProvider = void 0;
const vscode = __importStar(require("vscode"));
class DiagnosticsProvider {
    constructor() {
        this.recommendations = new Map();
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('ai-gauge');
        // Register code action provider for quick fixes
        this.codeActionProvider = vscode.languages.registerCodeActionsProvider([{ language: 'python' }, { language: 'javascript' }, { language: 'typescript' }], new AIGaugeCodeActionProvider(this.recommendations), { providedCodeActionKinds: [vscode.CodeActionKind.QuickFix] });
    }
    /**
     * Update diagnostics for a document
     */
    updateDiagnostics(uri, analyses) {
        const diagnostics = [];
        // Store recommendations for code actions
        this.recommendations.set(uri.toString(), analyses);
        for (const analysis of analyses) {
            if (analysis.verdict !== 'OVERKILL')
                continue;
            const range = new vscode.Range(analysis.lineNumber - 1, 0, analysis.lineNumber - 1, 1000);
            const message = this.formatDiagnosticMessage(analysis);
            const diagnostic = new vscode.Diagnostic(range, message, vscode.DiagnosticSeverity.Information);
            diagnostic.code = 'ai-gauge-overkill';
            diagnostic.source = 'AI-Gauge';
            diagnostics.push(diagnostic);
        }
        this.diagnosticCollection.set(uri, diagnostics);
    }
    /**
     * Clear diagnostics for a document
     */
    clearDiagnostics(uri) {
        this.diagnosticCollection.delete(uri);
        this.recommendations.delete(uri.toString());
    }
    /**
     * Format the diagnostic message
     */
    formatDiagnosticMessage(analysis) {
        const alt = analysis.recommendedAlternative;
        if (!alt) {
            return `Model "${analysis.currentModel.modelId}" may be overkill for this task.`;
        }
        const parts = [
            `💡 Model "${analysis.currentModel.modelId}" may be overkill.`,
            `Consider "${alt.modelId}" for ${analysis.costSavingsPercent}% cost savings.`
        ];
        if (analysis.latencySavingsMs > 0) {
            parts.push(`(${analysis.latencySavingsMs}ms faster)`);
        }
        if (analysis.carbonSavingsPercent > 0) {
            parts.push(`🌱 ${analysis.carbonSavingsPercent}% less CO₂`);
        }
        return parts.join(' ');
    }
    dispose() {
        this.diagnosticCollection.dispose();
        this.codeActionProvider.dispose();
    }
}
exports.DiagnosticsProvider = DiagnosticsProvider;
/**
 * Code Action Provider for AI-Gauge quick fixes
 */
class AIGaugeCodeActionProvider {
    constructor(recommendations) {
        this.recommendations = recommendations;
    }
    provideCodeActions(document, range, context) {
        const actions = [];
        const analyses = this.recommendations.get(document.uri.toString());
        if (!analyses)
            return actions;
        for (const analysis of analyses) {
            if (analysis.verdict !== 'OVERKILL' || !analysis.recommendedAlternative)
                continue;
            // Check if this diagnostic is in range
            if (analysis.lineNumber - 1 !== range.start.line)
                continue;
            // Create quick fix action
            const action = new vscode.CodeAction(`Replace with ${analysis.recommendedAlternative.modelId} (${analysis.costSavingsPercent}% savings)`, vscode.CodeActionKind.QuickFix);
            action.edit = new vscode.WorkspaceEdit();
            // Find and replace the model name in the code
            const line = document.lineAt(analysis.lineNumber - 1);
            const oldModel = analysis.currentModel.modelId;
            const newModel = analysis.recommendedAlternative.modelId;
            const newText = line.text.replace(new RegExp(`["']${this.escapeRegex(oldModel)}["']`, 'g'), `"${newModel}"`);
            action.edit.replace(document.uri, line.range, newText);
            action.isPreferred = true;
            actions.push(action);
            // Add "Learn more" action
            const learnMore = new vscode.CodeAction('AI-Gauge: Learn more about this recommendation', vscode.CodeActionKind.Empty);
            learnMore.command = {
                title: 'Learn More',
                command: 'vscode.open',
                arguments: [vscode.Uri.parse('https://github.com/your-org/ai-gauge#recommendations')]
            };
            actions.push(learnMore);
        }
        return actions;
    }
    escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
}
//# sourceMappingURL=diagnosticsProvider.js.map