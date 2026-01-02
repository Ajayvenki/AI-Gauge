/**
 * Diagnostics Provider
 * 
 * Shows VS Code diagnostics (squiggly lines) for LLM calls that are identified as overkill.
 * Provides quick-fix code actions to apply recommendations.
 */

import * as vscode from 'vscode';
import { AnalysisResult } from './aiGaugeClient';

export class DiagnosticsProvider implements vscode.Disposable {
    private diagnosticCollection: vscode.DiagnosticCollection;
    private codeActionProvider: vscode.Disposable;
    private recommendations: Map<string, AnalysisResult[]> = new Map();

    constructor() {
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('ai-gauge');
        
        // Register code action provider for quick fixes
        this.codeActionProvider = vscode.languages.registerCodeActionsProvider(
            [{ language: 'python' }, { language: 'javascript' }, { language: 'typescript' }],
            new AIGaugeCodeActionProvider(this.recommendations),
            { providedCodeActionKinds: [vscode.CodeActionKind.QuickFix] }
        );
    }

    /**
     * Update diagnostics for a document
     */
    updateDiagnostics(uri: vscode.Uri, analyses: AnalysisResult[]): void {
        const diagnostics: vscode.Diagnostic[] = [];
        
        // Store recommendations for code actions
        this.recommendations.set(uri.toString(), analyses);

        for (const analysis of analyses) {
            if (analysis.verdict !== 'OVERKILL') continue;

            const range = new vscode.Range(
                analysis.lineNumber - 1, 0,
                analysis.lineNumber - 1, 1000
            );

            const message = this.formatDiagnosticMessage(analysis);
            
            const diagnostic = new vscode.Diagnostic(
                range,
                message,
                vscode.DiagnosticSeverity.Information
            );
            
            diagnostic.code = 'ai-gauge-overkill';
            diagnostic.source = 'AI-Gauge';
            
            diagnostics.push(diagnostic);
        }

        this.diagnosticCollection.set(uri, diagnostics);
    }

    /**
     * Clear diagnostics for a document
     */
    clearDiagnostics(uri: vscode.Uri): void {
        this.diagnosticCollection.delete(uri);
        this.recommendations.delete(uri.toString());
    }

    /**
     * Format the diagnostic message
     */
    private formatDiagnosticMessage(analysis: AnalysisResult): string {
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

    dispose(): void {
        this.diagnosticCollection.dispose();
        this.codeActionProvider.dispose();
    }
}

/**
 * Code Action Provider for AI-Gauge quick fixes
 */
class AIGaugeCodeActionProvider implements vscode.CodeActionProvider {
    constructor(private recommendations: Map<string, AnalysisResult[]>) {}

    provideCodeActions(
        document: vscode.TextDocument,
        range: vscode.Range,
        context: vscode.CodeActionContext
    ): vscode.CodeAction[] {
        const actions: vscode.CodeAction[] = [];
        
        const analyses = this.recommendations.get(document.uri.toString());
        if (!analyses) return actions;

        for (const analysis of analyses) {
            if (analysis.verdict !== 'OVERKILL' || !analysis.recommendedAlternative) continue;
            
            // Check if this diagnostic is in range
            if (analysis.lineNumber - 1 !== range.start.line) continue;

            // Create quick fix action
            const action = new vscode.CodeAction(
                `Replace with ${analysis.recommendedAlternative.modelId} (${analysis.costSavingsPercent}% savings)`,
                vscode.CodeActionKind.QuickFix
            );

            action.edit = new vscode.WorkspaceEdit();
            
            // Find and replace the model name in the code
            const line = document.lineAt(analysis.lineNumber - 1);
            const oldModel = analysis.currentModel.modelId;
            const newModel = analysis.recommendedAlternative.modelId;
            
            const newText = line.text.replace(
                new RegExp(`["']${this.escapeRegex(oldModel)}["']`, 'g'),
                `"${newModel}"`
            );
            
            action.edit.replace(document.uri, line.range, newText);
            action.isPreferred = true;

            actions.push(action);

            // Add "Learn more" action
            const learnMore = new vscode.CodeAction(
                'AI-Gauge: Learn more about this recommendation',
                vscode.CodeActionKind.Empty
            );
            learnMore.command = {
                title: 'Learn More',
                command: 'vscode.open',
                arguments: [vscode.Uri.parse('https://github.com/your-org/ai-gauge#recommendations')]
            };
            actions.push(learnMore);
        }

        return actions;
    }

    private escapeRegex(string: string): string {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
}
