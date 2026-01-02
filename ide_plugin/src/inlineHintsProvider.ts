/**
 * Inline Hints Provider
 * 
 * Shows inline hints next to LLM calls with cost/performance information.
 * Example: client.chat.completions.create(...)  // $5.00/1k tokens • slow
 */

import * as vscode from 'vscode';
import { AnalysisResult } from './aiGaugeClient';

export class InlineHintsProvider implements vscode.InlayHintsProvider {
    private hints: Map<string, AnalysisResult[]> = new Map();

    /**
     * Update hints for a document
     */
    updateHints(uri: vscode.Uri, analyses: AnalysisResult[]): void {
        this.hints.set(uri.toString(), analyses);
        // Trigger refresh - VS Code will call provideInlayHints again
        this._onDidChangeInlayHints.fire();
    }

    private _onDidChangeInlayHints = new vscode.EventEmitter<void>();
    onDidChangeInlayHints = this._onDidChangeInlayHints.event;

    provideInlayHints(
        document: vscode.TextDocument,
        range: vscode.Range
    ): vscode.InlayHint[] {
        const analyses = this.hints.get(document.uri.toString());
        if (!analyses) return [];

        const hints: vscode.InlayHint[] = [];

        for (const analysis of analyses) {
            // Only show hints in visible range
            if (analysis.lineNumber - 1 < range.start.line || 
                analysis.lineNumber - 1 > range.end.line) {
                continue;
            }

            const line = document.lineAt(analysis.lineNumber - 1);
            const position = new vscode.Position(analysis.lineNumber - 1, line.text.length);

            const label = this.formatHintLabel(analysis);
            const hint = new vscode.InlayHint(position, label, vscode.InlayHintKind.Parameter);
            
            hint.paddingLeft = true;
            hint.tooltip = this.formatTooltip(analysis);

            hints.push(hint);
        }

        return hints;
    }

    /**
     * Format the inline hint label
     */
    private formatHintLabel(analysis: AnalysisResult): string {
        const cost = `$${analysis.currentModel.estimatedCostPer1k.toFixed(2)}/1k`;
        const latency = analysis.currentModel.latencyTier;
        const carbon = `${analysis.currentCarbonGrams.toFixed(3)}g CO₂`;
        
        if (analysis.verdict === 'OVERKILL') {
            return `  ⚠️ ${cost} • ${carbon} → 💡 save ${analysis.costSavingsPercent}%`;
        }
        
        return `  ✓ ${cost} • ${carbon}`;
    }

    /**
     * Format the hover tooltip
     */
    private formatTooltip(analysis: AnalysisResult): vscode.MarkdownString {
        const md = new vscode.MarkdownString();
        md.isTrusted = true;

        md.appendMarkdown(`## AI-Gauge Analysis\n\n`);
        md.appendMarkdown(`**Verdict:** ${this.getVerdictEmoji(analysis.verdict)} ${analysis.verdict}\n\n`);
        md.appendMarkdown(`**Confidence:** ${(analysis.confidence * 100).toFixed(0)}%\n\n`);
        
        md.appendMarkdown(`### Current Model\n`);
        md.appendMarkdown(`- **Model:** ${analysis.currentModel.modelId}\n`);
        md.appendMarkdown(`- **Cost:** $${analysis.currentModel.estimatedCostPer1k.toFixed(2)}/1k tokens\n`);
        md.appendMarkdown(`- **Latency:** ${analysis.currentModel.latencyTier}\n`);
        md.appendMarkdown(`- **CO₂:** ${analysis.currentCarbonGrams.toFixed(3)}g per call\n\n`);

        if (analysis.recommendedAlternative) {
            md.appendMarkdown(`### Recommended Alternative\n`);
            md.appendMarkdown(`- **Model:** ${analysis.recommendedAlternative.modelId}\n`);
            md.appendMarkdown(`- **Cost:** $${analysis.recommendedAlternative.estimatedCostPer1k.toFixed(2)}/1k tokens\n`);
            md.appendMarkdown(`- **Latency:** ${analysis.recommendedAlternative.latencyTier}\n`);
            if (analysis.alternativeCarbonGrams) {
                md.appendMarkdown(`- **CO₂:** ${analysis.alternativeCarbonGrams.toFixed(3)}g per call\n\n`);
            }
            md.appendMarkdown(`### Savings\n`);
            md.appendMarkdown(`- **Cost:** ${analysis.costSavingsPercent}% reduction\n`);
            if (analysis.latencySavingsMs > 0) {
                md.appendMarkdown(`- **Latency:** ${analysis.latencySavingsMs}ms faster\n`);
            }
            if (analysis.carbonSavingsPercent > 0) {
                md.appendMarkdown(`- **CO₂:** ${analysis.carbonSavingsPercent}% reduction\n`);
            }
        }

        if (analysis.reasoning) {
            md.appendMarkdown(`\n---\n*${analysis.reasoning}*`);
        }

        return md;
    }

    private getVerdictEmoji(verdict: string): string {
        switch (verdict) {
            case 'OVERKILL': return '⚠️';
            case 'APPROPRIATE': return '✅';
            case 'UNDERPOWERED': return '⬆️';
            default: return '❓';
        }
    }
}
