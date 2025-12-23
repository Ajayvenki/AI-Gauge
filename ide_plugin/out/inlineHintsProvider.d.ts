/**
 * Inline Hints Provider
 *
 * Shows inline hints next to LLM calls with cost/performance information.
 * Example: client.chat.completions.create(...)  // $5.00/1k tokens • slow
 */
import * as vscode from 'vscode';
import { AnalysisResult } from './aiGaugeClient';
export declare class InlineHintsProvider implements vscode.InlayHintsProvider {
    private hints;
    /**
     * Update hints for a document
     */
    updateHints(uri: vscode.Uri, analyses: AnalysisResult[]): void;
    private _onDidChangeInlayHints;
    onDidChangeInlayHints: vscode.Event<void>;
    provideInlayHints(document: vscode.TextDocument, range: vscode.Range): vscode.InlayHint[];
    /**
     * Format the inline hint label
     */
    private formatHintLabel;
    /**
     * Format the hover tooltip
     */
    private formatTooltip;
    private getVerdictEmoji;
}
