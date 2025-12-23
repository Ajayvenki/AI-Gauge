/**
 * Diagnostics Provider
 *
 * Shows VS Code diagnostics (squiggly lines) for LLM calls that are identified as overkill.
 * Provides quick-fix code actions to apply recommendations.
 */
import * as vscode from 'vscode';
import { AnalysisResult } from './aiGaugeClient';
export declare class DiagnosticsProvider implements vscode.Disposable {
    private diagnosticCollection;
    private codeActionProvider;
    private recommendations;
    constructor();
    /**
     * Update diagnostics for a document
     */
    updateDiagnostics(uri: vscode.Uri, analyses: AnalysisResult[]): void;
    /**
     * Clear diagnostics for a document
     */
    clearDiagnostics(uri: vscode.Uri): void;
    /**
     * Format the diagnostic message
     */
    private formatDiagnosticMessage;
    dispose(): void;
}
