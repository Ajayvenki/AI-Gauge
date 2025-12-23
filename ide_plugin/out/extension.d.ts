/**
 * AI-Gauge VS Code Extension
 *
 * Analyzes LLM API calls in code and provides cost optimization recommendations
 * using a fine-tuned Phi-3.5 model via HuggingFace, Ollama, or local inference server.
 */
import * as vscode from 'vscode';
export declare function activate(context: vscode.ExtensionContext): Promise<void>;
export declare function deactivate(): void;
