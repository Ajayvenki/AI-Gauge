"use strict";
/**
 * AI-Gauge VS Code Extension
 *
 * Analyzes LLM API calls in code and provides cost optimization recommendations
 * using a fine-tuned Phi-3.5 model via HuggingFace, Ollama, or local inference server.
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
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const llmCallDetector_1 = require("./llmCallDetector");
const aiGaugeClient_1 = require("./aiGaugeClient");
const diagnosticsProvider_1 = require("./diagnosticsProvider");
const inlineHintsProvider_1 = require("./inlineHintsProvider");
const cp = __importStar(require("child_process"));
const os = __importStar(require("os"));
let diagnosticsProvider;
let inlineHintsProvider;
let detector;
let client;
let statusBarItem;
function getClientConfig() {
    const config = vscode.workspace.getConfiguration('aiGauge');
    return {
        serverUrl: config.get('modelServerUrl') || 'http://localhost:8080',
        useOllamaDirect: config.get('useOllamaDirect') || false,
        ollamaUrl: config.get('ollamaUrl') || 'http://localhost:11434',
        ollamaModel: config.get('ollamaModel') || 'ai-gauge',
        // HuggingFace settings (cloud - simplest)
        useHuggingFace: config.get('useHuggingFace') || true,
        huggingFaceApiKey: config.get('huggingFaceApiKey') || '',
        huggingFaceModel: config.get('huggingFaceModel') || 'ajayvenkatesan/ai-gauge'
    };
}
async function activate(context) {
    console.log('AI-Gauge extension activated');
    // Initialize status bar
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBarItem.command = 'aiGauge.showSetupStatus';
    context.subscriptions.push(statusBarItem);
    // Check and setup dependencies
    const setupComplete = await performAutoSetup(context);
    if (!setupComplete) {
        // Show setup status in status bar
        updateStatusBar('setup-required');
        return; // Don't initialize other components if setup failed
    }
    // Initialize components
    const config = vscode.workspace.getConfiguration('aiGauge');
    client = new aiGaugeClient_1.AIGaugeClient(getClientConfig());
    detector = new llmCallDetector_1.LLMCallDetector();
    diagnosticsProvider = new diagnosticsProvider_1.DiagnosticsProvider();
    inlineHintsProvider = new inlineHintsProvider_1.InlineHintsProvider();
    // Watch for configuration changes
    context.subscriptions.push(vscode.workspace.onDidChangeConfiguration((e) => {
        if (e.affectsConfiguration('aiGauge')) {
            client.updateConfig(getClientConfig());
        }
    }));
    // Register commands
    context.subscriptions.push(vscode.commands.registerCommand('aiGauge.analyzeCurrentFile', () => analyzeCurrentFile()), vscode.commands.registerCommand('aiGauge.analyzeWorkspace', () => analyzeWorkspace()), vscode.commands.registerCommand('aiGauge.toggleRealTimeAnalysis', () => toggleRealTimeAnalysis()));
    // Register inline hints provider
    context.subscriptions.push(vscode.languages.registerInlayHintsProvider([{ language: 'python' }, { language: 'javascript' }, { language: 'typescript' }], inlineHintsProvider));
    // Watch for document changes if real-time analysis is enabled
    if (config.get('realTimeAnalysis')) {
        context.subscriptions.push(vscode.workspace.onDidChangeTextDocument((event) => {
            debounce(() => onDocumentChange(event), 1000);
        }));
    }
    // Analyze on file open
    context.subscriptions.push(vscode.window.onDidChangeActiveTextEditor((editor) => {
        if (editor && config.get('enabled')) {
            analyzeDocument(editor.document);
        }
    }));
    // Register setup status command
    context.subscriptions.push(vscode.commands.registerCommand('aiGauge.showSetupStatus', () => showSetupStatus()));
    updateStatusBar('ready');
}
/**
 * Perform automatic setup of dependencies
 */
async function performAutoSetup(context) {
    const setupState = context.globalState.get('aiGauge.setupComplete', false);
    if (setupState) {
        // Quick check if everything is still working
        const healthCheck = await checkDependenciesHealth();
        if (healthCheck.healthy) {
            return true;
        }
    }
    // Perform full setup check
    const dependencies = await checkDependencies();
    if (dependencies.ollamaInstalled && dependencies.modelExists && dependencies.pythonReady) {
        context.globalState.update('aiGauge.setupComplete', true);
        return true;
    }
    // Ask user if they want auto-setup
    const setupChoice = await vscode.window.showInformationMessage('AI-Gauge requires Ollama and the AI-Gauge model to be installed. Would you like to set this up automatically?', 'Yes, Setup Automatically', 'No, I\'ll do it manually');
    if (setupChoice !== 'Yes, Setup Automatically') {
        vscode.window.showWarningMessage('AI-Gauge setup cancelled. You can run setup manually or use the status bar to retry.');
        return false;
    }
    // Perform auto-setup
    const success = await runAutoSetup(dependencies);
    if (success) {
        context.globalState.update('aiGauge.setupComplete', true);
        vscode.window.showInformationMessage('AI-Gauge setup complete! You can now analyze your LLM calls.');
        return true;
    }
    else {
        vscode.window.showErrorMessage('AI-Gauge setup failed. Please check the output for details and try again.');
        return false;
    }
}
/**
 * Check the health of existing dependencies
 */
async function checkDependenciesHealth() {
    try {
        // Check Ollama
        const ollamaRunning = await isOllamaRunning();
        if (!ollamaRunning) {
            return { healthy: false, details: 'Ollama not running' };
        }
        // Check model
        const modelExists = await checkModelExists();
        if (!modelExists) {
            return { healthy: false, details: 'AI-Gauge model not found' };
        }
        return { healthy: true, details: 'All dependencies healthy' };
    }
    catch (error) {
        return { healthy: false, details: `Health check failed: ${error}` };
    }
}
/**
 * Check all dependencies
 */
async function checkDependencies() {
    const ollamaInstalled = await checkOllamaInstalled();
    const modelExists = ollamaInstalled ? await checkModelExists() : false;
    const pythonReady = await checkPythonEnvironment();
    return { ollamaInstalled, modelExists, pythonReady };
}
/**
 * Run the automatic setup process
 */
async function runAutoSetup(dependencies) {
    return await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'Setting up AI-Gauge',
        cancellable: true
    }, async (progress, token) => {
        try {
            let step = 0;
            const totalSteps = 3;
            // Step 1: Install Ollama if needed
            if (!dependencies.ollamaInstalled) {
                progress.report({ message: 'Installing Ollama...', increment: (step / totalSteps) * 100 });
                if (token.isCancellationRequested)
                    return false;
                const ollamaSuccess = await installOllama();
                if (!ollamaSuccess) {
                    throw new Error('Failed to install Ollama');
                }
                step++;
            }
            // Step 2: Start Ollama service
            progress.report({ message: 'Starting Ollama service...', increment: (step / totalSteps) * 100 });
            if (token.isCancellationRequested)
                return false;
            const serviceSuccess = await startOllamaService();
            if (!serviceSuccess) {
                throw new Error('Failed to start Ollama service');
            }
            step++;
            // Step 3: Pull AI-Gauge model
            if (!dependencies.modelExists) {
                progress.report({ message: 'Downloading AI-Gauge model...', increment: (step / totalSteps) * 100 });
                if (token.isCancellationRequested)
                    return false;
                const modelSuccess = await pullAIGaugeModel();
                if (!modelSuccess) {
                    throw new Error('Failed to download AI-Gauge model');
                }
            }
            step++;
            // Step 4: Setup Python environment
            if (!dependencies.pythonReady) {
                progress.report({ message: 'Setting up Python environment...', increment: (100 / totalSteps) * step });
                if (token.isCancellationRequested)
                    return false;
                const pythonSuccess = await setupPythonEnvironment();
                if (!pythonSuccess) {
                    throw new Error('Failed to setup Python environment');
                }
            }
            progress.report({ message: 'Setup complete!', increment: 100 });
            return true;
        }
        catch (error) {
            console.error('Auto-setup failed:', error);
            vscode.window.showErrorMessage(`Setup failed: ${error}`);
            return false;
        }
    });
}
/**
 * Check if Ollama is installed
 */
async function checkOllamaInstalled() {
    return new Promise((resolve) => {
        cp.exec('ollama --version', (error) => {
            resolve(!error);
        });
    });
}
/**
 * Check if Ollama service is running
 */
async function isOllamaRunning() {
    return new Promise((resolve) => {
        cp.exec('curl -s http://localhost:11434/api/tags', (error, stdout) => {
            resolve(!error && stdout.includes('models'));
        });
    });
}
/**
 * Check if AI-Gauge model exists
 */
async function checkModelExists() {
    return new Promise((resolve) => {
        cp.exec('ollama list', (error, stdout) => {
            resolve(!error && stdout.includes('ai-gauge'));
        });
    });
}
/**
 * Check Python environment
 */
async function checkPythonEnvironment() {
    return new Promise((resolve) => {
        cp.exec('python3 --version', (error) => {
            resolve(!error);
        });
    });
}
/**
 * Install Ollama based on platform
 */
async function installOllama() {
    const platform = os.platform();
    let installCommand;
    switch (platform) {
        case 'darwin':
            installCommand = 'brew install ollama';
            break;
        case 'linux':
            installCommand = 'curl -fsSL https://ollama.ai/install.sh | sh';
            break;
        case 'win32':
            vscode.window.showInformationMessage('Please download Ollama from https://ollama.ai/download and install it manually.');
            return false;
        default:
            vscode.window.showErrorMessage(`Unsupported platform: ${platform}`);
            return false;
    }
    return new Promise((resolve) => {
        cp.exec(installCommand, (error) => {
            if (error) {
                console.error('Ollama install failed:', error);
                resolve(false);
            }
            else {
                resolve(true);
            }
        });
    });
}
/**
 * Start Ollama service
 */
async function startOllamaService() {
    return new Promise((resolve) => {
        // Start Ollama in background
        const ollamaProcess = cp.spawn('ollama', ['serve'], {
            detached: true,
            stdio: 'ignore'
        });
        ollamaProcess.unref();
        // Wait a bit for service to start
        setTimeout(() => {
            isOllamaRunning().then(resolve);
        }, 3000);
    });
}
/**
 * Pull AI-Gauge model
 */
async function pullAIGaugeModel() {
    return new Promise((resolve) => {
        cp.exec('ollama pull ajayvenki01/ai-gauge', (error) => {
            if (error) {
                console.error('Model pull failed:', error);
                resolve(false);
            }
            else {
                resolve(true);
            }
        });
    });
}
/**
 * Setup Python environment
 */
async function setupPythonEnvironment() {
    // For now, just check if pip is available
    // In a full implementation, this could install requirements.txt
    return new Promise((resolve) => {
        cp.exec('python3 -m pip --version', (error) => {
            resolve(!error);
        });
    });
}
/**
 * Show setup status
 */
function showSetupStatus() {
    // This could show a webview with detailed setup status
    vscode.window.showInformationMessage('AI-Gauge Setup Status', 'Check Output Panel for details');
}
/**
 * Update status bar
 */
function updateStatusBar(status) {
    if (!statusBarItem)
        return;
    switch (status) {
        case 'ready':
            statusBarItem.text = '$(check) AI-Gauge';
            statusBarItem.tooltip = 'AI-Gauge is ready';
            statusBarItem.backgroundColor = undefined;
            break;
        case 'setup-required':
            statusBarItem.text = '$(warning) AI-Gauge Setup Required';
            statusBarItem.tooltip = 'Click to setup AI-Gauge';
            statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.warningBackground');
            break;
        case 'setting-up':
            statusBarItem.text = '$(sync~spin) Setting up AI-Gauge...';
            statusBarItem.tooltip = 'AI-Gauge setup in progress';
            break;
        case 'error':
            statusBarItem.text = '$(error) AI-Gauge Error';
            statusBarItem.tooltip = 'Setup failed - click for details';
            statusBarItem.backgroundColor = new vscode.ThemeColor('statusBarItem.errorBackground');
            break;
    }
    statusBarItem.show();
}
/**
 * Analyze the currently active file for LLM calls
 */
async function analyzeCurrentFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active file to analyze');
        return;
    }
    await analyzeDocument(editor.document);
}
/**
 * Analyze all supported files in the workspace
 */
async function analyzeWorkspace() {
    const files = await vscode.workspace.findFiles('**/*.{py,js,ts}', '**/node_modules/**');
    const progress = await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: 'AI-Gauge: Analyzing workspace',
        cancellable: true
    }, async (progress, token) => {
        let analyzed = 0;
        for (const file of files) {
            if (token.isCancellationRequested)
                break;
            const doc = await vscode.workspace.openTextDocument(file);
            await analyzeDocument(doc);
            analyzed++;
            progress.report({
                message: `${analyzed}/${files.length} files`,
                increment: (1 / files.length) * 100
            });
        }
        return analyzed;
    });
    vscode.window.showInformationMessage(`AI-Gauge: Analyzed ${progress} files`);
}
/**
 * Toggle real-time analysis on/off
 */
function toggleRealTimeAnalysis() {
    const config = vscode.workspace.getConfiguration('aiGauge');
    const current = config.get('realTimeAnalysis');
    config.update('realTimeAnalysis', !current, vscode.ConfigurationTarget.Global);
    vscode.window.showInformationMessage(`AI-Gauge: Real-time analysis ${!current ? 'enabled' : 'disabled'}`);
}
/**
 * Analyze a document for LLM calls and provide recommendations
 */
async function analyzeDocument(document) {
    const supportedLanguages = ['python', 'javascript', 'typescript'];
    if (!supportedLanguages.includes(document.languageId)) {
        return;
    }
    const config = vscode.workspace.getConfiguration('aiGauge');
    if (!config.get('enabled'))
        return;
    try {
        // Step 1: Detect LLM calls in the code
        const llmCalls = detector.detectCalls(document);
        if (llmCalls.length === 0) {
            diagnosticsProvider.clearDiagnostics(document.uri);
            return;
        }
        // Step 2: For each detected call, analyze with AI-Gauge
        const analyses = await Promise.all(llmCalls.map(call => client.analyze(call)));
        // Step 3: Show diagnostics for OVERKILL verdicts
        const costThreshold = config.get('costThreshold') || 20;
        const recommendations = analyses.filter(a => a.verdict === 'OVERKILL' && a.costSavingsPercent >= costThreshold);
        diagnosticsProvider.updateDiagnostics(document.uri, recommendations);
        // Step 4: Update inline hints if enabled
        if (config.get('showInlineHints')) {
            inlineHintsProvider.updateHints(document.uri, analyses);
        }
    }
    catch (error) {
        console.error('AI-Gauge analysis failed:', error);
    }
}
function onDocumentChange(event) {
    analyzeDocument(event.document);
}
function debounce(func, wait) {
    let timeout;
    return ((...args) => {
        if (timeout)
            clearTimeout(timeout);
        timeout = setTimeout(() => func(...args), wait);
    });
}
function deactivate() {
    diagnosticsProvider?.dispose();
}
//# sourceMappingURL=extension.js.map