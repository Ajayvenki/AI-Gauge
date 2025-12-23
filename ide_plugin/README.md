# AI-Gauge VS Code Extension

Analyzes LLM API calls in your code and suggests cheaper model alternatives.

## Setup (3 Steps)

### Step 1: Start the Inference Server

```bash
# In the AI-Gauge root directory
python inference_server.py

# Server runs at http://localhost:8080
```

### Step 2: Build the Extension

```bash
cd ide_plugin
npm install
npm run compile
```

### Step 3: Install in VS Code

**Option A: Install from folder (Development)**
1. Open VS Code
2. Press `Cmd+Shift+P` → "Developer: Install Extension from Location..."
3. Select the `ide_plugin` folder

**Option B: Package and install VSIX**
```bash
npm install -g @vscode/vsce
vsce package
# Then: "Extensions: Install from VSIX" → select ai-gauge-0.1.0.vsix
```

## Usage

1. Open a Python, JavaScript, or TypeScript file with LLM API calls
2. The extension automatically detects calls like:
   ```python
   client.chat.completions.create(model="gpt-5.2", ...)
   ```
3. If the model is overkill, you'll see an inline hint with alternatives

## Commands

- `AI-Gauge: Analyze Current File` - Analyze the active file
- `AI-Gauge: Analyze Workspace` - Analyze all supported files
- `AI-Gauge: Toggle Real-Time Analysis` - Enable/disable live analysis

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `aiGauge.enabled` | true | Enable/disable the extension |
| `aiGauge.realTimeAnalysis` | false | Analyze as you type |
| `aiGauge.showInlineHints` | true | Show inline cost hints |
| `aiGauge.costThreshold` | 20 | Min % savings to show hint |
| `aiGauge.modelServerUrl` | http://localhost:8080 | Inference server URL |

## Features

- 🔍 **Auto-Detection**: OpenAI, Anthropic, Google API patterns
- 💡 **Smart Analysis**: Fine-tuned Phi-3.5 model
- 💰 **Cost Savings**: Shows savings with alternatives
- 🌱 **Carbon Estimates**: CO₂ per call
- ⚡ **Latency Hints**: fast/medium/slow indicators

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VS Code Extension                         │
├─────────────────────────────────────────────────────────────┤
│  extension.ts          - Main entry, registers providers     │
│  llmCallDetector.ts    - Detects LLM calls via regex/AST     │
│  aiGaugeClient.ts      - Calls local Phi-3.5 inference       │
│  diagnosticsProvider.ts - Shows warnings + quick fixes       │
│  inlineHintsProvider.ts - Shows inline cost/latency hints    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Local Inference Server (port 8080)              │
│                  Fine-tuned Phi-3.5 Model                    │
│                                                              │
│  Endpoint: POST /analyze                                     │
│  Input: { model_used, provider, context, code_snippet }      │
│  Output: { verdict, confidence, recommendation, savings }    │
└─────────────────────────────────────────────────────────────┘
```

## Detection Patterns

The extension detects LLM calls using regex patterns:

**OpenAI (Python)**
```python
client.chat.completions.create(model="gpt-4o", ...)
```

**Anthropic (Python)**
```python
client.messages.create(model="claude-3-opus", ...)
```

**Google (Python)**
```python
genai.GenerativeModel("gemini-pro")
```

## User Experience

1. **Inline Hints** (always visible):
   ```python
   response = client.chat.completions.create(...)  ⚠️ $5.00/1k • slow → 💡 save 90%
   ```

2. **Diagnostics** (squiggly underline):
   - Yellow information squiggle on overkill model usage
   - Hover for detailed analysis

3. **Quick Fix** (lightbulb):
   - Click to replace model with recommended alternative

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `aiGauge.enabled` | `true` | Enable/disable extension |
| `aiGauge.realTimeAnalysis` | `false` | Analyze as you type |
| `aiGauge.showInlineHints` | `true` | Show inline hints |
| `aiGauge.costThreshold` | `20` | Min % savings to show |
| `aiGauge.modelServerUrl` | `http://localhost:8080` | Inference server URL |

## Development

```bash
# Install dependencies
npm install

# Compile
npm run compile

# Watch mode
npm run watch

# Package extension
vsce package
```

## Running the Inference Server

The extension requires a local inference server running the fine-tuned Phi-3.5 model:

```bash
# Using llama.cpp
./llama-server -m phi-3.5-ai-gauge.gguf -c 4096 --port 8080

# Or using mlx-lm (Apple Silicon)
mlx_lm.server --model ./mlx_output --port 8080
```

## Future Enhancements

1. **Real LLM Interception**: Hook into actual API calls at runtime
2. **Usage Tracking**: Track model usage patterns over time
3. **Team Analytics**: Aggregate insights across team
4. **Auto-Remediation**: Automatically downgrade models in non-prod
