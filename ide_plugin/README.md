# AI-Gauge VS Code Extension

Analyzes LLM API calls in your code and suggests cheaper model alternatives using agent orchestration.

## 🚀 Quick Start (2 Minutes)

### 1. Install from VS Code Marketplace
```
Ctrl+Shift+X → Search "AI-Gauge" → Install → Reload VS Code
```

### 2. That's it! ✨
AI-Gauge automatically:
- ✅ Copies bundled Python code to your local storage
- ✅ Installs Python dependencies
- ✅ Sets up Ollama for agent analysis
- ✅ Starts the inference server
- ✅ Configures everything automatically

### 3. Start Coding
Get instant cost optimization hints as you write LLM API calls!

---

## 🎯 What It Does

AI-Gauge analyzes your code using a sophisticated agent pipeline and provides real-time feedback on LLM model usage:

```python
# Your code:
response = client.chat.completions.create(
    model="gpt-4",  # ⚠️ Overkill for simple tasks!
    messages=[...]
)

# AI-Gauge shows:
# 💡 Switch to GPT-3.5-turbo → Save 90% ($4.50 → $0.45 per 1K calls)
```

---

## ✨ Features

### 🔍 Smart Detection
- **Auto-Detection**: Finds OpenAI, Anthropic, Google, and custom API calls
- **Real-Time Analysis**: Analyzes as you type (optional)
- **Multi-Language**: Python, JavaScript, TypeScript support

### 🤖 Agent Orchestration
- **3-Agent Pipeline**: Metadata extraction, complexity analysis, and reporting
- **Local AI Integration**: Uses Ollama SLM within analyzer agent
- **Intelligent Recommendations**: Context-aware model suggestions

### 💰 Cost Optimization
- **Savings Alerts**: Shows potential cost reductions
- **Model Recommendations**: Suggests appropriate alternatives based on task complexity
- **Usage Tracking**: Monitors your API spending patterns

### 🌱 Environmental Impact
- **Carbon Tracking**: Estimates CO₂ footprint per API call
- **Green Suggestions**: Recommends efficient models
- **Sustainability Focus**: Helps reduce AI's environmental impact

### 🎨 User Experience
- **Inline Hints**: Cost and latency indicators in your code
- **Quick Fixes**: One-click model replacement
- **Hover Details**: Detailed analysis on demand

---

## 🛠️ Commands

- `AI-Gauge: Analyze Current File` - Analyze the active file
- `AI-Gauge: Analyze Workspace` - Analyze all supported files
- `AI-Gauge: Toggle Real-Time Analysis` - Enable/disable live analysis

---

## ⚙️ Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `aiGauge.enabled` | `true` | Enable/disable the extension |
| `aiGauge.showInlineHints` | `true` | Show inline cost hints |
| `aiGauge.costThreshold` | `20` | Min % savings to show hint |
| `aiGauge.modelServerUrl` | `http://localhost:8080` | Inference server URL |
| `aiGauge.serverAutoStart` | `true` | Automatically start inference server |
| `aiGauge.serverHealthCheckInterval` | `30` | Health check interval (seconds) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VS Code Extension                         │
├─────────────────────────────────────────────────────────────┤
│  extension.ts          - Main entry, server lifecycle mgmt   │
│  llmCallDetector.ts    - Detects LLM calls via regex/AST     │
│  aiGaugeClient.ts      - Communicates with inference server  │
│  diagnosticsProvider.ts - Shows warnings + quick fixes       │
│  inlineHintsProvider.ts - Shows inline cost/latency hints    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Inference Server (Flask)                    │
├─────────────────────────────────────────────────────────────┤
│  • REST API for extension communication                      │
│  • Agent orchestration via LangGraph                        │
│  • Health monitoring and automatic recovery                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Decision Module (LangGraph)                 │
├─────────────────────────────────────────────────────────────┤
│  Agent 1: Metadata Extractor - Analyzes call patterns       │
│  Agent 2: Analyzer (Ollama SLM) - Assesses task complexity   │
│  Agent 3: Reporter - Generates cost-saving recommendations  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Cards Database                           │
├─────────────────────────────────────────────────────────────┤
│  • Single source of truth for model metadata                │
│  • Tiers, costs, carbon factors, performance data           │
│  • Used by all agents for business logic                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 Detection Patterns

The extension detects LLM calls using intelligent patterns:

### OpenAI (Python)
```python
client.chat.completions.create(model="gpt-4o", ...)
client.beta.chat.completions.parse(model="gpt-4o-mini", ...)
```

### Anthropic (Python)
```python
client.messages.create(model="claude-3-opus", ...)
```

### Google (Python)
```python
model = genai.GenerativeModel("gemini-pro")
```

### OpenAI (JavaScript/TypeScript)
```javascript
const completion = await openai.chat.completions.create({
  model: "gpt-4",
  messages: [...]
});
```

---

## 💡 User Experience Examples

### Inline Hints (always visible):
```python
response = client.chat.completions.create(...)  # ⚠️ $5.00/1k • slow → 💡 save 90%
```

### Diagnostics (squiggly underline):
- Yellow information squiggle on overkill model usage
- Hover for detailed analysis with reasoning

### Quick Fix (lightbulb):
- Click the lightbulb to replace model with recommended alternative
- Automatic code transformation

---

## 🔧 Manual Setup (Advanced Users Only)

If auto-setup fails, you can manually configure:

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Ollama (for Agent Analysis)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull required model (handled automatically by server)
ollama pull llama3.2:3b
```

### 3. Start Inference Server
```bash
python src/inference_server.py
```

### 4. Verify Installation
```bash
# Check server health
curl http://localhost:8080/health

# Should return: {"status":"ok","agents":"ready","ollama":"connected"}
```

---

## 🛠️ Development

For extension developers:

### Prerequisites
- Node.js 16+
- VS Code 1.74+

### Setup
```bash
cd ide_plugin
npm install
npm run compile
```

### Development Commands
```bash
npm run watch      # Watch mode compilation
npm run compile    # One-time compilation
vsce package       # Create VSIX package
```

### Testing
- Open the extension in VS Code's Extension Development Host
- Test with files containing LLM API calls

---

## 🚀 Future Enhancements

- **Real API Interception**: Hook into actual API calls at runtime
- **Usage Analytics**: Track model usage patterns over time
- **Team Insights**: Aggregate cost savings across teams
- **Auto-Remediation**: Automatically optimize models in development
- **Multi-IDE Support**: Extend beyond VS Code

---

## 📊 Performance & Privacy

- **⚡ Smart**: Agent orchestration provides context-aware analysis
- **🔒 Private**: All analysis happens locally on your machine
- **📱 Offline**: Works without internet after initial setup
- **🧠 Intelligent**: Multi-agent pipeline with local AI integration
- **🌍 Green**: Helps reduce AI's carbon footprint through optimization
- **🔄 Automatic**: Server lifecycle managed by extension
- **💪 Reliable**: Health checks and automatic recovery

---

**Ready to optimize your AI costs with agent-powered analysis? Install AI-Gauge today!** 🚀

[Install from VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=Ajayvenki2910.ai-gauge)
