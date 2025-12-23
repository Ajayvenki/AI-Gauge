# AI-Gauge VS Code Extension

Analyzes LLM API calls in your code and suggests cheaper model alternatives.

## 🚀 Quick Start (2 Minutes)

### 1. Install from VS Code Marketplace
```
Ctrl+Shift+X → Search "AI-Gauge" → Install → Reload VS Code
```

### 2. That's it! ✨
AI-Gauge automatically:
- ✅ Installs Ollama (if missing)
- ✅ Downloads the AI-Gauge analysis model
- ✅ Configures everything automatically

### 3. Start Coding
Get instant cost optimization hints as you write LLM API calls!

---

## 🎯 What It Does

AI-Gauge analyzes your code and provides real-time feedback on LLM model usage:

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

### 💰 Cost Optimization
- **Savings Alerts**: Shows potential cost reductions
- **Model Recommendations**: Suggests appropriate alternatives
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
| `aiGauge.realTimeAnalysis` | `false` | Analyze as you type |
| `aiGauge.showInlineHints` | `true` | Show inline cost hints |
| `aiGauge.costThreshold` | `20` | Min % savings to show hint |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VS Code Extension                         │
├─────────────────────────────────────────────────────────────┤
│  extension.ts          - Main entry, registers providers     │
│  llmCallDetector.ts    - Detects LLM calls via regex/AST     │
│  aiGaugeClient.ts      - Calls local Ollama inference        │
│  diagnosticsProvider.ts - Shows warnings + quick fixes       │
│  inlineHintsProvider.ts - Shows inline cost/latency hints    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Local Ollama Inference                         │
│                  Fine-tuned Phi-3.5 Model                   │
│                                                              │
│  Model: ajayvenki01/ai-gauge                                 │
│  Runs: Locally on user machine                               │
│  Privacy: 100% local processing                              │
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

### 1. Install Ollama
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull AI-Gauge Model
```bash
ollama pull ajayvenki01/ai-gauge
```

### 3. Verify Installation
```bash
ollama list  # Should show ai-gauge model
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

- **⚡ Fast**: Local inference, no network latency
- **🔒 Private**: All analysis happens locally
- **📱 Offline**: Works without internet after setup
- **🧠 Smart**: Fine-tuned Phi-3.5 model for accuracy
- **🌍 Green**: Helps reduce AI's carbon footprint

---

**Ready to optimize your AI costs? Install AI-Gauge today!** 🚀

[Install from VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=Ajayvenki2910.ai-gauge)
