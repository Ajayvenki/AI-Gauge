# AI-Gauge 🌱

**LLM Cost & Carbon Optimizer** - Detects when you're overpaying for AI and suggests cheaper alternatives.

## What It Does

```python
# Before: Using $15/1M model for a typo fix ❌
response = client.chat.completions.create(
    model="gpt-5.2",
    messages=[{"role": "user", "content": "Fix typo: 'teh'"}]
)

# AI-Gauge says: 💡 OVERKILL! Use gpt-4o-mini instead
# → Saves 99% cost, reduces CO₂ by 94%
```

## Quick Start

```bash
# Setup
pip install -r requirements.txt

# Run tests (10 real-world scenarios)
python test_samples/test_model_comparison.py

# Start inference server (for VS Code plugin)
python inference_server.py
```

## VS Code Plugin

### Install from Source (Now)
```bash
cd ide_plugin
npm install
npm run compile
# Then in VS Code: "Developer: Install Extension from Location..."
```

### Install from Marketplace (Coming Soon)
```
ext install ai-gauge.ai-gauge
```

### How It Works
1. Plugin detects LLM API calls in your code
2. Sends to local inference server (http://localhost:8080)
3. Fine-tuned Phi-3.5 model analyzes the task
4. Shows inline hint if model is overkill

## Test Results (90% Accuracy)

| Case | Task | Model | Verdict | Status |
|------|------|-------|---------|--------|
| 1 | Fix typo | gpt-5.2 | OVERKILL | ✅ |
| 2 | Einstein's Riddle | gpt-5.2 | APPROPRIATE | ✅ |
| 3 | Code review | gpt-4o | APPROPRIATE | ✅ |
| 4 | Date format | claude-opus | OVERKILL | ✅ |
| 5 | Research agent | gpt-5.2 | APPROPRIATE | ✅ |
| 6 | Extract email | gpt-5.2 | OVERKILL | ✅ |
| 7 | Architecture design | gpt-5.2 | APPROPRIATE | ✅ |
| 8 | Translation | gpt-5.2 | OVERKILL | ✅ |
| 9 | Math proof | o3 | ⚠️ | ❌ |
| 10 | Format JSON | claude-opus | OVERKILL | ✅ |

## Architecture

```
Your Code → VS Code Plugin → Inference Server → Local Phi-3.5 → Recommendation
                                    ↓
                           3-Agent LangGraph Pipeline
                           1. Metadata Extractor
                           2. Task Analyzer  
                           3. Report Generator
```

## Files

```
AI-Gauge/
├── decision_module.py     # Core 3-agent pipeline
├── local_inference.py     # Phi-3.5 model wrapper
├── inference_server.py    # Flask API for plugin
├── model_cards.py         # Model database
├── ide_plugin/            # VS Code extension
│   ├── package.json
│   └── src/
└── test_samples/          # Test suite
```

## Model Tiers

| Tier | Models | Cost | CO₂ Factor |
|------|--------|------|------------|
| Budget | gpt-4o-mini, claude-haiku | $ | 0.3-1.0x |
| Standard | gpt-4o, claude-sonnet | $$ | 1.0-2.5x |
| Premium | gpt-4.1, o4-mini | $$$ | 3.0-5.0x |
| Frontier | gpt-5.2, o3, claude-opus | $$$$ | 8.0-12.0x |

## License

MIT
