# AI-Gauge 🌱

**Intelligent LLM Cost & Carbon Optimization**

AI-Gauge analyzes LLM API calls in your code and recommends more cost-effective and environmentally friendly model alternatives when appropriate.

## Key Features

- 🔍 **Overkill Detection**: Detects when frontier models are used for simple tasks
- 💰 **Cost Savings**: Suggests cheaper alternatives with comparable quality
- 🌱 **Carbon Awareness**: Uses carbon factor estimates based on academic research
- 🚀 **Local Inference**: Fine-tuned Phi-3.5 model runs entirely on your machine
- 🔌 **IDE Plugin**: VS Code extension with inline hints and quick fixes

## Test Results

```
✅ Pass Rate: 20/20 (100%)
├─ OVERKILL Detection:   10/10 passed
└─ APPROPRIATE Validation: 10/10 passed
```

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         3-Agent Pipeline                          │
├──────────────────────────────────────────────────────────────────┤
│  Agent 1: Metadata Extractor                                      │
│    └─ Parses model, tokens, tools, system prompt                  │
│                                                                    │
│  Agent 2: Analyzer (Fine-tuned Phi-3.5)                           │
│    └─ Assesses task complexity & minimum tier needed              │
│    └─ Heuristic adjustments for edge cases                        │
│                                                                    │
│  Agent 3: Report Generator                                        │
│    └─ Produces verdict: OVERKILL | APPROPRIATE | UNDERPOWERED     │
│    └─ Recommends alternatives with cost/carbon savings            │
└──────────────────────────────────────────────────────────────────┘
```

## Model Tier System

| Tier | Examples | Carbon Factor | Use Case |
|------|----------|---------------|----------|
| **Budget** | GPT-4o-mini, Claude Haiku | 0.3-1.0 | Simple tasks, trivial queries |
| **Standard** | GPT-4o, Claude Sonnet | 1.0-2.5 | Code review, moderate analysis |
| **Premium** | GPT-4.1, Claude 4 | 3.0-5.0 | Complex reasoning, research |
| **Frontier** | GPT-5.2, o3, Claude Opus | 8.0-12.0 | Agentic tasks, expert puzzles |

## Carbon Factor References

Based on peer-reviewed research:
- Patterson et al. (2021) "Carbon Emissions and Large Neural Network Training" [arXiv:2104.10350](https://arxiv.org/abs/2104.10350)
- Luccioni et al. (2024) "Power Hungry Processing" [arXiv:2311.16863](https://arxiv.org/abs/2311.16863)
- Google 2025 Environmental Report

## Quick Start

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run analysis on your code
python main.py --file your_code.py

# 3. Run tests
python test_samples/test_comprehensive.py
```

## IDE Plugin

See [ide_plugin/README.md](ide_plugin/README.md) for VS Code extension setup.

## Heuristic Adjustments

The fine-tuned model is augmented with 5 heuristic rules for edge cases:

1. **Tool Use** → at least standard tier
2. **Complex Tasks** → frontier tier
3. **Agentic Tasks** → frontier tier
4. **Extended Reasoning** → at least premium tier
5. **Expert Logic Puzzles** → frontier tier

Backups for testing without heuristics:
- `decision_module_original_no_heuristics.py.bak`
- `decision_module_with_heuristics.py.bak`

## License

MIT
