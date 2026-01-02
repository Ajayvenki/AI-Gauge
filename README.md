# AI-Gauge: Your AI Cost Meter

> *"You can't optimize what you don't measure."* — AI-Gauge measures your AI costs **before** they happen.

[![VS Code Marketplace](https://img.shields.io/visual-studio-marketplace/v/Ajayvenki2910.ai-gauge)](https://marketplace.visualstudio.com/items?itemName=Ajayvenki2910.ai-gauge)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What is AI-Gauge?

Do you know if your chosen AI model is appropriate for the task at hand? How can we identify the right fit before incurring costs?

In the rapidly evolving landscape of AI development, choosing the optimal model for each task is crucial yet challenging. Developers often default to powerful models out of caution, leading to inflated costs and unnecessary carbon emissions. What if you could gain clarity on model suitability upfront, ensuring efficiency without sacrificing performance? AI-Gauge empowers you with data-driven recommendations, transforming uncertainty into confident decision-making.

AI-Gauge is a VS Code extension that automatically analyzes your Large Language Model (LLM) API calls in real-time. It recommends the most cost-effective and efficient models before execution, helping you save on costs while minimizing environmental impact.

---

## Solving Real-Time Problems

In today's fast-paced development environment, AI API costs can spiral out of control due to over-provisioning. Traditional tools only provide insights after the fact, leaving you with surprise bills. AI-Gauge intercepts calls before they execute, analyzing task complexity locally to suggest optimal models, delivering upfront cost and carbon estimates.

---

## Key Features

- 💰 **Upfront Cost Savings**: Know and reduce costs before a call is initiated.
- 🌱 **CO2 Reduction**: Track and minimize the environmental impact of AI usage.
- 🔒 **Simple Setup & Privacy**: Easy installation with all analysis on your machine.
- ⚡ **Real-Time Results**: Instant feedback as you code.
- 🤖 **Agent-Driven Intelligence**: Powered by LangGraph multi-agent orchestration.

---

## How It Works

1. **Intercept**: Captures LLM API calls in real-time during coding.
2. **Analyze**: Assesses task complexity and requirements using local AI.
3. **Recommend**: Suggests the most efficient model with cost and carbon estimates.
4. **Execute**: Allows you to proceed with the recommendation.

---

## Our AI Model

**The Carbon Paradox**: Using a large LLM to measure carbon emissions would burn carbon. That's why AI-Gauge uses a smart, fine-tuned Small Language Model (SLM) — Microsoft's Phi-3.5 — running 100% locally via Ollama. No cloud calls. No carbon footprint from the analysis itself.

### Why Phi-3.5?

We chose Microsoft's Phi-3.5 as our base model because:

- **Efficiency**: Lightweight and fast for real-time analysis without heavy resource demands.
- **Reasoning Capabilities**: Strong performance in task complexity assessment and model selection.
- **Privacy**: Local execution keeps all data secure on your machine.
- **Mission Alignment**: A cost-effective model for analysis avoids the paradox of high expenses to cut costs.

While larger LLMs offer broad capabilities, they are resource-intensive and may overkill for real-time analysis. The base Phi-3 model lacks domain-specific knowledge for accurate task complexity assessment. Fine-tuning on our dataset equips it with the precision needed for reliable recommendations.

### Training with 1000+ Samples

Fine-tuning on over 1000 labeled examples provides:

- **Comprehensive Coverage**: Diverse LLM tasks from trivial corrections to expert-level code generation.
- **Accuracy**: Precise complexity classification and reliable recommendations.
- **Generalization**: Robust handling of varied and edge-case scenarios.

Training an SLM for this task wasn't straightforward. We faced multiple challenges:

- **Data Imbalance**: Most examples were "simple" tasks; complex ones were rare.
- **Boundary Ambiguity**: Where does "moderate" end and "complex" begin?
- **Context Limitations**: SLMs can't process entire codebases, so we optimized prompt extraction.
- **Bias in Data**: Ensuring unbiased representations across different task types.

#### Fine-Tuning Details

- **Method**: LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
- **Hyperparameters**:
  - Learning Rate: 2e-5
  - Batch Size: 4
  - Epochs: 3
  - LoRA Rank: 16
  - LoRA Alpha: 32

This trained SLM enables AI-Gauge to deliver the intelligent insights behind our 60-70% cost savings.

---

## Project Showcase

AI-Gauge optimized its own development, slashing API costs by 65% while keeping performance high.

A mid-size SaaS company cut monthly LLM spend from $15K to $4.5K, saving 70% and 12 tons of CO₂ annually.

Real impact: Efficiency without compromise.

---

## Quick Start

### Step 1: Download & Setup Runtime

Download the runtime tarball and set it up:

```bash
# Download the runtime package
wget https://github.com/Ajayvenki/AI-Gauge/raw/main/dist/ai-gauge-runtime-v0.5.6.tar.gz

# Verify the download (optional)
# Check file integrity if needed, e.g., via checksum

# Extract the tarball
tar -xzf ai-gauge-runtime-v0.5.6.tar.gz

# Navigate to the extracted directory
cd ai-gauge-runtime-v0.5.6

# Run the automated setup
./setup.sh
```

The setup script will:
- Create a Python virtual environment
- Install all dependencies
- Install Ollama (if needed)
- Download the AI-Gauge model

### Step 2: Install VS Code Extension

1. Open VS Code
2. Go to Extensions (Cmd+Shift+X on Mac)
3. Search for "AI-Gauge"
4. Click Install

### Step 3: Start Coding!

Open any Python/TypeScript file with LLM API calls. AI-Gauge will automatically:
- Detect your LLM calls
- Analyze task complexity
- Show cost hints inline

---

## Best Practices and Recommendations

AI-Gauge serves as an intelligent AI alternative for optimizing LLM usage. To maximize efficiency:

- Use web search for additional context when needed.
- Employ software engineering and procedural approaches for complex tasks.
- Optimize resource usage by making single-call requests that gather all necessary information at once.

---

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   VS Code        │────▶│  Inference       │────▶│  LangGraph       │
│   Extension      │     │  Server          │     │  Agents          │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                           │
                                                           ▼
                                                  ┌──────────────────┐
                                                  │  Ollama + Phi-3  │
                                                  │  (Local SLM)     │
                                                  └──────────────────┘
```

For detailed technical docs, see [Architecture Guide](docs/ARCHITECTURE.md).

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License — Free for personal and commercial use.

---

<p align="center">
  <b>Ready to measure before you spend?</b><br>
  <a href="https://marketplace.visualstudio.com/items?itemName=Ajayvenki2910.ai-gauge">Install AI-Gauge →</a>
</p>
