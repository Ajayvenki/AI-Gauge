# AI-Gauge — Project Memory & Context

> **Purpose**: This document is the "single source of truth" for the AI-Gauge project. If a conversation is lost, pass this file to a new assistant to restore full context. Update after each major milestone.

---

## 🎯 Project Vision

**AI-Gauge** (a.k.a. "Prompt Carbon Tracker") is a developer tool that analyzes LLM API calls **before** they are executed to:
1. **Estimate cost** (USD) of the planned request
2. **Estimate carbon footprint** (when data is available)
3. **Recommend more efficient model alternatives** that can accomplish the task at lower cost/carbon

### The Core Problem
Developers often use oversized frontier models (like GPT-5.2) for trivial tasks because:
- They can't easily assess task complexity
- They default to "the best model" without considering efficiency
- There's no visibility into cost/carbon impact until after the fact

### The Solution
AI-Gauge intercepts LLM request metadata and runs it through a **3-agent LangGraph pipeline** that:
1. Uses GPT-5.2 to **intelligently analyze** the task (not keyword matching)
2. Finds appropriate alternative models based on **actual requirements**
3. Provides actionable recommendations with cost/carbon savings

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AI-GAUGE SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Developer's Code (main.py)                                             │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  model = "gpt-5.2"                                            │      │
│  │  system_prompt = "You are an elite..."                        │      │
│  │  user_prompt = "Rewrite this tooltip..."                      │      │
│  │  context = {...}                                              │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                              │                                           │
│                              ▼                                           │
│  Decision Module (decision_module.py) — LangGraph Pipeline              │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                                                                │      │
│  │   ┌────────────────┐   ┌────────────────┐   ┌──────────────┐ │      │
│  │   │   Agent 1      │──▶│   Agent 2      │──▶│   Agent 3    │ │      │
│  │   │   Metadata     │   │   Researcher   │   │   Reviewer   │ │      │
│  │   │   Collector    │   │                │   │              │ │      │
│  │   │                │   │                │   │              │ │      │
│  │   │ Uses GPT-5.2   │   │ Queries model  │   │ Validates &  │ │      │
│  │   │ to analyze:    │   │ registry,      │   │ formats      │ │      │
│  │   │ - Task intent  │   │ calculates     │   │ recommendation│ │      │
│  │   │ - Complexity   │   │ costs, finds   │   │ for human    │ │      │
│  │   │ - Requirements │   │ alternatives   │   │ consumption  │ │      │
│  │   └────────────────┘   └────────────────┘   └──────────────┘ │      │
│  │                                                                │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                              │                                           │
│                              ▼                                           │
│  Model Registry (model_cards.py)                                        │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  25+ models from OpenAI, Anthropic, Google                    │      │
│  │  Official data only: pricing, context window, capabilities    │      │
│  │  Carbon factors: 0.0 = unknown (no guessing)                  │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ File Structure & Purpose

| File | Purpose |
|------|---------|
| `main.py` | **"Bad Example"** - Simulates a developer using GPT-5.2 for a trivial copy-editing task. Demonstrates the problem AI-Gauge solves. |
| `model_cards.py` | **Model Registry** - Dataclass-based registry of 25+ models with official pricing, context windows, capabilities. Sources cited in notes. Unknown values marked explicitly. |
| `decision_module.py` | **LangGraph 3-Agent Pipeline** - The core recommendation engine. Uses GPT-5.2 for intelligent task analysis. |
| `MEMORY.md` | **This file** - Long-term project memory for context restoration. |
| `requirements.txt` | Dependencies: openai, python-dotenv, langgraph |

---

## 🔧 Technical Decisions

### 1. Task Analysis Approach
**Previous**: Keyword-based classification (e.g., "if 'code' in prompt → complex")
**Current**: GPT-5.2 analyzes the actual task and extracts structured metadata:
- `task_intention`: What is the task trying to accomplish?
- `task_category`: text_generation, code_generation, analysis, etc.
- `actual_complexity`: trivial/simple/moderate/complex/expert (with reasoning)
- `requires_vision/audio/reasoning/long_context`: Boolean flags
- `accuracy_requirement`: low/medium/high/critical
- `model_appropriate`: Is the chosen model overkill/appropriate/underpowered?
- `recommended_tier`: budget/standard/premium/frontier

### 2. Model Recommendations
**Previous**: Label matching (complexity → hardcoded model list)
**Current**: Tier-based selection filtered by requirements:
- `budget`: gpt-4o-mini, claude-haiku-4-5, gemini-2.0-flash-lite
- `standard`: gpt-4o, claude-sonnet-4-5, gemini-2.5-flash
- `premium`: gpt-5-mini, o4-mini, gemini-2.5-pro
- `frontier`: gpt-5.2, claude-opus-4-5, gemini-3-pro

### 3. Carbon Factors
**Constraint**: Only use official/verified data. Unknown values = `0.0`.
When carbon_factor is 0 or unknown:
- `calculate_carbon_cost()` returns `None`
- Carbon savings displayed as "unknown" instead of fabricated numbers

### 4. API Usage
- OpenAI Responses API: `client.responses.create(model=..., input=..., max_output_tokens=...)`
- Analysis model: GPT-5.2 (user requirement; later to be replaced with SLM)

### 5. Model Card Data Sources
All model card fields populated from official documentation only:
- OpenAI: https://platform.openai.com/docs/models/
- Anthropic: https://platform.claude.com/docs/en/about-claude/models/
- Google: https://ai.google.dev/gemini-api/docs/pricing

---

## 📊 Current Model Registry (Summary)

### OpenAI (14 models)
| Model | Context | Max Output | Price (in/out) |
|-------|---------|------------|----------------|
| gpt-5.2 | 400K | 128K | $1.75/$14 |
| gpt-5.2-pro | 400K | 128K | $21/$168 |
| gpt-5 | 400K | 128K | $1.25/$10 |
| gpt-5-mini | 400K | 128K | $0.25/$2 |
| gpt-5-nano | 400K | 128K | $0.05/$0.40 |
| gpt-4.1 | 1M+ | 32K | $2/$8 |
| gpt-4.1-mini | 1M+ | 32K | $0.40/$1.60 |
| gpt-4.1-nano | 1M+ | 32K | $0.10/$0.40 |
| gpt-4o | 128K | 16K | $2.50/$10 |
| gpt-4o-mini | 128K | 16K | $0.15/$0.60 |
| o3 | 200K | 100K | $2/$8 |
| o4-mini | 200K | 100K | $1.10/$4.40 |

### Anthropic (5 models)
| Model | Context | Max Output | Price (in/out) |
|-------|---------|------------|----------------|
| claude-opus-4-5 | 200K | 64K | $5/$25 |
| claude-sonnet-4-5 | 200K | 64K | $3/$15 |
| claude-haiku-4-5 | 200K | 64K | $1/$5 |
| claude-3-5-sonnet | 200K | 8K | $3/$15 |
| claude-3-5-haiku | 200K | 8K | $0.80/$4 |

### Google (6 models)
| Model | Context | Max Output | Price (in/out) |
|-------|---------|------------|----------------|
| gemini-3-pro-preview | 1M | 64K | $2/$12 |
| gemini-2.5-pro | 1M | 65K | $1.25/$10 |
| gemini-2.5-flash | 1M | 65K | $0.30/$2.50 |
| gemini-2.5-flash-lite | 1M | 65K | $0.10/$0.40 |
| gemini-2.0-flash | 1M | 8K | $0.10/$0.40 |
| gemini-2.0-flash-lite | 1M | 8K | $0.075/$0.30 |

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="sk-..."

# Run the "bad example" (main.py)
python main.py

# Run the decision module demo
python decision_module.py

# Test the model registry
python model_cards.py
```

---

## 📍 Current Status (12 Dec 2025)

### ✅ Completed
1. Model cards updated with official sources (OpenAI, Anthropic, Google)
2. Decision module rewritten to use GPT-5.2 for task analysis (not keywords)
3. Tier-based model recommendations (not label matching)
4. Carbon handling updated to show "unknown" instead of fabricated values
5. All syntax verified and compiling

### 🔜 Future Work (Parked)
1. **IDE Integration**: How to intercept LLM calls before they execute (VS Code extension, proxy, etc.)
2. **SLM Replacement**: Replace GPT-5.2 analysis with fine-tuned local model (Qwen2-0.5B)
3. **CLI Tool**: `$ carbon stats` command with Rich terminal UI
4. **Actual Carbon Data**: Research and populate real carbon factors from published studies

---

## 💡 Key Insights

1. **The "bad example" pattern**: Developer code often looks complex (enterprise prompts, context objects) but the actual task is trivial (rewrite one sentence).

2. **GPT-5.2 can detect this**: When asked to analyze the task, it correctly identifies that the actual complexity is low even when the prompt dressing is elaborate.

3. **Cost savings are primary**: Since carbon data is largely unknown, the immediate value is in cost savings. Carbon tracking becomes meaningful once verified data is available.

4. **Pre-execution is the goal**: The ultimate goal is to analyze requests *before* they execute, not after. This requires IDE/proxy integration (future phase).

---

*Last updated: 12 December 2025*
