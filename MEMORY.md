# AI-Gauge — Project Memory & Context

> **Purpose**: This document is the "single source of truth" for the AI-Gauge project. If a conversation is lost, pass this file to a new assistant to restore full context. Update after each major milestone.

---

## 🎯 Project Vision

**AI-Gauge** (a.k.a. "Prompt Carbon Tracker") is a developer tool designed to be integrated as an **IDE plugin** that automatically intercepts LLM API calls **before** they execute to:
1. **Assess if the model choice is appropriate** for the actual task
2. **Estimate cost and carbon footprint** 
3. **Recommend alternatives ONLY when the model is overkill**

### The Core Problem
Developers often use oversized frontier models (like GPT-5.2) for trivial tasks because:
- They can't easily assess task complexity
- They default to "the best model" without considering efficiency
- There's no visibility into cost/carbon impact until after the fact

### The Solution
An IDE plugin intercepts the LLM API call, extracts all metadata **automatically** (no user input required), and runs it through a **3-agent LangGraph pipeline** that:
1. **Extracts pure metadata** from the intercepted call
2. **Uses GPT-5.2 to analyze** if the model choice is appropriate
3. **Only recommends alternatives if the model is overkill** (not forced recommendations)
4. **Explains WHY** the recommended model is sufficient for the task

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    IDE PLUGIN (Future: VS Code Extension)               │
│                                                                          │
│  Intercepts API calls like:                                             │
│    client.responses.create(model="gpt-5.2", input=[...])                │
│                                                                          │
│  Auto-extracts: model, prompt, system_prompt, tools, context            │
│  NO user input required - fully automatic                               │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    AI-GAUGE DECISION PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐   │
│   │   Agent 1        │──▶│   Agent 2        │──▶│   Agent 3        │   │
│   │   METADATA       │   │   ANALYZER       │   │   REPORTER       │   │
│   │   EXTRACTOR      │   │                  │   │                  │   │
│   │                  │   │                  │   │                  │   │
│   │ Pure extraction  │   │ Uses GPT-5.2 to  │   │ Generates        │   │
│   │ NO analysis      │   │ assess if model  │   │ human-readable   │   │
│   │ NO recommendations│  │ is appropriate   │   │ summary with     │   │
│   │                  │   │                  │   │ reasoning        │   │
│   │ Outputs:         │   │ Key decision:    │   │                  │   │
│   │ - token counts   │   │ is_appropriate?  │   │ Explains WHY     │   │
│   │ - has_tools      │   │                  │   │ alternative is   │   │
│   │ - has_context    │   │ Only finds alts  │   │ sufficient for   │   │
│   │                  │   │ if NOT appropriate│  │ this task        │   │
│   └──────────────────┘   └──────────────────┘   └──────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Model Registry (model_cards.py)                                        │
│  ├── OpenAI: GPT-5.2, GPT-5, GPT-4.1, GPT-4o, 4o-mini, etc.            │
│  ├── Anthropic: Claude Opus 4.5, Sonnet 4.5, Haiku 4.5 (official data) │
│  └── Google: Gemini 3 Pro, 2.5 Pro, 2.5 Flash, etc. (official data)    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ File Structure & Purpose

| File | Purpose |
|------|---------|
| `main.py` | **"Bad Example"** - Simulates a developer using GPT-5.2 for a trivial task. |
| `model_cards.py` | **Model Registry** - 25+ models with official pricing. Carbon factors: 0.0 = unknown. |
| `decision_module.py` | **LangGraph 3-Agent Pipeline** - Core recommendation engine. |
| `MEMORY.md` | **This file** - Long-term project memory. |
| `requirements.txt` | Dependencies: openai, python-dotenv, langgraph |

---

## 🔧 Technical Decisions

### 1. IDE Plugin Metadata Collection (Point a)
**Design**: The plugin intercepts LLM SDK calls at runtime:
- Monkey-patches `client.responses.create()`, `client.chat.completions.create()`
- Extracts model, prompt, system_prompt, tools, context from the actual call
- **NO manual metadata passing required** - fully automatic

### 2. Carbon Calculation (Point b)
**Methodology**: Based on CodeCarbon and "Power Hungry Processing" (Luccioni et al., 2024)

$$\text{CO}_2\text{eq (g)} = \text{Energy (kWh)} \times \text{Carbon Intensity (gCO}_2\text{/kWh)} \times \text{PUE}$$

Where:
- **Energy** = (GPU Power × Inference Time) / 3600 / 1000
- **Inference Time** = Total Tokens / Throughput (tokens/sec)
- **GPU Power**: Estimated by model tier (budget: 200W, frontier: 700W)
- **Carbon Intensity**: By provider (OpenAI: 200, Google: 150, default: 400 gCO₂/kWh)
- **PUE**: 1.2 (typical cloud datacenter)

### 3. Agent Responsibilities (Point c)
**Agent 1 (Metadata Extractor)**:
- Pure extraction only: token counts, has_tools, has_context
- NO analysis, NO recommendations, NO tier suggestions

**Agent 2 (Analyzer)**:
- Uses GPT-5.2 to analyze task complexity
- **Key output**: `is_model_appropriate` (boolean)
- Only searches for alternatives if model is NOT appropriate

**Agent 3 (Reporter)**:
- Generates human-readable summary
- Explains WHY the recommended model is sufficient

### 4. Conditional Recommendations (Point d)
**Previous**: Always suggested alternatives, sorted by savings
**Current**: 
- First validates if model is appropriate
- Only suggests alternatives if `is_model_appropriate = false`
- If appropriate, outputs "Model choice is appropriate. No changes recommended."

### 5. Recommendation Reasoning (Point e)
When recommending a switch, explains specifically:
- WHY the alternative model suits the task
- WHY the original model is overkill
- What capabilities the task actually requires

Example output:
```
💡 WHY THIS MODEL IS SUFFICIENT
   GPT-4o-mini can handle this task because it's designed for simple text 
   editing and summarization. GPT-5.2 is overkill since this is a trivial 
   single-sentence rewrite that doesn't require frontier reasoning capabilities.
```
**Current**: Tier-based selection filtered by requirements:
- `budget`: gpt-4o-mini, claude-haiku-4-5, gemini-2.0-flash-lite
- `standard`: gpt-4o, claude-sonnet-4-5, gemini-2.5-flash
- `premium`: gpt-5-mini, o4-mini, gemini-2.5-pro
- `frontier`: gpt-5.2, claude-opus-4-5, gemini-3-pro

### 3. Carbon Calculation (Research-Based)
**Formula**: CO2eq = Energy (kWh) × Carbon Intensity (gCO2/kWh) × PUE

**Energy Estimation**:
```
Energy (kWh) = (TDP_watts × inference_seconds) / 3600000
inference_seconds = output_tokens / tokens_per_second
```

**TDP Estimates by Model Tier**:
- Frontier models (gpt-5.2, claude-opus-4-5): 350W
- Standard models (gpt-4o, claude-sonnet-4-5): 250W
- Lightweight models (gpt-4o-mini, haiku): 150W

**Constants**:
- Carbon Intensity: 400 gCO2/kWh (global average from ElectricityMaps)
- PUE (Power Usage Effectiveness): 1.2 (typical datacenter overhead)
- Tokens per second: 50 (average inference speed)

**Sources**: CodeCarbon, ML CO2 Impact, "Power Hungry Processing" paper

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
2. Decision module rewritten with 3-agent LangGraph workflow
3. **Agent 1 (Metadata Extractor)**: Pure extraction - collects prompts, context, tools; analyzes task complexity WITHOUT making tier recommendations
4. **Agent 2 (Analyzer)**: Checks if original model is appropriate FIRST; only finds alternatives if NOT appropriate
5. **Agent 3 (Reporter)**: Generates explanation with WHY reasoning for recommendations
6. Carbon calculation enhanced with TDP-based energy estimation (research-based formula)
7. Conditional recommendations: suggestions only when model is inappropriate for task
8. IDE plugin interception design: metadata auto-collected, users don't send details

### 🔜 Future Work (Parked)
1. **IDE Integration**: VS Code extension to intercept LLM calls before execution
2. **SLM Replacement**: Replace GPT-5.2 analysis with fine-tuned local model (Qwen2-0.5B)
3. **CLI Tool**: `$ carbon stats` command with Rich terminal UI
4. **Real-time Grid Carbon**: Integration with ElectricityMaps for location-based carbon intensity

---

## 💡 Key Insights

1. **IDE plugin interception**: The plugin auto-collects metadata (prompts, context, tools) before LLM calls execute. Users don't manually send details.

2. **Conditional recommendations**: Suggestions only appear when the original model is inappropriate for the task. If gpt-5.2 is used for complex reasoning, no alternative is suggested.

3. **Appropriateness-first logic**: Agent 2 validates whether the original model fits the task complexity BEFORE searching for alternatives.

4. **WHY explanations**: When alternatives are suggested, the system explains why the lighter model is sufficient (e.g., "Simple paraphrase task doesn't require frontier reasoning").

5. **Carbon calculation is estimated**: Using TDP-based energy estimation since actual GPU power draw data isn't published by providers.

---

*Last updated: 12 December 2025*
