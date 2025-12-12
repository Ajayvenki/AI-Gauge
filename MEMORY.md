# AI-Gauge — Long-Term Memory

> **Purpose**: Preserve context across chat sessions so work can resume without re-explaining goals.

---

## 🎯 Project Vision

**AI-Gauge** is a developer-focused tool that measures the **carbon cost** and **token usage** of LLM API calls, then recommends more efficient alternatives.

### Core Formula

```
Carbon Cost = Tokens Used × Model Size × Energy per Token × Carbon Intensity
```

### High-Level Goals (Milestone Roadmap)

| # | Goal | Key Deliverables |
|---|------|------------------|
| 1 | `@track` decorator + context manager | Token counter (tiktoken), carbon formula, decorator, context manager, SQLite storage |
| 2 | `$ carbon stats` CLI | DB schema, stats aggregation, Rich terminal UI, JSON/CSV export |
| 3 | VS Code extension with inline hints | Extension scaffold, CodeLens provider, Python bridge, Dashboard WebView |
| 4 | Suggestion engine | Load Qwen2-0.5B / Ollama, system prompts, LangGraph integration, quality eval |
| 5 | Custom fine-tuned model | Collect training data, fine-tune Qwen2-0.5B, ONNX deployment, integration |

---

## 📍 Current Status (as of 12 Dec 2025)

**Stage 1 — Architecture Complete**

### Implemented Components

1. **`main.py`** — "Bad Example" Developer Code
   - Deliberately uses oversized `gpt-5.2` for trivial copy-editing
   - Showcases the problem: developers blindly use big models
   - Outputs metadata payload for decision module to intercept
   - Uses OpenAI Responses API with mock fallback

2. **`model_cards.py`** — Comprehensive Model Registry  
   - 30+ models from OpenAI, Anthropic, Google
   - Dataclass structure with: context_window, max_output_tokens, carbon_factor, best_for, strengths, weaknesses, latency_tier
   - Helper functions: `get_model_card()`, `list_models_by_provider()`, `get_cheapest_capable_model()`

3. **`decision_module.py`** — LangGraph 3-Agent Pipeline
   - **Agent 1 (Metadata Collector)**: Extracts model, tokens, classifies task complexity
   - **Agent 2 (Researcher)**: Queries model_cards, calculates carbon, finds alternatives
   - **Agent 3 (Reviewer)**: Validates recommendations, generates human-readable summary
   - Uses LangGraph `StateGraph` for orchestration
   - Outputs JSON recommendation + formatted summary

---

## 🗂️ Key Files

| File | Purpose |
|------|---------|
| `main.py` | Developer's "bad" code — uses GPT-5.2 for trivial task |
| `model_cards.py` | Registry of 30+ models with carbon factors |
| `decision_module.py` | LangGraph 3-agent recommendation engine |
| `requirements.txt` | openai, python-dotenv, langgraph |
| `MEMORY.md` | This file — long-term memory |

---

## 🔑 Technical Decisions

1. **Responses API**: Using `client.responses.create` with `max_output_tokens` (not `max_tokens`)
2. **LangGraph**: 3-agent linear pipeline (Metadata → Researcher → Reviewer)
3. **Carbon Factor**: Relative scale (0.2 for small models → 5.0 for flagship models)
4. **Complexity Classification**: trivial, simple, moderate, complex, expert
5. **Mock Fallback**: When API unavailable, deterministic mock response for testing

---

## 📝 Session Summary (12 Dec 2025)

- Restructured `main.py` as pure "bad example" (no `build_metadata()`)
- Created comprehensive `model_cards.py` with 30+ models
- Built LangGraph `decision_module.py` with 3 agents
- All scripts tested and working

---

*Update this file at major milestones or when resuming after lost context.*
