# AI-GAUGE Project Memory

> **Purpose**: This file contains the complete project context. If chat history is lost, pass this file to an LLM to restore full understanding of the project state, goals, and next steps.

---

## 📌 Latest Update (January 2, 2026 - v0.4.3 Release)

### Repository-Based Extension Approach ✅ COMPLETE

**Problem Solved**: Fixed critical "fetch failed" errors in marketplace extension caused by bundled Python bytecode incompatibility.

#### What Changed: From Bundled to Repository-Based
- **BEFORE**: Extension bundled Python files → Marketplace rejection due to bytecode issues
- **AFTER**: Extension detects local AI-Gauge installation → Clean, reliable operation

#### New Architecture
1. **Runtime Package**: Minimal 33KB `ai-gauge-runtime-v0.4.3.tar.gz` with essential files only
2. **Repository Detection**: Extension automatically finds local AI-Gauge installation
3. **Clean Installation**: Users download runtime package instead of cloning full repo

#### User Installation Flow (v0.4.3)
```bash
# 1. Download runtime package
wget https://github.com/ajayvenki2910/ai-gauge/releases/download/v0.4.3/ai-gauge-runtime-v0.4.3.tar.gz

# 2. Extract and setup
tar -xzf ai-gauge-runtime-v0.4.3.tar.gz
cd ai-gauge-runtime-v0.4.3
./setup.sh  # Installs Ollama, downloads model, sets up dependencies

# 3. Install VS Code extension
# Search "AI-Gauge" in marketplace
```

#### Files Created/Modified
- **`runtime/`**: New directory with flattened Python modules (no src/ subdirectory)
- **`ai-gauge-runtime-v0.4.3.tar.gz`**: 33KB compressed runtime package
- **`ide_plugin/src/extension.ts`**: Modified to detect local repo paths instead of bundled files
- **`runtime/inference_server.py`**: Fixed imports for standalone operation
- **`runtime/decision_module.py`**: Fixed relative imports in flattened structure
- **`README.md`**: Updated with new installation instructions
- **`CHANGELOG.md`**: Added v0.4.3 release notes

#### Technical Fixes
- **Import Issues**: Changed `from src.module` to `from module` in runtime package
- **Server Startup**: Fixed module resolution in standalone environment
- **Extension Size**: Reduced from ~2.2MB to 33KB (no bundled Python files)
- **Compatibility**: Works across different Python environments and OS versions

#### Backend Priority (Unchanged)
1. **HuggingFace** (cloud) - Default, requires API key
2. **Ollama** (local) - Offline, requires 2.2GB download  
3. **llama_cpp** (local) - Advanced users

#### Key Benefits
- ✅ **No More Fetch Failures**: Repository-based approach eliminates bundling issues
- ✅ **Clean User Experience**: 33KB download vs full repo clone
- ✅ **Zero Workspace Clutter**: No __pycache__, tests, or dev files in user workspace
- ✅ **Professional Distribution**: Feels like real software, not a development project
- ✅ **Runtime Package Detection**: Extension automatically finds runtime packages in any location
- ✅ **Reliable Operation**: No Python bytecode compatibility issues

#### Extension Fixes (Jan 2, 2026)
- **Detection Logic**: Updated `isValidRepo()` to check runtime package structure (`inference_server.py` in root)
- **Server Startup**: Modified `startInferenceServer()` to use correct path for runtime packages
- **Backward Compatibility**: Supports both runtime package and development repository structures
- **Error Resolution**: Fixed "fetch failed" errors when extension couldn't locate local installation
- **Package Optimization**: Reduced extension size from 33KB to 21KB by excluding development files (.d.ts, .js.map)
- **Clean Packaging**: Added .vscodeignore to only include essential runtime files

#### Next Steps
- Upload `ai-gauge-runtime-v0.4.3.tar.gz` to GitHub releases
- Publish extension v0.4.3 to VS Code marketplace
- Test end-to-end installation on clean machine
- Monitor for any remaining issues

---

## 📌 Previous Update (Dec 23, 2025 - Night)

### HuggingFace Cloud Inference ✅ COMPLETE

**What changed**: Added HuggingFace Inference API as the PRIMARY backend. Users now just need an HF API key - no local model downloads required.

#### User Flow (Simplified)
1. Get free HuggingFace API key at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Install VS Code extension
3. Set API key in settings: `aiGauge.huggingFaceApiKey: "hf_..."`
4. Run `python demo_single_test.py` - DONE!

#### Files Updated
- **`local_inference.py`**:
  - Added `HF_API_KEY`, `HF_MODEL_ID`, `HF_INFERENCE_URL` config vars
  - Added `is_huggingface_available()` function
  - Added `get_huggingface_response()` function with retry on model loading
  - `analyze_with_local_model()` priority: HuggingFace → Ollama → llama_cpp
  - Default backend changed from `ollama` to `huggingface`

- **`ide_plugin/package.json`**:
  - New settings: `useHuggingFace`, `huggingFaceApiKey`, `huggingFaceModel`
  - `useHuggingFace` defaults to `true`

- **`ide_plugin/src/aiGaugeClient.ts`**:
  - Added `analyzeWithHuggingFace()` method
  - `ClientConfig` interface extended with HF fields
  - Priority order: HuggingFace → Ollama → Inference Server

- **`ide_plugin/src/extension.ts`**:
  - `getClientConfig()` now reads HuggingFace settings

- **`INSTALL.md`**: Complete rewrite with HuggingFace as primary option

#### Three Backends (Priority Order)
1. **HuggingFace** (cloud) - Simplest, just API key, default
2. **Ollama** (local) - Offline, requires 2.2GB download
3. **llama_cpp** (local) - For advanced users

#### Environment Variables
```bash
# Cloud (HuggingFace)
export HF_API_KEY="hf_your_token"
export HF_MODEL_ID="ajayvenkatesan/ai-gauge"  # Your HF repo

# Local (Ollama)
export AI_GAUGE_BACKEND="ollama"
export OLLAMA_URL="http://localhost:11434"
```

---

## 📌 Previous Update (Dec 23, 2025 - Evening)

### Ollama Integration ✅ COMPLETE

**What changed**: Full Ollama backend support added. Users can now run AI-Gauge without llama-cpp-python by using Ollama.

#### New Files Created
- **`Modelfile`**: Ollama model definition for importing the GGUF model
- **`INSTALL.md`**: Comprehensive installation guide for end users
- **`setup.sh`**: Automated setup script

#### Updated Files
- **`local_inference.py`**: 
  - Added `is_ollama_available()` function
  - Added `get_ollama_response()` function  
  - `analyze_with_local_model()` now tries Ollama first, falls back to llama_cpp
  - New config: `INFERENCE_BACKEND`, `OLLAMA_URL`, `OLLAMA_MODEL`

- **`inference_server.py`**:
  - Health endpoint now shows backend status (Ollama vs llama_cpp)
  - Startup message shows active backend

- **`ide_plugin/package.json`**:
  - New settings: `useOllamaDirect`, `ollamaUrl`, `ollamaModel`
  
- **`ide_plugin/src/aiGaugeClient.ts`**:
  - Supports direct Ollama connection (bypasses inference server)
  - New `analyzeWithOllama()` method
  - `ClientConfig` interface for flexible configuration

- **`ide_plugin/src/extension.ts`**:
  - Watches for config changes and updates client
  - `getClientConfig()` helper function

- **`README.md`**: Complete rewrite with compelling narrative

- **`demo_single_test.py`**: Cleaned up, no print statements

#### How Ollama Setup Works
```bash
# 1. Create model from GGUF
ollama create ai-gauge -f Modelfile

# 2. Run server (or start inference_server.py)
ollama serve

# 3. Test
python test_samples/demo_single_test.py
```

#### Two Ways to Use
1. **Inference Server** (recommended): `python inference_server.py`
2. **Direct Ollama**: Set `aiGauge.useOllamaDirect: true` in VS Code

---

## 📌 Previous Update (Dec 23, 2025 - Morning)

### Recommendation Report Format Overhaul ✅ COMPLETE

**What changed**: Moved enhanced report formatting logic from `test_model_comparison.py` into core `decision_module.py` so it's used everywhere.

**New `format_recommendation_report()` function in `decision_module.py`** now generates beautifully formatted reports with:

1. **📋 METADATA Section** (All intercepted call data):
   - Model requested, provider, current tier
   - Estimated tokens (input/output/total)
   - Cost per 1M tokens
   - Carbon factor and CO₂ calculation with formula shown: `(tokens × carbon_factor × 0.0001)`

2. **🎯 ANALYSIS Section**:
   - Task summary, category, complexity level
   - Minimum required tier for task

3. **✅ VERDICT with Green Emoji** (when OVERKILL):
   - Shows: `✅ VERDICT: OVERKILL — Cost savings available!`

4. **📦 ALL MODELS IN RECOMMENDED TIER** (not just top 3):
   - Lists **every stable model** in the recommended tier
   - For each shows: display name, provider, cost/1M, savings %, CO₂ savings %, actual CO₂ value

5. **📝 REASONING Section**:
   - Task complexity analysis
   - Current vs minimum tier comparison
   - Why current choice is wasteful
   - Model reasoning from Phi-3.5 analysis

6. **🌍 CARBON IMPACT Section**:
   - Current model CO₂ per call
   - Average CO₂ for recommended tier
   - Potential CO₂ reduction percentage

**Example Output** (from `demo_single_test.py` on "fix typo" with GPT-5.2):
```
✅ VERDICT: OVERKILL — Cost savings available!

📋 METADATA
   Model: GPT-5.2 | Tier: FRONTIER
   Tokens: 20 in / 15 out (≈35 total)
   Cost: $0.000245 | CO₂: 0.0327g (35 × 8.0 × 0.0001)

📦 ALL BUDGET TIER MODELS:
   ├─ GPT-5 Nano: $0.05/$0.40 | Saves 97% cost, 95% CO₂
   ├─ GPT-4o Mini: $0.15/$0.60 | Saves 95% cost, 95% CO₂
   ├─ Gemini 2.0 Flash-Lite: $0.07/$0.30 | Saves 98% cost, 97% CO₂
   └─ [8 total budget models shown]

📝 REASONING
   • Task Complexity: SIMPLE
   • Current: FRONTIER tier vs Required: BUDGET tier
   • Model reasoning: "The task is straightforward and does not require advanced language understanding"

🌍 CARBON IMPACT
   • Current: 0.0327g CO₂ per call
   • Avg Budget tier: 0.0018g CO₂ per call
   • Potential reduction: 95%
```

### Files Updated
- `decision_module.py`: Added `format_recommendation_report()` function and enhanced `reporter_agent()`
- `decision_module.py`: Added `get_all_models_in_tier()` to show all models, not just top 3
- Created `demo_single_test.py`: Single test case demo showing the full formatted output
- `test_model_comparison.py`: Simplified to just call `analyze_llm_call()` and display results

### Result
✅ All recommendation reports now show complete metadata, detailed reasoning, and all models in the tier - not just test file, but core API output.

---

## 🎯 Project Overview

**AI-Gauge** is a smart LLM cost optimizer that analyzes code before API calls are made and recommends cheaper model alternatives when the task doesn't require an expensive model.

### Core Value Proposition
- Developers unconsciously use expensive models (GPT-5.2, Claude Opus) for simple tasks
- AI-Gauge intercepts these calls and says: *"This is a typo fix. You're using a $15/1M token model. Use gpt-4o-mini ($0.15/1M) instead - saves 99% cost and reduces CO₂"*

### How It Works
```
User Code → IDE Plugin Intercepts LLM Call → Local Phi-3.5 Model Analyzes → Recommendation
```

---

## 🏗️ Architecture

### 3-Agent LangGraph Pipeline (`decision_module.py`)
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Agent 1       │────▶│   Agent 2       │────▶│   Agent 3       │
│   Metadata      │     │   Analyzer      │     │   Reporter      │
│   Extractor     │     │   (Local Model) │     │   (Summary)     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **Metadata Extractor**: Parses model, prompt, tokens, tools from intercepted call
2. **Analyzer**: Uses local Phi-3.5 to classify task complexity → determines if model is OVERKILL or APPROPRIATE
3. **Reporter**: Generates recommendation with alternatives, cost savings, CO₂ estimates

### Local Model
- **Model**: Phi-3.5 fine-tuned on 1,289 labeled examples
- **Format**: GGUF Q4_K_M (2.2GB, optimized for Mac Metal)
- **Location**: `training_data/models/ai-gauge-q4_k_m.gguf`
- **Inference**: `llama-cpp-python` (no API keys needed)

### Model Database (`model_cards.py`)
- Comprehensive data for OpenAI, Anthropic, Google models
- Includes: pricing, carbon factors, capabilities, tier classifications
- Tiers: `budget` → `standard` → `premium` → `frontier`

---

## 📂 Project Structure

```
AI-Gauge/
├── decision_module.py     # Core 3-agent LangGraph pipeline
├── local_inference.py     # Phi-3.5 local model wrapper
├── model_cards.py         # Model database (pricing, carbon, tiers)
├── main.py                # Demo script
├── requirements.txt       # Python dependencies
│
├── ide_plugin/            # VS Code extension
│   ├── package.json       # Extension manifest
│   ├── src/
│   │   ├── extension.ts        # Main activation
│   │   ├── llmCallDetector.ts  # Detect API calls in code
│   │   ├── aiGaugeClient.ts    # Communicate with local server
│   │   ├── diagnosticsProvider.ts
│   │   └── inlineHintsProvider.ts
│   └── out/               # Compiled JS
│
├── test_samples/          # Test suite
│   ├── test_model_comparison.py  # 10 test cases
│   └── test_results.json
│
└── training_data/         # Model training
    ├── dataset_train.json # 1,289 labeled examples
    ├── train_model.py     # Training script
    └── models/            # GGUF models
```

---

## 🚀 Stages & Progress

### Stage 1: Core Engine ✅ COMPLETE
- [x] 3-agent LangGraph pipeline
- [x] Model cards database (OpenAI, Anthropic, Google)
- [x] Carbon factor calculations
- [x] Cost savings estimation

### Stage 2: Local Model ✅ COMPLETE
- [x] Generated 1,289 training examples
- [x] Fine-tuned Phi-3.5 on task classification
- [x] Converted to GGUF Q4_K_M format
- [x] Local inference with llama-cpp-python
- [x] 90% accuracy on 10-case test suite

### Stage 3: IDE Plugin 🔄 IN PROGRESS
- [x] VS Code extension structure created
- [x] LLM call detector (Python, JS, TS)
- [x] Diagnostics and inline hints providers
- [ ] **TODO**: Set up local inference server
- [ ] **TODO**: Package for VS Code Marketplace
- [ ] **TODO**: Publish extension

### Stage 4: Distribution 📋 PLANNED
- [ ] Publish to VS Code Marketplace
- [ ] Host model on HuggingFace or Ollama
- [ ] One-click install experience

---

## 📊 Test Results (Latest)

| # | Test Case | Model | Expected | Actual | Status |
|---|-----------|-------|----------|--------|--------|
| 1 | Fix typo | gpt-5.2 | OVERKILL | OVERKILL | ✅ |
| 2 | Einstein's Riddle | gpt-5.2 | APPROPRIATE | APPROPRIATE | ✅ |
| 3 | Code review | gpt-4o | APPROPRIATE | APPROPRIATE | ✅ |
| 4 | Date format | claude-opus | OVERKILL | OVERKILL | ✅ |
| 5 | Research agent | gpt-5.2 | APPROPRIATE | APPROPRIATE | ✅ |
| 6 | Extract email | gpt-5.2 | OVERKILL | OVERKILL | ✅ |
| 7 | System architecture | gpt-5.2 | APPROPRIATE | APPROPRIATE | ✅ |
| 8 | Translation | gpt-5.2 | OVERKILL | OVERKILL | ✅ |
| 9 | Math proof | o3 | APPROPRIATE | ⚠️ OVERKILL | ❌ |
| 10 | Format JSON | claude-opus | OVERKILL | OVERKILL | ✅ |

**Accuracy: 90% (9/10)**

---

## 🔧 Key Functions

### `analyze_llm_call()` in `decision_module.py`
Main entry point. Called by IDE plugin or tests.

```python
result = analyze_llm_call(
    model="gpt-5.2",
    prompt="Fix the typo: 'teh'",
    system_prompt="You are a proofreader.",
    context={},
    tools=[],
    verbose=False  # Suppress agent logs
)

# Returns:
{
    'verdict': 'OVERKILL',
    'is_appropriate': False,
    'recommendation': {...},      # Contains minimum_tier, alternatives, reasoning
    'alternatives': [...],        # List of suitable alternatives
    'carbon_analysis': {...},     # CO₂ and cost breakdowns
    'summary': '...formatted report...',  # Beautiful formatted report (Dec 23 update)
    'metadata': {...}             # Extracted call metadata
}
```

### `format_recommendation_report()` in `decision_module.py` (NEW - Dec 23)
Generates the beautifully formatted recommendation report shown to users.

```python
summary = format_recommendation_report(
    verdict='OVERKILL',
    is_appropriate=False,
    metadata={...},
    current_model={...},
    carbon_analysis={...},
    alternatives=[...],
    task_analysis={...}
)
# Returns formatted string with all sections mentioned above
```

### Carbon Calculation
```python
# Carbon factor from model_cards.py
# Baseline: GPT-3.5 = 1.0
# Formula: CO₂ (grams) = tokens * carbon_factor * 0.0001

carbon_factors = {
    'budget': 0.3-1.0,
    'standard': 1.0-2.5,
    'premium': 3.0-5.0,
    'frontier': 8.0-12.0
}

# Example: GPT-5.2 with 35 tokens
# CO₂ = 35 × 8.0 × 0.0001 = 0.0280g
```

---

## 💡 Key Decisions Made

1. **No OpenAI API required** - All inference is local via Phi-3.5
2. **Verbose mode** - Added `verbose=False` to suppress agent execution logs
3. **Minimal documentation** - Only README.md and MEMORY.md
4. **VS Code first** - Ship as plugin from Marketplace (not CLI)
5. **Model hosting** - Plan to use HuggingFace or Ollama for distribution

---

## 🛠️ Commands

```bash
# Setup
cd AI-Gauge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run tests
python test_samples/test_model_comparison.py

# Build VS Code extension
cd ide_plugin
npm install && npm run compile
```

---

## 📋 Next Steps (Priority Order)

1. ✅ **Improve test output format** - DONE Dec 23
   - Moved logic to `decision_module.py` 
   - Created `format_recommendation_report()` function
   - Shows metadata, all tier models, reasoning, CO₂ calculation
   - Created `demo_single_test.py` to showcase single test case

2. **Set up local inference server** - IN PROGRESS
   - Created `inference_server.py` (Flask API)
   - Endpoints: `/health`, `/analyze`, `/models`, `/models/<tier>`
   - Missing: Flask and flask-cors installed in requirements.txt
   - Plan: Start server before VS Code plugin connects

3. **Package VS Code extension** - PLANNED
   - Pre-requisite: Inference server working
   - Command: `vsce package` in ide_plugin folder
   - Then publish to Marketplace

4. **Host model** - PLANNED
   - Upload GGUF to HuggingFace or Ollama
   - Users can auto-download on first plugin install

---

## 🔗 Dependencies

**Python** (requirements.txt):
- langchain, langgraph
- llama-cpp-python (with Metal support on Mac)
- python-dotenv

**VS Code Extension** (package.json):
- tree-sitter for code parsing
- TypeScript 5.0+

---

*Last Updated: December 22, 2025*
