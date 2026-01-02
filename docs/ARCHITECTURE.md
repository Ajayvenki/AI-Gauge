# AI-Gauge Architecture Guide

## Overview

AI-Gauge is a sophisticated system for optimizing LLM API costs through intelligent pre-call analysis. The system uses a **server-first architecture** with **agent orchestration** to intercept API calls before execution, analyze task requirements using local AI, and provide cost/carbon estimates to help developers make informed decisions.

## System Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   VS Code       │    │  Inference      │    │   Decision      │    │   Local AI      │
│  Extension      │◄──►│   Server        │◄──►│   Module        │◄──►│   (Ollama)     │
│                 │    │                 │    │   Agents        │    │                 │
│ • Call Intercept│    │ • API Endpoint  │    │ • Agent Pipeline│    │ • SLM for       │
│ • Server Mgmt   │    │ • Health Checks │    │ • Model Cards   │    │   Analysis      │
│ • UI Integration│    │ • Auto-startup  │    │ • Orchestration │    │ • Task Assess   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Call Interception**: VS Code extension detects LLM API calls in user code
2. **Server Health Check**: Extension verifies inference server is running
3. **API Call**: Extension sends metadata to local inference server
4. **Agent Orchestration**: Decision module runs 3-agent LangGraph pipeline
5. **AI Analysis**: Analyzer agent uses Ollama SLM to assess task complexity
6. **Model Card Lookup**: All tier/cost/CO2 data retrieved from model cards
7. **Recommendation**: Structured analysis returned to extension
8. **User Decision**: Developer chooses whether to proceed with recommendation

## Technical Implementation

### VS Code Extension (TypeScript)

**Location**: `ide_plugin/`
**Purpose**: IDE integration, server management, and user interaction

**Key Components**:
- `src/extension.ts`: Main extension entry point with server lifecycle management
- `src/llmCallDetector.ts`: API call pattern recognition
- `src/aiGaugeClient.ts`: Server communication (server-mode only)
- `src/diagnosticsProvider.ts`: VS Code diagnostics integration
- `src/inlineHintsProvider.ts`: Real-time code hints

**Server Management**:
- Automatic inference server startup on extension activation
- Health checks before analysis attempts
- Graceful server shutdown on deactivation
- Port conflict handling and retries

**Call Detection**:
- Regex-based pattern matching for common LLM SDK calls
- AST analysis for complex call structures
- Real-time monitoring of open files

### Inference Server (Python/Flask)

**Location**: `src/inference_server.py`
**Purpose**: REST API endpoint for extension communication

**Key Features**:
- Flask-based REST API (`/analyze`, `/health`)
- Decision module integration
- CORS support for VS Code extension
- Error handling and logging

### Decision Module - Agent Orchestration (Python)

**Location**: `src/decision_module.py`
**Purpose**: Core analysis engine with LangGraph multi-agent system

**Architecture**: LangGraph-based 3-agent pipeline

**Agents**:
1. **Metadata Extractor**: Raw data collection and preprocessing from API calls
2. **Analyzer**: Uses Ollama SLM to assess task complexity and model appropriateness
3. **Reporter**: Generates human-readable recommendations using model card data

**State Management**:
- TypedDict-based state schema for agent communication
- Immutable state transitions between agents
- Comprehensive error handling and fallback logic

### Model Cards - Single Source of Truth (Python)

**Location**: `src/model_cards.py`
**Purpose**: Authoritative database of all model metadata

**Contains**:
- Model specifications (context windows, capabilities)
- Pricing information (per-token rates, regional variations)
- Carbon factors (energy consumption estimates)
- Tier classifications (budget/standard/premium/frontier)
- Performance metrics (latency, throughput)

**Key Functions**:
- `get_model_card()`: Retrieve complete model information
- `TIER_RANKINGS`: Hierarchical model tier system
- `is_model_overkill_by_tier()`: Tier-based overkill detection

### Local AI Inference - Ollama SLM (Python)

**Location**: `src/local_inference.py`
**Purpose**: Ollama integration for AI-powered task analysis

**Role in Architecture**:
- Provides Small Language Model (SLM) capabilities within analyzer agent
- Task complexity assessment and reasoning
- Local execution for privacy and performance
- Integrated into decision module agent pipeline

**Supported Models**:
- **Primary**: Fine-tuned Phi-3.5 model via Ollama
- **Capabilities**: Task analysis, complexity classification, model recommendations

## Model Selection & Training

### Training Data

**Dataset Composition**:
- 10,000+ labeled examples of LLM tasks
- Complexity classifications (trivial/simple/moderate/complex/expert)
- Cost and performance correlations
- Multi-provider model comparisons

**Data Sources**:
- Synthetic task generation
- Real API call logs (anonymized)
- Expert annotations
- Performance benchmarks

### Fine-tuning Process

**Methodology**:
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Task-specific prompt engineering
- Multi-turn conversation support
- Output format standardization

**Quality Assurance**:
- Cross-validation on held-out datasets
- Human evaluation of recommendations
- A/B testing with real users
- Continuous model improvement

## Cost & Carbon Calculations

### Cost Modeling

**Factors Considered**:
- Base model pricing (per-token rates)
- Input vs output token differentiation
- Regional pricing variations
- Volume discounts and tiers

**Real-time Updates**:
- API polling for current pricing
- Fallback to cached rates
- Provider-specific adjustments

### Carbon Footprint Analysis

**Methodology**:
- Energy consumption per model inference
- Data center location carbon intensity
- Renewable energy matching
- Hardware efficiency factors

**Data Sources**:
- Academic research (Patterson et al., Luccioni et al.)
- Provider sustainability reports
- Industry benchmarks
- Geographic carbon intensity databases

## Privacy & Security

### Data Handling

**Privacy Principles**:
- **Local-First Architecture**: All analysis happens on user's machine
- **No External Data Transmission**: Code and metadata stay local
- **Ephemeral Processing**: No persistent data retention
- **User-Controlled Server**: Inference server runs locally under user control

**Security Measures**:
- Local inference server with no external network access
- Secure inter-process communication between extension and server
- No API keys transmitted externally
- Minimal attack surface through local-only operation

### Server Architecture Security

**Local Server Benefits**:
- No internet exposure - server only accessible locally
- Automatic lifecycle management by extension
- Health checks ensure server integrity
- Graceful shutdown on extension deactivation

**Process Isolation**:
- Separate Python process for inference server
- Controlled communication via HTTP localhost
- Extension manages server startup/shutdown
- No persistent background services

### Compliance Considerations

**GDPR/CCPA Compliance**:
- No personal data processing
- Local-only operation
- User-controlled data retention

**Enterprise Features**:
- Air-gapped deployment support
- Custom model hosting
- Audit logging capabilities

## Performance Optimization

### Inference Speed

**Optimizations**:
- Model quantization (4-bit)
- GPU acceleration when available
- Request batching
- Caching of frequent analyses

**Latency Targets**:
- <500ms for simple tasks
- <2s for complex analysis
- Non-blocking UI experience

### Memory Management

**Resource Usage**:
- ~2GB RAM for model loading
- ~500MB additional for inference
- GPU memory optimization
- Automatic cleanup and caching

## Deployment & Distribution

### VS Code Marketplace

**Publishing Process**:
- Automated build pipeline
- Cross-platform compilation
- Version management
- Update distribution

### Model Distribution

**Ollama Integration**:
- Automatic model download
- Version compatibility checking
- Fallback handling
- Update management

### Installation Flow

**User Experience**:
1. One-click marketplace install
2. Automatic dependency setup
3. Model download and verification
4. Configuration and testing

## Challenges & Solutions

### Technical Challenges

**Model Accuracy**:
- Balancing precision vs speed
- Handling edge cases and ambiguous tasks
- Adapting to new model releases
- Cross-provider compatibility

**Solutions**:
- Continuous model improvement
- User feedback integration
- Ensemble analysis approaches
- Regular benchmark updates

### User Experience Challenges

**Adoption Barriers**:
- Learning curve for recommendations
- Trust in automated suggestions
- Integration with existing workflows

**Solutions**:
- Progressive disclosure of features
- Transparent reasoning explanations
- Customizable sensitivity settings
- Extensive documentation and examples

### Scalability Challenges

**Performance at Scale**:
- Large codebase analysis
- High-frequency API usage
- Team collaboration features

**Solutions**:
- Efficient caching strategies
- Background processing
- Distributed analysis options
- Enterprise deployment patterns

## Future Roadmap

### Short Term (3-6 months)
- Enhanced model accuracy through user feedback
- Additional provider support
- Improved UI/UX in VS Code
- Performance optimizations

### Medium Term (6-12 months)
- Multi-language SDK support
- Team dashboards and analytics
- Advanced carbon modeling
- Integration APIs

### Long Term (1-2 years)
- Enterprise features (SSO, compliance)
- Custom model training options
- Industry-specific optimizations
- Research partnerships

## Development & Testing

### Testing Strategy

**Unit Tests**: Component-level functionality
**Integration Tests**: End-to-end workflows
**Performance Tests**: Latency and resource usage
**User Acceptance Tests**: Real-world validation

### Quality Assurance

**Code Quality**:
- TypeScript strict mode
- Python type hints
- Automated linting and formatting
- Security scanning

**Model Quality**:
- Accuracy benchmarks
- Bias and fairness testing
- Robustness validation
- Continuous monitoring

## Contributing

### Development Setup

1. Clone repository
2. Install dependencies
3. Set up development environment
4. Run test suite
5. Build and test extension

### Code Organization

**Standards**:
- Clear separation of concerns
- Comprehensive documentation
- Type safety
- Error handling

**Review Process**:
- Code review requirements
- Testing standards
- Documentation updates
- Performance validation

This architecture guide provides the foundation for understanding and contributing to AI-Gauge. For specific implementation details, refer to the inline code documentation and issue discussions.