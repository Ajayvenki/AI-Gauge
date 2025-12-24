# AI-Gauge Architecture Guide

## Overview

AI-Gauge is a sophisticated system for optimizing LLM API costs through intelligent pre-call analysis. The system intercepts API calls before execution, analyzes task requirements using local AI, and provides cost/carbon estimates to help developers make informed decisions.

## System Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   VS Code       │    │  Decision       │    │   Local AI      │
│  Extension      │◄──►│   Pipeline      │◄──►│   Inference     │
│                 │    │                 │    │                 │
│ • Call Intercept│    │ • Metadata      │    │ • Phi-3.5 Model │
│ • UI Integration│    │ • Analysis      │    │ • Cost Analysis │
│ • User Feedback │    │ • Recommendations│    │ • Carbon Calc  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Call Interception**: VS Code extension detects LLM API calls in user code
2. **Metadata Extraction**: Comprehensive analysis of call parameters, context, and requirements
3. **AI Analysis**: Local model assesses task complexity and optimal model selection
4. **Recommendation Generation**: Cost, carbon, and performance analysis
5. **User Decision**: Developer chooses whether to proceed with recommendation

## Technical Implementation

### VS Code Extension (TypeScript)

**Location**: `ide_plugin/`
**Purpose**: IDE integration and user interaction

**Key Components**:
- `src/extension.ts`: Main extension entry point
- `src/llmCallDetector.ts`: API call pattern recognition
- `src/diagnosticsProvider.ts`: VS Code diagnostics integration
- `src/inlineHintsProvider.ts`: Real-time code hints

**Call Detection**:
- Regex-based pattern matching for common LLM SDK calls
- AST analysis for complex call structures
- Real-time monitoring of open files

### Decision Pipeline (Python)

**Location**: `src/decision_module.py`
**Purpose**: Orchestrates the analysis workflow

**Architecture**: LangGraph-based multi-agent system

**Agents**:
1. **Metadata Extractor**: Raw data collection and preprocessing
2. **Analyzer**: AI-powered task complexity assessment
3. **Reporter**: Human-readable recommendation generation

**State Management**:
- TypedDict-based state schema
- Immutable state transitions
- Comprehensive error handling

### Local Inference Engine (Python)

**Location**: `src/local_inference.py`
**Purpose**: Local AI model management and inference

**Supported Backends**:
- **Primary**: Ollama (recommended for ease of use)
- **Fallback**: llama-cpp-python (advanced users)
- **Future**: Direct model loading

**Model Details**:
- **Base Model**: Phi-3.5 (3.8B parameters)
- **Fine-tuning**: Custom dataset for cost optimization analysis
- **Quantization**: 4-bit quantization for efficiency
- **Context Window**: 4096 tokens

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
- **Zero External Data**: All analysis happens locally
- **No Call Content Storage**: Only metadata is processed
- **Ephemeral Processing**: No persistent data retention

**Security Measures**:
- Local model execution only
- No internet connectivity required for core functionality
- Secure API key handling
- Minimal attack surface

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