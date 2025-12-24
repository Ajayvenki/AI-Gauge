# Changelog

All notable changes to AI-Gauge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-12-24

### Added
- Comprehensive architecture documentation in `docs/ARCHITECTURE.md`
- Improved JSON parsing robustness for better model response handling
- Enhanced error handling and fallback mechanisms

### Changed
- **Major**: Removed llama-cpp-python fallback - now Ollama-only for simplicity
- **Major**: Complete README rewrite for professional presentation
- Improved model inference reliability with temperature optimization
- Streamlined documentation structure

### Removed
- llama-cpp-python backend support
- Verbose comparison tables from README
- Technical details from main README (moved to architecture docs)

### Fixed
- JSON parsing failures causing incorrect complexity assessments
- Inconsistent test results due to malformed model responses
- Memory file exposure in git repository

### Technical
- Enhanced `parse_model_response()` with regex fallback parsing
- Added `docs/MEMORY.md` to `.gitignore`
- Updated Ollama temperature to 0.0 for deterministic responses
- Improved local inference error handling

## [0.2.1] - 2025-12-20

### Added
- Initial VS Code marketplace release
- Basic Ollama integration
- Comprehensive test suite

### Fixed
- Installation and setup issues
- Model download automation

## [0.2.0] - 2025-12-15

### Added
- Core decision pipeline with LangGraph agents
- Local Phi-3.5 model fine-tuning
- VS Code extension framework
- Cost and carbon calculation modules

### Changed
- Migrated from simple analysis to multi-agent architecture

## [0.1.0] - 2025-12-01

### Added
- Initial prototype with basic LLM call interception
- Simple cost estimation
- Proof of concept implementation