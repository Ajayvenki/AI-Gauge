# Changelog

All notable changes to AI-Gauge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.4] - 2025-01-02

### Added
- **Package Optimization**: Added .vscodeignore to exclude development files from extension package
- **Runtime Package Detection**: Extension now properly detects runtime packages in any workspace location

### Changed
- **Extension Size**: Reduced from 33KB to 21KB (37% smaller) by excluding .d.ts and .js.map files
- **Packaging Strategy**: Switched to .vscodeignore for cleaner file inclusion/exclusion

### Fixed
- **Repository Detection**: Fixed extension failing to find runtime packages in new workspaces
- **Server Path Resolution**: Corrected inference server startup path for runtime package structure

## [0.4.3] - 2025-01-02

### Added
- **Runtime Package**: Minimal 33KB distribution package for clean user installation
- **Automated Setup**: One-command setup.sh handles Ollama installation and model download
- **Flattened Structure**: Runtime package uses direct imports for better compatibility

### Changed
- **Installation Process**: Users download runtime package instead of cloning full repository
- **Import Structure**: Fixed relative imports in runtime package for standalone operation
- **Documentation**: Updated README with new installation instructions

### Fixed
- **Import Issues**: Resolved module import problems in flattened runtime structure
- **Server Startup**: Inference server now starts correctly in runtime package environment

## [0.4.2] - 2025-01-02

### Changed
- **Architecture**: Extension now detects and uses local AI-Gauge repository instead of bundled files
- **Setup process**: Users clone repository and run setup script for complete environment configuration
- **Reliability**: Eliminated bundling issues by using local repository with proper Python environment

### Added
- **Repository detection**: Automatic detection of AI-Gauge repository in common locations
- **Enhanced setup**: setup.sh now automatically installs Ollama and pulls models
- **Better error handling**: Clear guidance when repository is not found

## [0.4.1] - 2025-01-02
- **Status display**: Dynamic backend reporting based on actual configuration rather than hardcoded values

## [0.4.0] - 2025-01-02

### Added
- **Self-contained extension**: Python code now bundled with VSIX package
- **Automatic file deployment**: Extension copies bundled files to user storage on first use
- **Zero-configuration setup**: Users just install extension, no repository cloning required
- **Improved user experience**: Seamless installation and setup process

### Changed
- **Installation flow**: Simplified to single extension install
- **Architecture**: Extension manages Python environment in user storage
- **Dependencies**: Python code and requirements bundled with extension
- **User requirements**: No longer need to clone repository or open workspace

### Removed
- **Workspace dependency**: Extension no longer requires AI-Gauge project in workspace
- **Manual repository cloning**: Users don't need to interact with git repository

### Technical
- Added `files` field to package.json to include Python source and requirements
- Modified `startInferenceServer()` to copy bundled files to `context.globalStoragePath`
- Updated `checkServerAvailable()` to check bundled files instead of workspace
- Enhanced `setupPythonEnvironment()` to work with user storage directory
- Added `copyDirectory()` helper for recursive file copying

## [0.3.10] - 2025-01-02

### Added
- **Workspace detection**: Extension now checks if AI-Gauge project is open in workspace
- **Automatic dependency installation**: Extension installs Python requirements.txt automatically
- **Improved server management**: Server starts from workspace directory instead of extension directory

### Changed
- **Installation requirements**: Clarified that AI-Gauge repository must be cloned and opened in VS Code
- **Server path resolution**: Extension finds server code in user's workspace, not extension installation
- **Setup flow**: Added workspace validation before attempting server startup

### Fixed
- **"fetch failed" errors**: Extension now properly detects when server code is not available
- **Server startup failures**: Fixed path resolution for server startup from workspace
- **Import conflicts**: Resolved Python import issues in inference server
- **Missing dependencies**: Extension now installs requirements.txt automatically

### Technical
- Added `checkServerAvailable()` function to validate workspace setup
- Modified `startInferenceServer()` to use workspace path instead of extension path
- Enhanced `setupPythonEnvironment()` to install requirements.txt
- Updated documentation to reflect workspace requirements

## [0.3.9] - 2025-12-24

### Added
- **Server-first architecture** with automatic inference server management
- **Agent orchestration pipeline** using LangGraph with 3 specialized agents
- **Automatic server lifecycle management** in VS Code extension
- **Health monitoring and recovery** for reliable operation
- **Model cards database** as single source of truth for all model metadata
- **Ollama integration within analyzer agent** for task complexity assessment

### Changed
- **Major**: Implemented server-first design - extension manages server, server orchestrates agents
- **Major**: Removed Ollama direct mode, unified to server-only architecture
- **Major**: Updated all documentation to reflect agent orchestration architecture
- Extension automatically starts/stops inference server with health checks
- Agents use Ollama SLM within structured workflow instead of direct calls
- Simplified user experience with zero manual configuration required

### Removed
- Direct Ollama mode from extension
- HuggingFace cloud fallback support
- Manual server management requirements
- Complex multi-backend configuration options

### Fixed
- Architectural inconsistency where direct mode bypassed agent orchestration
- Extension reliability issues with automatic setup and health monitoring
- Documentation outdated for server-first design

### Technical
- Added server management functions in `extension.ts` (ensureInferenceServer, startInferenceServer, checkServerHealth)
- Updated `aiGaugeClient.ts` to server-only communication
- Enhanced `decision_module.py` with active LangGraph 3-agent pipeline
- Implemented `model_cards.py` as centralized metadata database
- Published to VS Code marketplace as stable release

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