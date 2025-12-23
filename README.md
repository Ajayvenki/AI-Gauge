# AI-Gauge: LLM Cost Optimizer

Analyze your LLM API calls and get intelligent recommendations to optimize costs and performance.

## Quick Start

### 1. Install Ollama
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Set up AI-Gauge Model
```bash
# Download and set up the AI-Gauge model
git clone https://github.com/your-repo/ai-gauge.git
cd ai-gauge
./setup.sh
```

### 3. Install VSCode Extension
- Open VSCode
- Go to Extensions (Ctrl+Shift+X)
- Search for "AI-Gauge"
- Install and reload

### 4. Start Using
The extension will automatically detect your local Ollama model and provide cost optimization suggestions for your LLM calls.

## How It Works

1. **Intercept**: VSCode extension monitors your LLM API calls
2. **Analyze**: Local AI-Gauge model assesses if your model choice is appropriate
3. **Recommend**: Get suggestions for more cost-effective alternatives
4. **Optimize**: Reduce costs by 60-70% while maintaining performance

## Manual Testing

```bash
python test_samples/demo_single_test.py
```

## Architecture

- **Ollama**: Primary inference backend (recommended)
- **llama-cpp-python**: Local fallback for advanced users
- **VSCode Extension**: IDE integration and user interface

## Model Details

- **Base Model**: Fine-tuned Phi-3.5 (3.8B parameters)
- **Specialization**: LLM call analysis and cost optimization
- **Format**: GGUF (optimized for local inference)
- **Size**: ~2.2GB

## Publishing to Ollama Registry

For developers: To publish the model to Ollama's registry for easy distribution:

```bash
# Create the model locally (using archived Modelfile)
ollama create ai-gauge -f archive/ollama_setup/Modelfile

# Push to registry (requires Ollama account)
ollama push ai-gauge
```

Once published, end users can simply run `ollama pull ai-gauge`.

## Troubleshooting

### Ollama Not Detected
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve
```

### Model Not Found
```bash
# Pull the AI-Gauge model
ollama pull ai-gauge
```

### Extension Issues
- Reload VSCode after installation
- Check VSCode developer console for errors
- Ensure Ollama is running on localhost:11434

## Development

### Setting Up Development Environment
```bash
# Clone repository
git clone https://github.com/your-repo/ai-gauge.git
cd ai-gauge

# Install Python dependencies
pip install -r requirements.txt

# Set up Ollama model
./setup.sh
```

### Project Structure
```
ai-gauge/
├── src/                    # Core Python modules
├── ide_plugin/            # VS Code extension
├── training_data/         # Model training data
├── test_samples/          # Test scripts and demos
├── docs/                  # Documentation
├── archive/               # Archived files and old versions
├── requirements.txt       # Python dependencies
├── setup.sh              # Installation script
└── README.md             # This file
```

### Testing
```bash
# Run unit tests
python -m pytest

# Run demo
python test_samples/demo_single_test.py
```

## License

MIT License - see LICENSE file for details.

The model was fine-tuned on a dataset of real LLM API calls with human annotations for:
- Task complexity assessment
- Appropriate model tier selection
- Cost-benefit analysis
- Carbon impact estimation

## Integration

AI-Gauge is part of the AI-Gauge VS Code extension, which automatically analyzes your code and provides inline recommendations for cost optimization.

### VS Code Extension Setup
1. Install the AI-Gauge extension
2. The extension will automatically connect to your local Ollama model
3. Get real-time cost optimization hints as you code

## License

MIT License - see LICENSE file for details.

## Contact

For questions or contributions, visit the [AI-Gauge GitHub repository](https://github.com/your-org/ai-gauge).