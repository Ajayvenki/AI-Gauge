# AI-Gauge Runtime Package v0.4.3

This is the minimal runtime package for AI-Gauge. It contains only the essential files needed to run the AI-Gauge inference server.

## 🚀 Quick Start

1. **Run setup:**
   ```bash
   ./setup.sh
   ```
   This will install Python dependencies, Ollama, and download the AI-Gauge model.

2. **Install VS Code extension:**
   Search for "AI-Gauge" in VS Code extensions marketplace.

3. **Start coding!**
   The extension will automatically detect your local AI-Gauge installation and start the server when needed.

## What's Included

- `src/` - Python source code for the inference server
- `requirements.txt` - Python dependencies
- `setup.sh` - Automated setup script
- `README.md` - This file

## Requirements

- Python 3.8+
- macOS, Linux, or Windows
- Internet connection for model download

## Troubleshooting

If you encounter issues:
1. Ensure Ollama is running: `ollama list`
2. Check Python version: `python3 --version`
3. Verify dependencies: `pip list | grep -E "(flask|requests|torch)"`

For more help, visit: https://github.com/ajayvenki2910/ai-gauge
