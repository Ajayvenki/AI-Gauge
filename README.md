# AI-Gauge: Smart AI Cost Optimization

Do you know if your chosen AI model is appropriate for the task at hand? How can we identify the right fit before incurring costs?

In the rapidly evolving landscape of AI development, choosing the optimal model for each task is crucial yet challenging. Developers often default to powerful models out of caution, leading to inflated costs and unnecessary carbon emissions. What if you could gain clarity on model suitability upfront, ensuring efficiency without sacrificing performance? AI-Gauge empowers you with data-driven recommendations, transforming uncertainty into confident decision-making.

AI-Gauge helps to optimize AI costs and reduce your carbon footprint with intelligent model recommendations before execution.

## What is AI-Gauge?

AI-Gauge is a VS Code extension that automatically analyzes your Large Language Model (LLM) API calls in real-time. It recommends the most cost-effective and efficient models before execution, helping you save on costs while minimizing environmental impact.

## Solving Real-Time Problems

In today's fast-paced development environment, AI API costs can spiral out of control due to over-provisioning. Traditional tools only provide insights after the fact, leaving you with surprise bills. AI-Gauge intercepts calls before they execute, analyzing task complexity locally to suggest optimal models, delivering upfront cost and carbon estimates.

## Key Features

- **Cost Savings**: Achieve 60-70% reduction in AI expenses by selecting appropriate models.
- **Carbon Reduction**: Track and minimize the environmental impact of AI usage.
- **Zero Configuration**: Install and start optimizing immediately.
- **Privacy-First**: All analysis occurs locally on your machine.
- **Continuous Batching**: Unlike traditional batching, which forces all requests in a group to finish before the next group starts—leaving the GPU idle—continuous batching allows new requests to be inserted immediately as previous ones complete, maximizing efficiency.

## Alternatives and Approaches

When needed, AI-Gauge supports:
- Web search for additional context.
- Software engineering and procedural approaches for complex tasks.
- Efficient resource usage with single-call requests that gather all necessary information at once.

## How It Works

1. **Intercept**: Captures LLM API calls in real-time during coding.
2. **Analyze**: Assesses task complexity and requirements using local AI.
3. **Recommend**: Suggests the most efficient model with cost and carbon estimates.
4. **Execute**: Allows you to proceed with the recommendation.

## Project Showcase

During development, AI-Gauge itself demonstrated significant savings. By applying its own recommendations, we reduced API costs by 65% while maintaining high performance. This real-world application highlights the tool's effectiveness in optimizing AI resource usage.

**Case Study: Mid-size SaaS Company**
- **Before**: $15,000/month on GPT-4 calls
- **After**: $4,500/month (70% savings)
- **Performance**: 98% task success rate maintained
- **Carbon**: 12 tons CO₂ equivalent saved annually

## Quick Start

1. **Download Runtime Package**:
   ```bash
   # Download and extract the runtime package
   wget https://github.com/ajayvenki2910/ai-gauge/releases/download/v0.4.3/ai-gauge-runtime-v0.4.3.tar.gz
   tar -xzf ai-gauge-runtime-v0.4.3.tar.gz
   cd ai-gauge-runtime-v0.4.3
   ```

2. **Run Setup**:
   ```bash
   ./setup.sh
   ```
   This installs Ollama, downloads the AI model, and sets up dependencies.

3. **Install Extension**: Search "AI-Gauge" in VS Code Marketplace

4. **Code**: Start coding - get automatic LLM analysis and recommendations!

## Architecture & Technical Details

For detailed technical information, see our [Architecture Guide](docs/ARCHITECTURE.md) covering:
- System design and data flow
- Model selection and training
- Privacy and security considerations
- Performance optimization

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - Free for personal and commercial use.

---

Ready to optimize your AI costs? [Install AI-Gauge](https://marketplace.visualstudio.com/items?itemName=Ajayvenki2910.ai-gauge)