# AI-Gauge: Smart AI Cost Optimization

AI-Gauge is a VS Code extension that automatically analyzes your LLM API calls and recommends the most cost-effective models before execution. Save 60-70% on AI costs while maintaining performance and reducing your carbon footprint.

## The Problem

AI API costs are spiraling out of control. Organizations spend thousands monthly on over-provisioned models, with no visibility into which calls are truly necessary. Traditional monitoring tools only show costs after the fact - by then it's too late.

## The Solution

AI-Gauge intercepts your API calls **before** they execute, analyzing task complexity with a local AI model to recommend the optimal model. Get cost and carbon estimates upfront, not as a surprise bill.

### Key Benefits
- **60-70% Cost Savings**: Automatically switch to appropriate models
- **Carbon Reduction**: Track and minimize AI's environmental impact
- **Zero Configuration**: Install and start saving immediately
- **Privacy-First**: All analysis happens locally on your machine

## How It Works

1. **Intercept**: Catches LLM API calls in real-time as you code
2. **Analyze**: Uses local AI to assess task complexity and requirements
3. **Recommend**: Suggests the most efficient model with cost/carbon estimates
4. **Execute**: You decide whether to proceed with the recommendation

## Quick Start

1. Install from VS Code Marketplace: Search "AI-Gauge"
2. The extension automatically sets up Ollama and downloads the analysis model
3. Start coding - get inline recommendations for every API call

## Real Impact

**Mid-size SaaS Company Case Study:**
- **Before**: $15K/month on GPT-4 calls
- **After**: $4.5K/month (70% savings)
- **Performance**: 98% task success rate maintained
- **Carbon**: 12 tons CO₂ equivalent saved annually

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

**Ready to optimize your AI costs?** [Install AI-Gauge](https://marketplace.visualstudio.com/items?itemName=Ajayvenki2910.ai-gauge)