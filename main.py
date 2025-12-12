"""AI Gauge — Developer's Code (Poor Example)

This simulates a typical developer's code that:
  • Uses an oversized frontier model (GPT-5.2) for what appears to be a complex task
  • The task LOOKS sophisticated (enterprise-grade prompt engineering) but is actually trivial
  • Demonstrates the problem: developers can't easily assess if their model choice is efficient

The decision module (running in the background) will intercept and analyze this.

Run:
    python main.py
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — Appears sophisticated, actually simple task
# ---------------------------------------------------------------------------

# Enterprise-grade system prompt (looks important!)
SYSTEM_PROMPT = """\
You are an elite AI communications specialist at a Fortune 500 developer tools company.
Your expertise spans technical writing, UX copy, brand voice consistency, and conversion optimization.
Apply the following frameworks:
- AIDA (Attention, Interest, Desire, Action) for engagement
- Plain language principles (Flesch-Kincaid Grade 8 or below)
- Developer empathy mapping for audience resonance
Maintain strict adherence to brand guidelines while maximizing clarity and impact.
"""

# Detailed context object (seems complex!)
CONTEXT = {
    "company": {
        "name": "AI Gauge",
        "industry": "Developer Tools",
        "brand_voice": "Professional, empathetic, technically precise",
        "target_market": "B2B SaaS",
    },
    "project": {
        "type": "VS Code Extension",
        "surface": "Onboarding panel - first-run experience",
        "goal": "User activation and feature discovery",
    },
    "audience": {
        "primary": "Software engineers (mid to senior level)",
        "secondary": "Engineering managers, DevOps practitioners",
        "pain_points": ["LLM cost visibility", "Environmental impact awareness", "API optimization"],
    },
    "constraints": {
        "max_words": 30,
        "reading_time_seconds": 5,
        "avoid": ["hype", "buzzwords", "exclamation marks", "jargon"],
        "required_elements": ["value proposition", "action hint"],
    },
    "success_metrics": {
        "target_engagement_rate": 0.75,
        "target_feature_discovery": 0.60,
    },
}

# The actual task (trivially simple - just rewrite one sentence!)
USER_PROMPT = """\
Rewrite the following onboarding tooltip to improve clarity and user engagement:

ORIGINAL: "This extension helps you track LLM usage for greener prompts and lower costs."

Apply all brand guidelines, audience insights, and engagement frameworks from the context.
Deliver a single, refined sentence optimized for the specified constraints.
"""

INSTRUCTIONS = "Return exactly ONE sentence. No explanation, no alternatives, no preamble."

# Oversized model for a trivial task — the "bad" choice
MODEL_ID = "gpt-5.2"
MODEL_CONFIG = {
    "temperature": 0.3,
    "max_tokens": 100,
    "top_p": 0.95,
}

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def invoke_llm() -> Dict[str, Any]:
    """Invoke the LLM with the configured prompt. Returns metadata + response."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    # Build the full prompt
    full_user_content = (
        f"## Context\n```json\n{json.dumps(CONTEXT, indent=2)}\n```\n\n"
        f"## Task\n{USER_PROMPT}\n\n"
        f"## Output Requirements\n{INSTRUCTIONS}"
    )

    # Metadata for decision module to analyze
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_id": MODEL_ID,
        "model_config": MODEL_CONFIG,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "context": CONTEXT,
        "instructions": INSTRUCTIONS,
        "full_prompt_preview": full_user_content[:500] + "..." if len(full_user_content) > 500 else full_user_content,
    }

    # Invoke the model
    start_time = time.time()

    if not api_key:
        # Mock response for demo
        llm_output = "Monitor your AI costs and carbon footprint directly in VS Code."
        metadata["_mock"] = True
    else:
        try:
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=MODEL_ID,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_user_content},
                ],
            )
            llm_output = response.output_text or "(empty response)"
        except Exception as e:
            llm_output = f"(error) {e}"
            metadata["_error"] = str(e)

    elapsed_ms = (time.time() - start_time) * 1000

    # Add response metadata
    metadata["llm_output"] = llm_output
    metadata["response_time_ms"] = round(elapsed_ms, 2)

    return metadata


def main() -> None:
    """Run the 'poor example' script."""
    print("=" * 70)
    print("🔴 DEVELOPER CODE — Using GPT-5.2 for simple copy editing")
    print("=" * 70)
    print()

    result = invoke_llm()

    print(f"Model: {result['model_id']}")
    print(f"Response time: {result['response_time_ms']}ms")
    print(f"\n📝 Output:\n   \"{result['llm_output']}\"")
    print()
    print("-" * 70)
    print("Full metadata (for decision module):")
    print("-" * 70)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

