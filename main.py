"""AI Gauge - bad example fixture

One "reasonable-looking" GenAI call that *appears* important to a developer,
but is actually a small task from an LLM perspective.

This script ONLY:
- makes a call with a large model (gpt-5)
- prints a metadata payload for our future IDE plugin

Stop here. No decision agent, no suggestions.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI


def run() -> None:
    system_prompt = (
        "You are an expert product copy editor for a developer tools company. "
        "Be precise and concise."
    )

    # Looks important/complex to the developer; output is tiny.
    context = {
        "product": "AI Gauge",
        "surface": "VS Code extension onboarding panel",
        "audience": "software engineers",
        "tone": "professional",
        "constraints": {
            "max_words": 30,
            "avoid": ["hype", "buzzwords", "exclamation marks"],
        },
    }

    user_prompt = (
        "Rewrite this onboarding tip to be clearer:\n"
        "'This extension helps you track LLM usage for greener prompts and lower costs.'"
    )

    instructions = "Return exactly ONE sentence under 18 words. No explanation."

    # Included so the plugin can extract tool metadata.
    # (We are not executing tools here.)
    tools = [
        {
            "type": "function",
            "name": "noop",
            "description": "Placeholder tool definition.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        }
    ]

    model_planned = "gpt-5"
    model_config = {"temperature": 0.2, "max_tokens": 256, "top_p": 1.0}

    # Placeholder only. We will compute later via decision agent.
    cost_estimated_usd = 0.02
    carbon_estimated_g = 0.3

    metadata = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "context": context,
        "instructions": instructions,
        "tools": tools,
        "model_planned": model_planned,
        "model_config": model_config,
        "cost_estimated_usd": cost_estimated_usd,
        "carbon_estimated_g": carbon_estimated_g,
    }

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model_planned,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context: {json.dumps(context)}\n"
                f"Instructions: {instructions}\n"
                f"Task: {user_prompt}",
            },
        ],
        temperature=model_config["temperature"],
        max_tokens=model_config["max_tokens"],
        top_p=model_config["top_p"],
    )

    metadata["llm_output"] = response.choices[0].message.content
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    load_dotenv()
    run()
