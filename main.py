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

    # Load API key from environment (dotenv is called at program start)
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key) if api_key else None

    # Build a single prompt string for the Responses API input
    prompt_text = (
        f"System: {system_prompt}\n"
        f"Context: {json.dumps(context)}\n"
        f"Instructions: {instructions}\n"
        f"Task: {user_prompt}"
    )

    llm_output = None

    if client is not None:
        try:
            response = client.responses.create(
                model=model_planned,
                input=prompt_text,
                temperature=model_config["temperature"],
                max_tokens=model_config["max_tokens"],
            )

            # Responses can surface text in a few places depending on SDK version.
            # Try common accessors safely.
            if hasattr(response, "output_text") and response.output_text:
                llm_output = response.output_text
            else:
                # Try nested output content
                try:
                    # new SDK shapes: response.output -> list(dict)
                    out = response.output
                    if isinstance(out, (list, tuple)) and len(out) > 0:
                        first = out[0]
                        # content can be list of dicts with 'text' or 'type' keys
                        content = first.get("content") if isinstance(first, dict) else None
                        if isinstance(content, (list, tuple)) and len(content) > 0:
                            piece = content[0]
                            if isinstance(piece, dict) and "text" in piece:
                                llm_output = piece["text"]
                            elif isinstance(piece, str):
                                llm_output = piece
                except Exception:
                    llm_output = None

            # As a last resort try choices (compatibility layer)
            if not llm_output and hasattr(response, "choices"):
                try:
                    c = response.choices[0]
                    # Some SDKs put message content: c.message.content
                    if hasattr(c, "message") and getattr(c.message, "content", None):
                        llm_output = c.message.content
                    elif getattr(c, "text", None):
                        llm_output = c.text
                except Exception:
                    pass

        except Exception as exc:  # pragma: no cover - runtime network errors
            # If the API fails (bad key, network), fall back to a mocked output
            llm_output = None
            print(f"Warning: responses.create failed: {exc}")

    # If there's no API key or the call failed, provide a deterministic mock output.
    if not llm_output:
        # Produce a concise, single-sentence output consistent with instructions
        llm_output = (
            "Track LLM usage in VS Code to reduce carbon and cost with simple metrics."
        )

    metadata["llm_output"] = llm_output
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    load_dotenv()
    run()
