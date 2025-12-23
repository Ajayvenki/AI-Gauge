#!/usr/bin/env python3
"""
AI-Gauge Single Test Demo
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from decision_module import analyze_llm_call


def run_single_demo():
    """Run a single test case demonstrating AI-Gauge analysis."""
    
    result = analyze_llm_call(
        model="gpt-5.2",
        prompt="Fix the typo: 'The quik brown fox jumps ovre the lazy dog'",
        system_prompt="You are a proofreader.",
        context={},
        tools=[],
        verbose=True
    )
    
    return result


if __name__ == "__main__":
    run_single_demo()
