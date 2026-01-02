#!/usr/bin/env python3
"""
AI-Gauge Single Test Demo
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.decision_module import analyze_llm_call


def run_simple_test():
    """Test simple text correction task."""
    print("=== SIMPLE TASK TEST ===")
    result = analyze_llm_call(
        model="gpt-4",
        prompt="Fix the spelling: 'recieve'",
        system_prompt="",
        context={},
        tools=[],
        verbose=False
    )
    print(result['summary'])
    print(f"Verdict: {result['verdict']}")
    print(f"Recommendation: {result.get('recommendation', {}).get('model', 'None')}")
    print()
    return result


def run_complex_test():
    """Test complex code analysis task."""
    print("=== COMPLEX TASK TEST ===")
    result = analyze_llm_call(
        model="gpt-4",
        prompt="Analyze this Python code for security vulnerabilities, performance issues, and suggest architectural improvements: [paste 50+ lines of complex code here]",
        system_prompt="",
        context={},
        tools=[],
        verbose=False
    )
    print(result['summary'])
    print(f"Verdict: {result['verdict']}")
    print(f"Recommendation: {result.get('recommendation', {}).get('model', 'None')}")
    print()
    return result


def run_single_demo():
    """Run a single test case demonstrating AI-Gauge analysis."""
    
    result = analyze_llm_call(
        model="gpt-5.2",
        prompt="Fix the typo: 'The quik brown fox jumps ovre the lazy dog'",
        system_prompt="You are a proofreader.",
        context={},
        tools=[],
        verbose=False  # Hide agent execution details from users
    )
    
    # Print only the final clean report
    print(result['summary'])
    
    return result


if __name__ == "__main__":
    print("Testing AI-Gauge analysis with different task complexities...\n")
    
    # Test simple task
    simple_result = run_simple_test()
    
    # Test complex task  
    complex_result = run_complex_test()
    
    print("=== SUMMARY ===")
    print(f"Simple task verdict: {simple_result['verdict']}")
    print(f"Complex task verdict: {complex_result['verdict']}")
    
    if simple_result['verdict'] == 'OVERKILL' and complex_result['verdict'] == 'APPROPRIATE':
        print("✅ SUCCESS: AI-Gauge correctly distinguishes task complexity!")
    else:
        print("❌ ISSUE: AI-Gauge is not properly analyzing task complexity")
