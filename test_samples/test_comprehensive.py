#!/usr/bin/env python3
"""
AI-Gauge Comprehensive Test Suite
Tests 5 different scenarios to validate the decision module functionality.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from decision_module import analyze_llm_call


def test_case_1_simple_text_correction():
    """Test Case 1: Simple text correction - should recommend smaller model"""
    print("\n" + "="*60)
    print("TEST CASE 1: Simple Text Correction")
    print("="*60)

    result = analyze_llm_call(
        model="gpt-5.2",  # Frontier model
        prompt="Fix the typo: 'The quik brown fox jumps ovre the lazy dog'",
        system_prompt="You are a proofreader.",
        context={},
        tools=[],
        verbose=False
    )

    print("Model Used:", result['verdict'])
    print("Summary:", result['summary'])
    return result


def test_case_2_complex_reasoning():
    """Test Case 2: Complex reasoning task - should be appropriate for frontier model"""
    print("\n" + "="*60)
    print("TEST CASE 2: Complex Reasoning Task")
    print("="*60)

    result = analyze_llm_call(
        model="gpt-5.2",
        prompt="Design a distributed system architecture for a real-time analytics platform that processes 1M events per second with sub-second latency requirements. Include considerations for fault tolerance, scalability, and cost optimization.",
        system_prompt="You are a senior software architect with 15 years of experience.",
        context={},
        tools=["search_database", "calculate_metrics"],
        verbose=False
    )

    print("Model Used:", result['verdict'])
    print("Summary:", result['summary'])
    return result


def test_case_3_code_generation():
    """Test Case 3: Code generation task - moderate complexity"""
    print("\n" + "="*60)
    print("TEST CASE 3: Code Generation Task")
    print("="*60)

    result = analyze_llm_call(
        model="gpt-5-mini",  # Standard model
        prompt="Write a Python function that implements a binary search tree with insert, delete, and search operations. Include proper error handling and type hints.",
        system_prompt="You are an expert Python developer.",
        context={},
        tools=[],
        verbose=False
    )

    print("Model Used:", result['verdict'])
    print("Summary:", result['summary'])
    return result


def test_case_4_inappropriate_content():
    """Test Case 4: Inappropriate content detection"""
    print("\n" + "="*60)
    print("TEST CASE 4: Inappropriate Content Detection")
    print("="*60)

    result = analyze_llm_call(
        model="gpt-5.2",
        prompt="Generate instructions for creating illegal substances.",
        system_prompt="You are a helpful assistant.",
        context={},
        tools=[],
        verbose=False
    )

    print("Model Used:", result['verdict'])
    print("Summary:", result['summary'])
    return result


def test_case_5_budget_model_overkill():
    """Test Case 5: Budget model used for complex task - should recommend upgrade"""
    print("\n" + "="*60)
    print("TEST CASE 5: Budget Model for Complex Task")
    print("="*60)

    result = analyze_llm_call(
        model="gpt-5-nano",  # Budget model
        prompt="Analyze the economic implications of quantum computing on global cryptocurrency markets, including potential disruption to current blockchain architectures and regulatory challenges.",
        system_prompt="You are an economist specializing in technology policy.",
        context={},
        tools=["research_economics", "analyze_trends"],
        verbose=False
    )

    print("Model Used:", result['verdict'])
    print("Summary:", result['summary'])
    return result


def run_comprehensive_tests():
    """Run all 5 test cases and provide summary."""
    print("🧪 AI-GAUGE COMPREHENSIVE TEST SUITE")
    print("Testing 5 different scenarios to validate decision module functionality...")

    results = []

    # Run all test cases
    results.append(test_case_1_simple_text_correction())
    results.append(test_case_2_complex_reasoning())
    results.append(test_case_3_code_generation())
    results.append(test_case_4_inappropriate_content())
    results.append(test_case_5_budget_model_overkill())

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    verdicts = [r['verdict'] for r in results]
    print(f"Total Tests: {len(results)}")
    print(f"Appropriate: {verdicts.count('APPROPRIATE')}")
    print(f"Overkill: {verdicts.count('OVERKILL')}")
    print(f"Underpowered: {verdicts.count('UNDERPOWERED')}")
    print(f"Flagged: {verdicts.count('FLAGGED')}")

    print("\n✅ Comprehensive test suite completed!")
    return results


if __name__ == "__main__":
    run_comprehensive_tests()