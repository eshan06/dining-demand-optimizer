#!/usr/bin/env python3
"""
Quick Demo Runner for Palantir Hackathon Judges
==============================================

Simple script to run the ML algorithm demonstration.
"""

import sys
import os
import subprocess

def main():
    print("🧠 ML Algorithm Demo Runner")
    print("=" * 40)
    print("Choose a demo to run:")
    print("1. Judge Demo (Recommended) - Simple, focused")
    print("2. Full ML Thinking Demo - Detailed, comprehensive")
    print("3. Working Demo - Basic functionality")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        print("\n🚀 Running Judge Demo...")
        subprocess.run([sys.executable, "judge_demo.py"])
    elif choice == "2":
        print("\n🚀 Running Full ML Thinking Demo...")
        subprocess.run([sys.executable, "ml_thinking_demo.py"])
    elif choice == "3":
        print("\n🚀 Running Working Demo...")
        subprocess.run([sys.executable, "working_demo.py"])
    else:
        print("Invalid choice. Running Judge Demo by default...")
        subprocess.run([sys.executable, "judge_demo.py"])

if __name__ == "__main__":
    main()
