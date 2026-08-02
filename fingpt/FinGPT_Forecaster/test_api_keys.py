#!/usr/bin/env python3
"""
Test script to verify API key configuration for prepare_data.ipynb
This script helps diagnose the "num_rows is zero" issue by checking API key availability.
"""

import os
import sys

def test_api_keys():
    """Test if API keys are properly configured"""
    
    print("🔍 Testing API Key Configuration for FinGPT Forecaster")
    print("=" * 60)
    
    # Test Finnhub API Key
    finnhub_key = os.environ.get("FINNHUB_API_KEY")
    if finnhub_key and finnhub_key != "your finnhub key":
        print("✅ FINNHUB_API_KEY is set")
        try:
            import finnhub
            client = finnhub.Client(api_key=finnhub_key)
            # Test with a simple call
            client.company_profile2(symbol="AAPL")
            print("✅ Finnhub API connection successful")
        except Exception as e:
            print(f"❌ Finnhub API connection failed: {e}")
            return False
    else:
        print("❌ FINNHUB_API_KEY is not set or is still placeholder")
        print("   Set it with: export FINNHUB_API_KEY=your_key_here")
        return False
    
    # Test OpenAI API Key
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and openai_key != "your openai key":
        print("✅ OPENAI_API_KEY is set")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            print("✅ OpenAI client initialized successfully")
        except Exception as e:
            print(f"❌ OpenAI client initialization failed: {e}")
            return False
    else:
        print("❌ OPENAI_API_KEY is not set or is still placeholder")
        print("   Set it with: export OPENAI_API_KEY=your_key_here")
        return False
    
    print("=" * 60)
    print("✅ All API keys are properly configured!")
    print("You can now run prepare_data.ipynb without the 'num_rows is zero' issue.")
    return True

if __name__ == "__main__":
    success = test_api_keys()
    sys.exit(0 if success else 1)