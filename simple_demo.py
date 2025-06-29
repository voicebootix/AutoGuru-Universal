#!/usr/bin/env python3
"""
AutoGuru Universal - Simple Demo
"""

import os
import openai

def analyze_business_content(content, api_key):
    """Simple business content analysis"""
    try:
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a business analyst. Analyze this content and provide: 1) Business niche, 2) Target audience, 3) Brand voice, 4) Viral potential score (0-100). Respond in JSON format."},
                {"role": "user", "content": content}
            ],
            max_tokens=300
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {e}"

def main():
    print("AutoGuru Universal - Simple Demo")
    print("=" * 40)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Test cases
    test_cases = [
        "Transform your body with our 8-week HIIT program! Join thousands who've achieved their dream physique.",
        "Scale your consulting business to 7 figures with proven frameworks and systems.",
        "Capture life's precious moments with artistic vision and technical excellence."
    ]
    
    for i, content in enumerate(test_cases, 1):
        print(f"\nTest {i}: {content[:50]}...")
        result = analyze_business_content(content, api_key)
        print(f"Result: {result}")
    
    print("\nDemo complete!")

if __name__ == "__main__":
    main()
