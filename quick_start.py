#!/usr/bin/env python3
"""
AutoGuru Universal - Quick Start Script

This script provides a quick way to test AutoGuru Universal without complex setup.
It focuses on the AI analysis functionality which is the core feature.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_status(message: str, status: str = "INFO"):
    """Print colored status messages"""
    colors = {
        "INFO": "\033[94m",    # Blue
        "SUCCESS": "\033[92m", # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m",   # Red
        "RESET": "\033[0m"     # Reset
    }
    print(f"{colors.get(status, colors['INFO'])}[{status}] {message}{colors['RESET']}")

def check_python():
    """Check Python version"""
    print_status("Checking Python...", "INFO")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_status(f"✅ Python {version.major}.{version.minor}.{version.micro} found", "SUCCESS")
        return True
    else:
        print_status("❌ Python 3.8+ is required", "ERROR")
        return False

def install_dependencies():
    """Install required dependencies"""
    print_status("Installing dependencies...", "INFO")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "openai", "anthropic", "fastapi", "uvicorn", "httpx"], 
                      capture_output=True, check=True)
        print_status("✅ Core dependencies installed", "SUCCESS")
        return True
    except Exception as e:
        print_status(f"❌ Installation failed: {e}", "ERROR")
        return False

def test_ai_analysis():
    """Test AI analysis functionality"""
    print_status("Testing AI analysis...", "INFO")
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print_status("⚠️  No OpenAI API key found", "WARNING")
        print_status("💡 Set your API key:", "INFO")
        print_status("   export OPENAI_API_KEY=your_key_here", "INFO")
        print_status("   Or add it to your .env file", "INFO")
        return False
    
    # Simple AI test
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # Test with a simple prompt
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a business analyst. Analyze this content and identify the business niche."},
                {"role": "user", "content": "Transform your body with our 8-week HIIT program! Join thousands who've achieved their dream physique."}
            ],
            max_tokens=100
        )
        
        result = response.choices[0].message.content
        print_status("✅ AI analysis test successful", "SUCCESS")
        print_status(f"   Result: {result}", "INFO")
        return True
        
    except Exception as e:
        print_status(f"❌ AI test failed: {e}", "ERROR")
        return False

def create_simple_demo():
    """Create a simple demo script"""
    demo_script = '''#!/usr/bin/env python3
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
    print("🧪 AutoGuru Universal - Simple Demo")
    print("=" * 40)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # Test cases
    test_cases = [
        "Transform your body with our 8-week HIIT program! Join thousands who've achieved their dream physique.",
        "Scale your consulting business to 7 figures with proven frameworks and systems.",
        "Capture life's precious moments with artistic vision and technical excellence."
    ]
    
    for i, content in enumerate(test_cases, 1):
        print(f"\\n📊 Test {i}: {content[:50]}...")
        result = analyze_business_content(content, api_key)
        print(f"Result: {result}")
    
    print("\\n🎉 Demo complete!")

if __name__ == "__main__":
    main()
'''
    
    with open("simple_demo.py", "w") as f:
        f.write(demo_script)
    
    print_status("✅ Created simple_demo.py", "SUCCESS")

def show_next_steps():
    """Show next steps"""
    print_status("🎉 Quick setup complete!", "SUCCESS")
    print("\n" + "="*50)
    print("🚀 NEXT STEPS:")
    print("="*50)
    
    print("\n1. 📝 Set your OpenAI API key:")
    print("   export OPENAI_API_KEY=your_key_here")
    print("   Or add to .env file")
    
    print("\n2. 🧪 Test the simple demo:")
    print("   python simple_demo.py")
    
    print("\n3. 🚀 For full setup (with database):")
    print("   python auto_setup.py")
    
    print("\n4. 🌐 Get your API key from:")
    print("   https://platform.openai.com/api-keys")
    
    print("\n" + "="*50)

def main():
    """Main quick start function"""
    print_status("🚀 AutoGuru Universal - Quick Start", "INFO")
    print("="*50)
    
    # Step 1: Check Python
    if not check_python():
        return False
    
    # Step 2: Install dependencies
    if not install_dependencies():
        return False
    
    # Step 3: Test AI (if API key available)
    test_ai_analysis()
    
    # Step 4: Create demo
    create_simple_demo()
    
    # Step 5: Show next steps
    show_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Quick start completed!")
        else:
            print("\n❌ Quick start failed.")
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}") 