#!/usr/bin/env python3
"""
AutoGuru Universal - One-Click Runner

This script automatically sets up and starts AutoGuru Universal.
Just run this script and it will handle everything!
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_status(message: str, status: str = "INFO"):
    """Print status messages"""
    status_symbols = {
        "INFO": "[INFO]",
        "SUCCESS": "[SUCCESS]",
        "WARNING": "[WARNING]", 
        "ERROR": "[ERROR]"
    }
    print(f"{status_symbols.get(status, '[INFO]')} {message}")

def setup_environment():
    """Set up environment file"""
    print_status("Setting up environment...", "INFO")
    
    env_file = Path(".env")
    template_file = Path("env_template.txt")
    
    if not env_file.exists() and template_file.exists():
        try:
            import shutil
            shutil.copy2(template_file, env_file)
            print_status("Created .env file from template", "SUCCESS")
        except Exception as e:
            print_status(f"Failed to create .env: {e}", "ERROR")
            return False
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    print_status("Installing dependencies...", "INFO")
    
    try:
        # Install core dependencies
        subprocess.run([sys.executable, "-m", "pip", "install", 
                       "openai", "anthropic", "fastapi", "uvicorn", "httpx", "pydantic"], 
                      capture_output=True, check=True)
        
        # Install requirements if exists
        if Path("requirements.txt").exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                          capture_output=True, check=True)
        
        print_status("Dependencies installed", "SUCCESS")
        return True
    except Exception as e:
        print_status(f"Installation failed: {e}", "ERROR")
        return False

def check_api_key():
    """Check if API key is available"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print_status("No OpenAI API key found", "WARNING")
        print_status("Please add your API key to .env file:", "INFO")
        print_status("   OPENAI_API_KEY=your_key_here", "INFO")
        print_status("Get your key from: https://platform.openai.com/api-keys", "INFO")
        return False
    else:
        print_status("OpenAI API key found", "SUCCESS")
        return True

def start_application():
    """Start the FastAPI application"""
    print_status("Starting AutoGuru Universal...", "INFO")
    
    try:
        # Start the server
        print_status("Server starting at http://localhost:8000", "INFO")
        print_status("Press Ctrl+C to stop", "INFO")
        print_status("Opening browser...", "INFO")
        
        # Wait a moment for server to start
        time.sleep(2)
        
        # Open browser
        try:
            webbrowser.open("http://localhost:8000/docs")
        except:
            pass
        
        # Start the server
        subprocess.run([sys.executable, "-m", "uvicorn", "backend.main:app", 
                       "--host", "0.0.0.0", "--port", "8000", "--reload"])
        
    except KeyboardInterrupt:
        print_status("Server stopped by user", "INFO")
    except Exception as e:
        print_status(f"Server error: {e}", "ERROR")

def run_demo():
    """Run a quick demo if API key is available"""
    if not check_api_key():
        return
    
    print_status("Running quick demo...", "INFO")
    
    try:
        import openai
        
        # Test content
        test_content = "Transform your body with our 8-week HIIT program! Join thousands who've achieved their dream physique."
        
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a business analyst. Analyze this content and provide: 1) Business niche, 2) Target audience, 3) Brand voice, 4) Viral potential score (0-100). Respond in JSON format."},
                {"role": "user", "content": test_content}
            ],
            max_tokens=300
        )
        
        result = response.choices[0].message.content
        print_status("Demo analysis result:", "SUCCESS")
        print(f"Content: {test_content}")
        print(f"Analysis: {result}")
        
    except Exception as e:
        print_status(f"Demo failed: {e}", "ERROR")

def main():
    """Main function"""
    print_status("AutoGuru Universal - One-Click Runner", "INFO")
    print("=" * 60)
    
    # Step 1: Setup environment
    if not setup_environment():
        return
    
    # Step 2: Install dependencies
    if not install_dependencies():
        return
    
    # Step 3: Check API key
    has_api_key = check_api_key()
    
    # Step 4: Run demo if API key available
    if has_api_key:
        run_demo()
    
    print("\n" + "=" * 60)
    print_status("Ready to start AutoGuru Universal!", "SUCCESS")
    print("=" * 60)
    
    # Step 5: Start application
    start_application()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_status("Setup interrupted by user", "INFO")
    except Exception as e:
        print_status(f"Unexpected error: {e}", "ERROR") 