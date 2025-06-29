#!/usr/bin/env python3
"""
AutoGuru Universal - Environment Setup Script

This script helps you set up your local development environment by creating
a .env file from the template and providing setup instructions.
"""

import os
import shutil
import sys
from pathlib import Path


def setup_environment():
    """Set up the local development environment."""
    print("🚀 AutoGuru Universal - Environment Setup")
    print("=" * 50)
    
    # Check if .env already exists
    env_file = Path(".env")
    template_file = Path("env_template.txt")
    
    if env_file.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").lower().strip()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    # Copy template to .env
    if template_file.exists():
        try:
            shutil.copy2(template_file, env_file)
            print("✅ Created .env file from template")
        except Exception as e:
            print(f"❌ Error creating .env file: {e}")
            return
    else:
        print("❌ env_template.txt not found!")
        return
    
    print("\n📋 Next Steps:")
    print("1. Edit the .env file with your actual API keys and credentials")
    print("2. Set up PostgreSQL database:")
    print("   - Install PostgreSQL")
    print("   - Create database: autoguru_universal")
    print("   - Create user: autoguru with password: password")
    print("3. Set up Redis:")
    print("   - Install Redis")
    print("   - Start Redis server")
    print("4. Install Python dependencies: pip install -r requirements.txt")
    print("5. Run the application: python backend/main.py")
    
    print("\n🔑 Required API Keys:")
    print("- OpenAI API Key: https://platform.openai.com/api-keys")
    print("- Anthropic API Key: https://console.anthropic.com/")
    
    print("\n📚 For more information, check the README.md file")


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n🔍 Checking dependencies...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print("✅ Python version OK")
    
    # Check if requirements.txt exists
    if Path("requirements.txt").exists():
        print("✅ requirements.txt found")
    else:
        print("❌ requirements.txt not found")
        return False
    
    return True


def main():
    """Main setup function."""
    if not check_dependencies():
        print("❌ Setup failed due to missing dependencies")
        sys.exit(1)
    
    setup_environment()
    print("\n🎉 Setup complete! Happy coding!")


if __name__ == "__main__":
    main() 