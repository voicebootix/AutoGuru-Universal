#!/usr/bin/env python3
"""
AutoGuru Universal - Automated Setup Script

This script automatically sets up AutoGuru Universal for local development.
It handles environment setup, dependency installation, and service startup.
"""

import os
import sys
import subprocess
import time
import json
import platform
from pathlib import Path
from typing import Dict, Any, Optional

class AutoGuruSetup:
    """Automated setup for AutoGuru Universal"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.env_file = self.project_root / ".env"
        self.requirements_file = self.project_root / "requirements.txt"
        self.is_windows = platform.system() == "Windows"
        
    def print_status(self, message: str, status: str = "INFO"):
        """Print colored status messages"""
        colors = {
            "INFO": "\033[94m",    # Blue
            "SUCCESS": "\033[92m", # Green
            "WARNING": "\033[93m", # Yellow
            "ERROR": "\033[91m",   # Red
            "RESET": "\033[0m"     # Reset
        }
        print(f"{colors.get(status, colors['INFO'])}[{status}] {message}{colors['RESET']}")
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are installed"""
        self.print_status("Checking prerequisites...", "INFO")
        
        # Check Python
        try:
            python_version = sys.version_info
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
                self.print_status("Python 3.8+ is required", "ERROR")
                return False
            self.print_status(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} found", "SUCCESS")
        except Exception as e:
            self.print_status(f"Python check failed: {e}", "ERROR")
            return False
        
        # Check pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         capture_output=True, check=True)
            self.print_status("✅ pip is available", "SUCCESS")
        except Exception as e:
            self.print_status(f"pip check failed: {e}", "ERROR")
            return False
        
        # Check Docker (optional but recommended)
        try:
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.print_status("✅ Docker found", "SUCCESS")
                self.docker_available = True
            else:
                self.print_status("⚠️  Docker not found (will use alternative setup)", "WARNING")
                self.docker_available = False
        except FileNotFoundError:
            self.print_status("⚠️  Docker not found (will use alternative setup)", "WARNING")
            self.docker_available = False
        
        return True
    
    def setup_environment(self) -> bool:
        """Set up environment variables"""
        self.print_status("Setting up environment...", "INFO")
        
        # Check if .env exists
        if self.env_file.exists():
            self.print_status("✅ .env file already exists", "SUCCESS")
            return True
        
        # Create .env from template
        template_file = self.project_root / "env_template.txt"
        if template_file.exists():
            try:
                import shutil
                shutil.copy2(template_file, self.env_file)
                self.print_status("✅ Created .env file from template", "SUCCESS")
                return True
            except Exception as e:
                self.print_status(f"Failed to create .env: {e}", "ERROR")
                return False
        else:
            self.print_status("❌ env_template.txt not found", "ERROR")
            return False
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        self.print_status("Installing Python dependencies...", "INFO")
        
        if not self.requirements_file.exists():
            self.print_status("❌ requirements.txt not found", "ERROR")
            return False
        
        try:
            # Upgrade pip first
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         capture_output=True, check=True)
            
            # Install requirements
            result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                self.print_status("✅ Dependencies installed successfully", "SUCCESS")
                return True
            else:
                self.print_status(f"❌ Dependency installation failed: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            self.print_status(f"❌ Installation error: {e}", "ERROR")
            return False
    
    def get_api_keys(self) -> Dict[str, str]:
        """Get API keys from user or environment"""
        self.print_status("Setting up API keys...", "INFO")
        
        api_keys = {}
        
        # Check environment variables first
        openai_key = os.getenv('OPENAI_API_KEY')
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        
        if openai_key:
            api_keys['OPENAI_API_KEY'] = openai_key
            self.print_status("✅ OpenAI API key found in environment", "SUCCESS")
        else:
            self.print_status("⚠️  OpenAI API key not found", "WARNING")
            self.print_status("💡 Get your key from: https://platform.openai.com/api-keys", "INFO")
        
        if anthropic_key:
            api_keys['ANTHROPIC_API_KEY'] = anthropic_key
            self.print_status("✅ Anthropic API key found in environment", "SUCCESS")
        else:
            self.print_status("⚠️  Anthropic API key not found (optional)", "WARNING")
        
        return api_keys
    
    def update_env_file(self, api_keys: Dict[str, str]) -> bool:
        """Update .env file with API keys"""
        if not self.env_file.exists():
            return False
        
        try:
            # Read current .env content
            with open(self.env_file, 'r') as f:
                content = f.read()
            
            # Update API keys
            for key, value in api_keys.items():
                if value:  # Only update if we have a value
                    # Replace placeholder or add new line
                    placeholder = f"{key}=your_{key.lower()}_here"
                    if placeholder in content:
                        content = content.replace(placeholder, f"{key}={value}")
                    else:
                        # Add new line if not found
                        content += f"\n{key}={value}\n"
            
            # Write updated content
            with open(self.env_file, 'w') as f:
                f.write(content)
            
            self.print_status("✅ Updated .env file with API keys", "SUCCESS")
            return True
            
        except Exception as e:
            self.print_status(f"❌ Failed to update .env: {e}", "ERROR")
            return False
    
    def test_imports(self) -> bool:
        """Test if all modules can be imported"""
        self.print_status("Testing module imports...", "INFO")
        
        try:
            # Add backend to path
            backend_path = self.project_root / "backend"
            if backend_path not in sys.path:
                sys.path.insert(0, str(backend_path))
            
            # Test imports
            from backend.core.content_analyzer import UniversalContentAnalyzer
            from backend.models.content_models import BusinessNiche, Platform
            from backend.config.settings import get_settings
            
            self.print_status("✅ All core modules imported successfully", "SUCCESS")
            return True
            
        except ImportError as e:
            self.print_status(f"❌ Import error: {e}", "ERROR")
            return False
        except Exception as e:
            self.print_status(f"❌ Import test error: {e}", "ERROR")
            return False
    
    def run_simple_test(self) -> bool:
        """Run the simple test to verify functionality"""
        self.print_status("Running simple functionality test...", "INFO")
        
        try:
            result = subprocess.run([sys.executable, "simple_test.py"], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                self.print_status("✅ Simple test completed successfully", "SUCCESS")
                return True
            else:
                self.print_status(f"⚠️  Simple test had issues: {result.stderr}", "WARNING")
                return False
                
        except subprocess.TimeoutExpired:
            self.print_status("⚠️  Simple test timed out", "WARNING")
            return False
        except Exception as e:
            self.print_status(f"❌ Simple test error: {e}", "ERROR")
            return False
    
    def start_services(self) -> bool:
        """Start Docker services if available"""
        if not self.docker_available:
            self.print_status("Skipping Docker services (Docker not available)", "WARNING")
            return True
        
        self.print_status("Starting Docker services...", "INFO")
        
        try:
            # Start services
            result = subprocess.run(["docker-compose", "up", "-d"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                self.print_status("✅ Docker services started", "SUCCESS")
                
                # Wait for services to be ready
                self.print_status("Waiting for services to be ready...", "INFO")
                time.sleep(15)
                
                return True
            else:
                self.print_status(f"⚠️  Docker services failed: {result.stderr}", "WARNING")
                return False
                
        except Exception as e:
            self.print_status(f"❌ Docker error: {e}", "ERROR")
            return False
    
    def create_startup_scripts(self) -> bool:
        """Create convenient startup scripts"""
        self.print_status("Creating startup scripts...", "INFO")
        
        try:
            # Create Windows batch file
            if self.is_windows:
                batch_content = """@echo off
echo Starting AutoGuru Universal...
docker-compose up -d
timeout /t 10
pip install -r requirements.txt
python -c "import asyncio; from backend.database.connection import init_database; asyncio.run(init_database())"
start /b uvicorn backend.main:app --host 0.0.0.0 --port 8000
echo Open http://localhost:8000/docs to see your API!
pause
"""
                with open(self.project_root / "start.bat", "w") as f:
                    f.write(batch_content)
                self.print_status("✅ Created start.bat", "SUCCESS")
            
            # Create Python startup script
            python_startup = """#!/usr/bin/env python3
import subprocess
import sys
import time

def start_autoguru():
    print("🚀 Starting AutoGuru Universal...")
    
    # Start services
    subprocess.run(["docker-compose", "up", "-d"])
    time.sleep(10)
    
    # Initialize database
    subprocess.run([sys.executable, "-c", 
                   "import asyncio; from backend.database.connection import init_database; asyncio.run(init_database())"])
    
    # Start FastAPI server
    subprocess.run([sys.executable, "-m", "uvicorn", "backend.main:app", 
                   "--host", "0.0.0.0", "--port", "8000"])

if __name__ == "__main__":
    start_autoguru()
"""
            with open(self.project_root / "start_python.py", "w") as f:
                f.write(python_startup)
            self.print_status("✅ Created start_python.py", "SUCCESS")
            
            return True
            
        except Exception as e:
            self.print_status(f"❌ Failed to create startup scripts: {e}", "ERROR")
            return False
    
    def show_next_steps(self):
        """Show next steps to the user"""
        self.print_status("🎉 AutoGuru Universal setup complete!", "SUCCESS")
        print("\n" + "="*60)
        print("🚀 NEXT STEPS:")
        print("="*60)
        
        print("\n1. 📝 Add your API keys to .env file:")
        print("   OPENAI_API_KEY=your_openai_key_here")
        print("   ANTHROPIC_API_KEY=your_anthropic_key_here")
        
        print("\n2. 🚀 Start the application:")
        if self.is_windows:
            print("   Double-click: start.bat")
            print("   Or run: python start_python.py")
        else:
            print("   Run: python start_python.py")
            print("   Or run: ./run_local.sh")
        
        print("\n3. 🌐 Access the application:")
        print("   API: http://localhost:8000")
        print("   API Docs: http://localhost:8000/docs")
        print("   Frontend: Open frontend/index.html in browser")
        
        print("\n4. 🧪 Test the system:")
        print("   Run: python simple_test.py")
        print("   Run: python test_integration.py")
        
        print("\n5. 📚 Documentation:")
        print("   Read: ENVIRONMENT_SETUP.md")
        print("   Read: DOCKER_SETUP.md")
        
        print("\n" + "="*60)
        print("🎯 AutoGuru Universal is ready for any business niche!")
        print("   - Fitness & Wellness")
        print("   - Business Consulting")
        print("   - Creative Services")
        print("   - Education")
        print("   - E-commerce")
        print("   - And more!")
        print("="*60)
    
    def run_full_setup(self) -> bool:
        """Run the complete automated setup"""
        self.print_status("🚀 Starting AutoGuru Universal Automated Setup", "INFO")
        print("="*60)
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Step 2: Setup environment
        if not self.setup_environment():
            return False
        
        # Step 3: Install dependencies
        if not self.install_dependencies():
            return False
        
        # Step 4: Get API keys
        api_keys = self.get_api_keys()
        
        # Step 5: Update .env file
        if api_keys:
            self.update_env_file(api_keys)
        
        # Step 6: Test imports
        if not self.test_imports():
            return False
        
        # Step 7: Run simple test
        self.run_simple_test()
        
        # Step 8: Start services (if Docker available)
        self.start_services()
        
        # Step 9: Create startup scripts
        self.create_startup_scripts()
        
        # Step 10: Show next steps
        self.show_next_steps()
        
        return True


def main():
    """Main setup function"""
    setup = AutoGuruSetup()
    
    try:
        success = setup.run_full_setup()
        if success:
            print("\n✅ Setup completed successfully!")
        else:
            print("\n❌ Setup failed. Please check the error messages above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 