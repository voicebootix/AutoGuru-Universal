#!/usr/bin/env python3
"""
AutoGuru Universal - Startup Script

This script starts both the frontend and backend servers for development.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def run_command(command, cwd=None, shell=True):
    """Run a command and return the process"""
    print(f"Starting: {command}")
    return subprocess.Popen(command, cwd=cwd, shell=shell)

def main():
    print("🚀 Starting AutoGuru Universal...")
    print("=" * 50)
    
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Start backend server
    backend_cmd = "python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
    backend_process = run_command(backend_cmd, cwd=project_root / "backend")
    
    # Wait a moment for backend to start
    time.sleep(3)
    
    # Start frontend server
    frontend_cmd = "npm run dev"
    frontend_process = run_command(frontend_cmd, cwd=project_root / "frontend")
    
    print("\n✅ AutoGuru Universal is starting up!")
    print("📱 Frontend: http://localhost:3000")
    print("🔧 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop all servers")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping AutoGuru Universal...")
        backend_process.terminate()
        frontend_process.terminate()
        print("✅ Servers stopped")

if __name__ == "__main__":
    main() 