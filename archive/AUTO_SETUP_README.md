# AutoGuru Universal - Automated Setup

Welcome to AutoGuru Universal! This comprehensive social media automation platform works for ANY business niche automatically using AI-driven strategies.

## 🚀 Quick Start Options

### Option 1: One-Click Runner (Recommended)
```bash
python run_autoguru.py
```
This script will:
- ✅ Set up environment automatically
- ✅ Install all dependencies
- ✅ Check for API keys
- ✅ Run a quick demo (if API key available)
- ✅ Start the application
- ✅ Open your browser to the API docs

### Option 2: Quick Setup
```bash
python quick_start_windows.py
```
This script will:
- ✅ Install core dependencies
- ✅ Create demo scripts
- ✅ Create startup batch file
- ✅ Show you next steps

### Option 3: Full Automated Setup
```bash
python auto_setup.py
```
This script will:
- ✅ Check all prerequisites
- ✅ Set up environment
- ✅ Install dependencies
- ✅ Test imports
- ✅ Start Docker services (if available)
- ✅ Create startup scripts

## 📋 Prerequisites

- **Python 3.8+** (✅ Already installed)
- **OpenAI API Key** (Get from https://platform.openai.com/api-keys)
- **Docker** (Optional, for full database setup)

## 🔑 Setting Up API Keys

### Method 1: Environment Variable
```bash
set OPENAI_API_KEY=your_key_here
```

### Method 2: .env File
Edit the `.env` file and add:
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here  # Optional
```

## 🎯 What AutoGuru Universal Does

AutoGuru Universal automatically analyzes your business content and creates viral social media strategies for:

- **Fitness & Wellness** 🏃‍♀️
- **Business Consulting** 💼
- **Creative Services** 🎨
- **Education** 📚
- **E-commerce** 🛒
- **Local Services** 🏪
- **Technology & SaaS** 💻
- **Non-profit Organizations** 🤝

## 🚀 Starting the Application

### Windows Users:
1. Double-click `start_autoguru.bat`
2. Or run: `python run_autoguru.py`

### All Platforms:
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## 🌐 Accessing the Application

- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Frontend**: Open `frontend/index.html` in your browser

## 🧪 Testing the System

### Quick Demo:
```bash
python simple_demo.py
```

### Integration Test:
```bash
python test_integration.py
```

### Simple Test:
```bash
python simple_test.py
```

## 📁 Generated Files

After running the setup scripts, you'll have:

- `simple_demo.py` - Quick AI analysis demo
- `start_autoguru.bat` - Windows startup script
- `start_python.py` - Python startup script
- `.env` - Environment configuration

## 🔧 Troubleshooting

### No API Key Found:
1. Get your key from https://platform.openai.com/api-keys
2. Add it to `.env` file or set environment variable
3. Restart the application

### Import Errors:
1. Run: `pip install -r requirements.txt`
2. Or run: `python quick_start_windows.py`

### Port Already in Use:
1. Stop other applications using port 8000
2. Or change port in the startup command

### Docker Issues:
- Docker is optional for basic functionality
- Use `python run_autoguru.py` for non-Docker setup

## 📚 Documentation

- `ENVIRONMENT_SETUP.md` - Detailed environment setup
- `DOCKER_SETUP.md` - Docker configuration guide
- `backend/README.md` - Backend architecture
- `platforms/README.md` - Social media platforms

## 🎉 Success!

Once running, AutoGuru Universal will:

1. **Analyze** your business content automatically
2. **Identify** your niche and target audience
3. **Generate** viral social media strategies
4. **Create** platform-specific content
5. **Optimize** for maximum engagement

The AI automatically adapts to ANY business type - from fitness coaches to business consultants to artists!

## 🆘 Need Help?

1. Check the troubleshooting section above
2. Review the documentation files
3. Run the test scripts to verify functionality
4. Ensure your API key is properly configured

---

**AutoGuru Universal** - Universal social media automation for any business niche! 🚀 