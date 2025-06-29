# 🚀 AutoGuru Universal - Render Deployment Guide

## 📋 Overview

This guide will walk you through deploying AutoGuru Universal to Render, a cloud platform that provides hosting, databases, and background workers.

## 🎯 What We're Deploying

AutoGuru Universal is a comprehensive social media automation platform that works for ANY business niche:
- **Fitness & Wellness** 🏃‍♀️
- **Business Consulting** 💼
- **Creative Services** 🎨
- **Education** 📚
- **E-commerce** 🛒
- **Local Services** 🏪
- **Technology & SaaS** 💻
- **And more!**

## 🛠️ Prerequisites

1. **GitHub Account** - Your code must be in a GitHub repository
2. **Render Account** - Sign up at [render.com](https://render.com)
3. **OpenAI API Key** - Get from [platform.openai.com](https://platform.openai.com/api-keys)
4. **Anthropic API Key** (Optional) - Get from [console.anthropic.com](https://console.anthropic.com)

## 📦 Repository Structure

After cleanup, your repository should have this structure:
```
AutoGuru-Universal/
├── backend/                 # Core application code
│   ├── api/                # API routes
│   ├── config/             # Configuration files
│   ├── core/               # Core business logic
│   ├── database/           # Database models and connection
│   ├── models/             # Data models
│   ├── platforms/          # Social media platform integrations
│   ├── services/           # Business services
│   ├── tasks/              # Background tasks
│   ├── utils/              # Utilities
│   └── main.py             # FastAPI application
├── frontend/               # Frontend files
├── tests/                  # Test files
├── render.yaml             # Render deployment configuration
├── Procfile                # Process definitions
├── requirements.txt        # Python dependencies
├── env_template.txt        # Environment variables template
└── README.md               # Project documentation
```

## 🚀 Step 1: Prepare Your Repository

### 1.1 Push to GitHub
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 1.2 Verify Repository Structure
Ensure these files are in your repository root:
- ✅ `render.yaml`
- ✅ `Procfile`
- ✅ `requirements.txt`
- ✅ `backend/main.py`

## 🌐 Step 2: Deploy to Render

### 2.1 Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up with your GitHub account
3. Authorize Render to access your repositories

### 2.2 Deploy Using Blueprint
1. In Render dashboard, click **"New +"**
2. Select **"Blueprint"**
3. Connect your GitHub repository
4. Render will automatically detect `render.yaml`
5. Click **"Apply"**

### 2.3 Manual Deployment (Alternative)
If blueprint doesn't work:

1. **Create Database:**
   - Click **"New +"** → **"PostgreSQL"**
   - Name: `autoguru-db`
   - Plan: `Starter`
   - Click **"Create Database"**

2. **Create Web Service:**
   - Click **"New +"** → **"Web Service"**
   - Connect your GitHub repository
   - Name: `autoguru-universal`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

## ⚙️ Step 3: Configure Environment Variables

### 3.1 Required Variables
In your Render web service dashboard, go to **"Environment"** and add:

```bash
# Database (auto-configured by Render)
DATABASE_URL=postgresql://...  # Auto-filled by Render

# Security
SECRET_KEY=your-secret-key-here  # Generate a secure random string
ENVIRONMENT=production

# AI Services
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here  # Optional

# Application Settings
LOG_LEVEL=INFO
CORS_ORIGINS=*
```

### 3.2 Generate Secret Key
```bash
# In your terminal
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3.3 Optional Variables
```bash
# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Content Limits
MAX_CONTENT_LENGTH=10000
MAX_ANALYSIS_TOKENS=500

# Database Pool
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
```

## 🔍 Step 4: Monitor Deployment

### 4.1 Check Build Logs
1. Go to your web service in Render dashboard
2. Click **"Logs"** tab
3. Monitor the build process
4. Look for any errors

### 4.2 Common Issues & Solutions

**Issue: Import errors**
```
Solution: Ensure all dependencies are in requirements.txt
```

**Issue: Database connection failed**
```
Solution: Check DATABASE_URL is properly set
```

**Issue: Port binding error**
```
Solution: Ensure using $PORT environment variable
```

**Issue: Module not found**
```
Solution: Check Python path and imports in main.py
```

## ✅ Step 5: Verify Deployment

### 5.1 Health Check
Visit your Render URL + `/health`:
```
https://your-app-name.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "environment": "production",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "features": [
    "Content Analysis",
    "Business Niche Detection",
    "Viral Potential Scoring",
    "Platform Recommendations",
    "Hashtag Generation",
    "Universal Business Support"
  ]
}
```

### 5.2 API Documentation
Visit your Render URL + `/docs`:
```
https://your-app-name.onrender.com/docs
```

### 5.3 Test API Endpoints
```bash
# Test demo endpoint
curl https://your-app-name.onrender.com/demo

# Test health endpoint
curl https://your-app-name.onrender.com/health
```

## 🔧 Step 6: Configure Custom Domain (Optional)

### 6.1 Add Custom Domain
1. In Render dashboard, go to your web service
2. Click **"Settings"** → **"Custom Domains"**
3. Add your domain (e.g., `api.autoguru.com`)
4. Update DNS records as instructed

### 6.2 Update CORS Settings
If using custom domain, update `CORS_ORIGINS`:
```bash
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

## 📊 Step 7: Monitor & Scale

### 7.1 Monitor Performance
- **Logs**: Check application logs regularly
- **Metrics**: Monitor response times and errors
- **Database**: Watch connection pool usage

### 7.2 Scaling Options
- **Auto-scaling**: Enable in Render dashboard
- **Manual scaling**: Increase instance count
- **Database scaling**: Upgrade PostgreSQL plan

## 🧪 Step 8: Test Full Functionality

### 8.1 Test Content Analysis
```bash
curl -X POST "https://your-app-name.onrender.com/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "content": "Transform your body with our 8-week HIIT program!",
    "context": {"business_type": "fitness"}
  }'
```

### 8.2 Test Different Business Types
- **Fitness**: "Transform your body with our 8-week HIIT program!"
- **Business**: "Scale your consulting business to 7 figures"
- **Creative**: "Capture life's precious moments with artistic vision"
- **Education**: "Master new skills with our online courses"

## 🔒 Step 9: Security Considerations

### 9.1 Environment Variables
- ✅ Never commit API keys to repository
- ✅ Use Render's environment variable system
- ✅ Rotate keys regularly

### 9.2 API Security
- ✅ Implement proper authentication
- ✅ Use HTTPS (automatic with Render)
- ✅ Rate limiting enabled

### 9.3 Database Security
- ✅ Database credentials managed by Render
- ✅ Connection pooling configured
- ✅ Regular backups enabled

## 📈 Step 10: Production Optimization

### 10.1 Performance
- **Caching**: Implement Redis caching
- **CDN**: Use Cloudflare for static assets
- **Database**: Optimize queries and indexes

### 10.2 Monitoring
- **Uptime**: Set up uptime monitoring
- **Alerts**: Configure error notifications
- **Analytics**: Track API usage

## 🆘 Troubleshooting

### Common Issues

**Deployment Fails**
1. Check build logs for errors
2. Verify requirements.txt is complete
3. Ensure main.py has correct imports

**Database Connection Issues**
1. Verify DATABASE_URL is set
2. Check database is running
3. Test connection manually

**API Not Responding**
1. Check health endpoint
2. Verify environment variables
3. Review application logs

**Import Errors**
1. Check Python version compatibility
2. Verify all dependencies installed
3. Review import statements

### Getting Help
1. **Render Documentation**: [docs.render.com](https://docs.render.com)
2. **Render Support**: Available in dashboard
3. **Application Logs**: Check in Render dashboard

## 🎉 Success!

Once deployed, AutoGuru Universal will be available at:
```
https://your-app-name.onrender.com
```

### Available Endpoints:
- **Health Check**: `/health`
- **API Documentation**: `/docs`
- **Demo Analysis**: `/demo`
- **Content Analysis**: `/api/v1/analyze`
- **Persona Generation**: `/api/v1/generate-persona`
- **Viral Content**: `/api/v1/create-viral-content`

### Next Steps:
1. **Test all endpoints** with different business types
2. **Monitor performance** and logs
3. **Scale as needed** based on usage
4. **Implement authentication** for production use
5. **Add more features** as needed

---

**AutoGuru Universal** - Universal social media automation deployed and ready! 🚀 