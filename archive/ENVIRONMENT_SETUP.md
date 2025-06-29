# AutoGuru Universal - Environment Setup Guide

This guide will help you set up your local development environment for AutoGuru Universal.

## Quick Start

1. **Run the setup script:**
   ```bash
   python setup_env.py
   ```

2. **Edit the .env file** with your actual API keys and credentials

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up databases and start the application**

## Manual Setup

If you prefer to set up manually:

### 1. Create Environment File

Copy the template and create your `.env` file:
```bash
cp env_template.txt .env
```

### 2. Required Environment Variables

Edit your `.env` file with the following variables:

#### Database Configuration
```env
DATABASE_URL=postgresql://autoguru:password@localhost:5432/autoguru_universal
```

#### AI Services
```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

#### Redis/Celery Configuration
```env
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

#### FastAPI Configuration
```env
SECRET_KEY=your-secret-key-here
ENVIRONMENT=development
DEBUG=true
```

#### Logging Configuration
```env
LOG_LEVEL=INFO
```

## Database Setup

### PostgreSQL Installation

#### Windows
1. Download from: https://www.postgresql.org/download/windows/
2. Install with default settings
3. Create database and user:
   ```sql
   CREATE DATABASE autoguru_universal;
   CREATE USER autoguru WITH PASSWORD 'password';
   GRANT ALL PRIVILEGES ON DATABASE autoguru_universal TO autoguru;
   ```

#### macOS
```bash
brew install postgresql
brew services start postgresql
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### Create Database
```bash
sudo -u postgres psql
CREATE DATABASE autoguru_universal;
CREATE USER autoguru WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE autoguru_universal TO autoguru;
\q
```

## Redis Setup

### Windows
1. Download from: https://redis.io/download
2. Install and start Redis server

### macOS
```bash
brew install redis
brew services start redis
```

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

## API Keys Setup

### OpenAI API Key
1. Go to: https://platform.openai.com/api-keys
2. Create a new API key
3. Add to your `.env` file

### Anthropic API Key
1. Go to: https://console.anthropic.com/
2. Create a new API key
3. Add to your `.env` file

## Optional Configuration

### Social Media Platform APIs
Uncomment and configure these in your `.env` file as needed:
```env
# INSTAGRAM_ACCESS_TOKEN=your_instagram_token_here
# FACEBOOK_ACCESS_TOKEN=your_facebook_token_here
# TWITTER_API_KEY=your_twitter_api_key_here
# LINKEDIN_CLIENT_ID=your_linkedin_client_id_here
# LINKEDIN_CLIENT_SECRET=your_linkedin_client_secret_here
```

### Email Configuration
For email notifications:
```env
# SMTP_HOST=smtp.gmail.com
# SMTP_PORT=587
# SMTP_USERNAME=your_email@gmail.com
# SMTP_PASSWORD=your_app_password_here
```

### File Storage (AWS S3)
For cloud file storage:
```env
# AWS_ACCESS_KEY_ID=your_aws_access_key_here
# AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
# AWS_S3_BUCKET=autoguru-universal-assets
# AWS_REGION=us-east-1
```

## Security Notes

1. **Never commit your `.env` file** - it's already in `.gitignore`
2. **Use strong passwords** for database and API keys
3. **Rotate API keys regularly**
4. **Use environment-specific configurations** for production

## Troubleshooting

### Database Connection Issues
- Verify PostgreSQL is running
- Check database credentials in `.env`
- Ensure database and user exist

### Redis Connection Issues
- Verify Redis server is running
- Check Redis URL in `.env`
- Test connection: `redis-cli ping`

### API Key Issues
- Verify API keys are valid
- Check API key permissions
- Ensure no extra spaces in `.env` file

### Permission Issues
- Ensure proper file permissions
- Check database user permissions
- Verify Redis access rights

## Next Steps

After setting up your environment:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run database migrations:**
   ```bash
   python backend/main.py --migrate
   ```

3. **Start the application:**
   ```bash
   python backend/main.py
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the application logs
3. Check the project documentation
4. Create an issue in the project repository 