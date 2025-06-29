-- AutoGuru Universal - PostgreSQL Initialization Script

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Set timezone
SET timezone = 'UTC';

-- Create additional schemas if needed
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS content;
CREATE SCHEMA IF NOT EXISTS social;

-- Grant permissions to autoguru user
GRANT ALL PRIVILEGES ON SCHEMA public TO autoguru;
GRANT ALL PRIVILEGES ON SCHEMA analytics TO autoguru;
GRANT ALL PRIVILEGES ON SCHEMA content TO autoguru;
GRANT ALL PRIVILEGES ON SCHEMA social TO autoguru;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO autoguru;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO autoguru;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO autoguru;

ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT ALL ON TABLES TO autoguru;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT ALL ON SEQUENCES TO autoguru;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT ALL ON FUNCTIONS TO autoguru;

ALTER DEFAULT PRIVILEGES IN SCHEMA content GRANT ALL ON TABLES TO autoguru;
ALTER DEFAULT PRIVILEGES IN SCHEMA content GRANT ALL ON SEQUENCES TO autoguru;
ALTER DEFAULT PRIVILEGES IN SCHEMA content GRANT ALL ON FUNCTIONS TO autoguru;

ALTER DEFAULT PRIVILEGES IN SCHEMA social GRANT ALL ON TABLES TO autoguru;
ALTER DEFAULT PRIVILEGES IN SCHEMA social GRANT ALL ON SEQUENCES TO autoguru;
ALTER DEFAULT PRIVILEGES IN SCHEMA social GRANT ALL ON FUNCTIONS TO autoguru;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_content_created_at ON content.content(created_at);
CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics.events(timestamp);
CREATE INDEX IF NOT EXISTS idx_social_posts_scheduled_at ON social.posts(scheduled_at);

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'AutoGuru Universal database initialized successfully';
END $$; 