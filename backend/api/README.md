# AutoGuru Universal - Content API Documentation

## Overview

The Content API provides comprehensive endpoints for the complete content processing workflow. All endpoints are designed to work universally for ANY business niche through AI-powered analysis and adaptation.

## Base URL

```
/api/v1/content
```

## Endpoints

### 1. Content Analysis

**POST** `/analyze`

Analyze content to detect business niche, target audience, and optimization opportunities.

**Request Body:**
```json
{
  "content": "Transform your fitness journey with our personalized training programs...",
  "context": {
    "business_name": "FitLife Pro",
    "industry": "fitness"
  },
  "platforms": ["instagram", "tiktok", "linkedin"]
}
```

**Response:**
- Complete content analysis with business niche detection
- Target audience demographics and psychographics
- Brand voice extraction
- Viral potential scores per platform
- Actionable recommendations

### 2. Persona Generation

**POST** `/generate-persona`

Generate a comprehensive business persona based on content analysis.

**Request Body:**
```json
{
  "content_analysis": { /* Previous analysis results */ },
  "business_preferences": {
    "tone": "professional",
    "values": ["innovation", "customer-first"],
    "goals": ["brand awareness", "lead generation"]
  },
  "target_demographics": {
    "age_range": "25-45",
    "location": "urban"
  }
}
```

### 3. Viral Content Generation

**POST** `/generate-viral`

Generate platform-optimized viral content from original content.

**Request Body:**
```json
{
  "original_content": "Your content here...",
  "persona": { /* Business persona data */ },
  "target_platforms": ["instagram", "tiktok"],
  "content_goals": ["engagement", "reach"]
}
```

### 4. Hashtag Optimization

**POST** `/optimize-hashtags`

Generate optimized hashtags for maximum reach and engagement.

**Request Body:**
```json
{
  "content": "Your content here...",
  "platform": "instagram",
  "business_niche": "fitness",
  "max_hashtags": 30,
  "include_trending": true
}
```

### 5. Content Publishing

**POST** `/publish`

Publish content to multiple social media platforms.

**Request Body:**
```json
{
  "platform_content": [
    {
      "platform": "instagram",
      "content_text": "...",
      /* Platform-specific content */
    }
  ],
  "platform_credentials": {
    "instagram": {
      "access_token": "encrypted_token"
    }
  },
  "schedule": {
    "time": "2024-01-15T10:00:00Z",
    "timezone": "UTC"
  },
  "publish_immediately": false
}
```

### 6. Publishing Status

**GET** `/publishing-status/{task_id}`

Get real-time status of a publishing task.

**Response:**
- Task status (pending, processing, completed, failed)
- Progress percentage
- Detailed results for completed tasks
- Error information for failed tasks

### 7. Content Analytics

**GET** `/analytics/{content_id}`

Get comprehensive analytics for published content across all platforms.

**Query Parameters:**
- `platforms`: Filter by specific platforms
- `date_range`: Date range (1d, 7d, 30d, all)

**Response:**
- Platform-specific metrics (views, likes, shares, comments)
- Aggregate performance data
- AI-powered insights
- Optimization recommendations

## Universal Features

All endpoints are designed to work universally for any business niche:

- **Educational businesses** (courses, tutoring, coaching)
- **Business consulting** and coaching
- **Fitness and wellness** professionals
- **Creative professionals** (artists, designers, photographers)
- **E-commerce** and retail businesses
- **Local service** businesses
- **Technology** and SaaS companies
- **Non-profit** organizations

The AI automatically adapts strategies, content, and recommendations based on the detected business type without any hardcoded logic.

## Authentication

All endpoints require API key authentication:
```
Headers: {
  "X-API-Key": "your-api-key"
}
```

## Error Handling

All endpoints return consistent error responses:
```json
{
  "error": "Error type",
  "detail": "Detailed error message"
}
```

HTTP Status Codes:
- `200`: Success
- `400`: Bad Request
- `401`: Unauthorized
- `404`: Not Found
- `500`: Internal Server Error

## Rate Limiting

API rate limits are enforced per API key:
- 60 requests per minute
- 3600 requests per hour

## Background Tasks

Long-running operations (publishing, analysis) are handled asynchronously with task IDs for status tracking.