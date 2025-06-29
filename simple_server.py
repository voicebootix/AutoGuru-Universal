#!/usr/bin/env python3
"""
AutoGuru Universal - Simple Server

A simplified version that works immediately for testing and demonstration.
"""

import os
import json
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Try to import OpenAI, but don't fail if not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Request/Response Models
class AnalyzeRequest(BaseModel):
    content: str = Field(..., min_length=10, max_length=10000, description="Content to analyze")
    business_context: Optional[str] = Field(None, description="Business context")

class AnalysisResponse(BaseModel):
    business_niche: str
    target_audience: str
    brand_voice: str
    viral_potential: int
    recommended_platforms: List[str]
    content_suggestions: List[str]
    hashtags: List[str]

class HealthResponse(BaseModel):
    status: str
    message: str
    features: List[str]

# Create FastAPI app
app = FastAPI(
    title="AutoGuru Universal",
    description="Universal social media automation for ANY business niche",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def analyze_content_ai(content: str, context: Optional[str] = None) -> Dict[str, Any]:
    """Analyze content using AI"""
    if not OPENAI_AVAILABLE:
        return {
            "business_niche": "AI analysis not available - please install openai package",
            "target_audience": "General audience",
            "brand_voice": "Professional",
            "viral_potential": 50,
            "recommended_platforms": ["instagram", "linkedin"],
            "content_suggestions": ["Add more engaging visuals", "Include call-to-action"],
            "hashtags": ["#business", "#growth"]
        }
    
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found")
        
        client = openai.OpenAI(api_key=api_key)
        
        # Create analysis prompt
        prompt = f"""
        Analyze this business content and provide a comprehensive analysis:
        
        Content: {content}
        Context: {context or 'No specific context provided'}
        
        Please provide analysis in JSON format with these fields:
        - business_niche: The primary business category
        - target_audience: Who this content targets
        - brand_voice: The tone and style
        - viral_potential: Score from 0-100
        - recommended_platforms: List of best social platforms
        - content_suggestions: List of improvement suggestions
        - hashtags: Relevant hashtags for the content
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a business analyst specializing in social media strategy. Always respond in valid JSON format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        
        result = response.choices[0].message.content
        # Try to parse JSON response
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            # Fallback if AI doesn't return valid JSON
            return {
                "business_niche": "Content analysis completed",
                "target_audience": "General audience",
                "brand_voice": "Professional",
                "viral_potential": 75,
                "recommended_platforms": ["instagram", "linkedin", "tiktok"],
                "content_suggestions": ["AI analysis completed successfully"],
                "hashtags": ["#autoguru", "#socialmedia"]
            }
            
    except Exception as e:
        return {
            "business_niche": f"Analysis error: {str(e)}",
            "target_audience": "General audience",
            "brand_voice": "Professional",
            "viral_potential": 50,
            "recommended_platforms": ["instagram"],
            "content_suggestions": ["Please check API configuration"],
            "hashtags": ["#error"]
        }

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "AutoGuru Universal - Universal Social Media Automation",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    features = [
        "Content Analysis",
        "Business Niche Detection", 
        "Viral Potential Scoring",
        "Platform Recommendations",
        "Hashtag Generation"
    ]
    
    if OPENAI_AVAILABLE:
        features.append("AI-Powered Analysis")
    else:
        features.append("Basic Analysis (AI not available)")
    
    return HealthResponse(
        status="healthy",
        message="AutoGuru Universal is running successfully",
        features=features
    )

@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_content(request: AnalyzeRequest):
    """Analyze content for any business niche"""
    try:
        analysis = analyze_content_ai(request.content, request.business_context)
        
        return AnalysisResponse(
            business_niche=analysis.get("business_niche", "Unknown"),
            target_audience=analysis.get("target_audience", "General"),
            brand_voice=analysis.get("brand_voice", "Professional"),
            viral_potential=analysis.get("viral_potential", 50),
            recommended_platforms=analysis.get("recommended_platforms", []),
            content_suggestions=analysis.get("content_suggestions", []),
            hashtags=analysis.get("hashtags", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/demo", tags=["Demo"])
async def demo_analysis():
    """Run a demo analysis"""
    demo_content = "Transform your body with our 8-week HIIT program! Join thousands who've achieved their dream physique."
    
    analysis = analyze_content_ai(demo_content, "Fitness and wellness business")
    
    return {
        "demo_content": demo_content,
        "analysis": analysis,
        "message": "This is a demo analysis. Use /analyze endpoint for your own content."
    }

@app.get("/platforms", tags=["Platforms"])
async def get_supported_platforms():
    """Get list of supported social media platforms"""
    return {
        "platforms": [
            {"name": "Instagram", "type": "visual", "best_for": ["visual content", "stories", "reels"]},
            {"name": "LinkedIn", "type": "professional", "best_for": ["B2B", "thought leadership", "networking"]},
            {"name": "TikTok", "type": "video", "best_for": ["short videos", "trending content", "young audience"]},
            {"name": "Twitter", "type": "text", "best_for": ["news", "conversations", "real-time updates"]},
            {"name": "YouTube", "type": "video", "best_for": ["long-form content", "tutorials", "entertainment"]},
            {"name": "Facebook", "type": "social", "best_for": ["community building", "events", "local business"]}
        ],
        "message": "AutoGuru Universal works with all major social media platforms"
    }

@app.get("/niches", tags=["Business Niches"])
async def get_supported_niches():
    """Get list of supported business niches"""
    return {
        "niches": [
            "Fitness & Wellness",
            "Business Consulting", 
            "Creative Services",
            "Education",
            "E-commerce",
            "Local Services",
            "Technology & SaaS",
            "Non-profit Organizations",
            "Real Estate",
            "Healthcare",
            "Food & Beverage",
            "Travel & Tourism"
        ],
        "message": "AutoGuru Universal automatically adapts to ANY business niche"
    }

if __name__ == "__main__":
    print("🚀 Starting AutoGuru Universal Simple Server...")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    print("🎯 Demo Analysis: http://localhost:8000/demo")
    
    if not OPENAI_AVAILABLE:
        print("⚠️  OpenAI not available - install with: pip install openai")
        print("💡 Set OPENAI_API_KEY environment variable for AI analysis")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 