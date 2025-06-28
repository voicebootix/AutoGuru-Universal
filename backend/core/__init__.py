"""
Core modules for AutoGuru Universal.

This package contains the main AI-powered engines for content analysis
and viral content generation that work universally across all business niches.
"""

from backend.core.content_analyzer import (
    UniversalContentAnalyzer,
    ContentAnalysisResult,
    TargetAudience,
    BrandVoice,
    BusinessNiche,
    Platform
)

from backend.core.viral_engine import (
    ViralContentEngine,
    BusinessPersona,
    Content,
    PLATFORM_CONSTRAINTS,
    VIRAL_PATTERNS
)

__all__ = [
    # Content Analyzer
    "UniversalContentAnalyzer",
    "ContentAnalysisResult",
    "TargetAudience",
    "BrandVoice",
    "BusinessNiche",
    "Platform",
    
    # Viral Engine
    "ViralContentEngine",
    "BusinessPersona",
    "Content",
    "PLATFORM_CONSTRAINTS",
    "VIRAL_PATTERNS"
]