"""
Universal Content Creator Base Class

This module provides the foundation for all content creation modules in AutoGuru Universal.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
import base64
import logging
from io import BytesIO
import os
import json

# Image processing imports
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
except ImportError:
    Image = ImageDraw = ImageFont = ImageFilter = ImageEnhance = None

# Video processing imports
try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
except ImportError:
    VideoFileClip = AudioFileClip = CompositeVideoClip = None

# Configure logging
logger = logging.getLogger(__name__)

class ContentType(Enum):
    """Types of content that can be created"""
    IMAGE = "image"
    VIDEO = "video"
    ADVERTISEMENT = "advertisement"
    COPY = "copy"
    BRAND_ASSET = "brand_asset"
    SOCIAL_POST = "social_post"

class CreativeStyle(Enum):
    """Creative style preferences"""
    MINIMAL = "minimal"
    BOLD = "bold"
    PROFESSIONAL = "professional"
    PLAYFUL = "playful"
    LUXURY = "luxury"
    MODERN = "modern"
    VINTAGE = "vintage"
    ARTISTIC = "artistic"

class QualityLevel(Enum):
    """Quality levels for content creation"""
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    PREMIUM = "premium"
    ULTRA = "ultra"

@dataclass
class CreativeRequest:
    """Request for creative content generation"""
    request_id: str
    client_id: str
    content_type: ContentType
    business_niche: str
    target_audience: Dict[str, Any]
    creative_brief: str
    style_preferences: List[CreativeStyle]
    quality_level: QualityLevel
    platform_requirements: Dict[str, Any]
    brand_guidelines: Dict[str, Any]
    deadline: Optional[datetime] = None
    budget_tier: str = "standard"
    revision_limit: int = 3
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class CreativeAsset:
    """Created creative asset with metadata"""
    asset_id: str
    request_id: str
    content_type: ContentType
    file_path: str
    file_format: str
    dimensions: Dict[str, int]
    file_size: int
    quality_score: float
    brand_compliance_score: float
    platform_optimized_versions: Dict[str, str]
    metadata: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class ContentCreationError(Exception):
    """Custom exception for content creation errors"""
    pass

class UniversalContentCreator(ABC):
    """Base class for all content creation modules"""
    
    def __init__(self, client_id: str, creator_type: str):
        self.client_id = client_id
        self.creator_type = creator_type
        self.ai_service = self._initialize_ai_service()
        self.brand_analyzer = self._initialize_brand_analyzer()
        self.quality_assessor = self._initialize_quality_assessor()
        
    def _initialize_ai_service(self):
        """Initialize AI creative service"""
        from backend.services.ai_creative_service import AICreativeService
        return AICreativeService()
        
    def _initialize_brand_analyzer(self):
        """Initialize brand analyzer"""
        from backend.services.brand_analyzer import BrandAnalyzer
        return BrandAnalyzer()
        
    def _initialize_quality_assessor(self):
        """Initialize quality assessor"""
        from backend.services.quality_assessor import QualityAssessor
        return QualityAssessor()
        
    @abstractmethod
    async def create_content(self, request: CreativeRequest) -> CreativeAsset:
        """Create content based on creative request"""
        pass
    
    @abstractmethod
    async def optimize_for_platform(self, asset: CreativeAsset, platform: str) -> CreativeAsset:
        """Optimize content for specific platform"""
        pass
    
    @abstractmethod
    async def analyze_performance(self, asset: CreativeAsset) -> Dict[str, Any]:
        """Analyze content performance metrics"""
        pass
    
    async def detect_business_niche_from_content(self, content_data: Dict[str, Any]) -> str:
        """Detect business niche from content context"""
        return await self.ai_service.detect_business_niche(
            content_text=content_data.get('text', ''),
            visual_elements=content_data.get('visual_elements', []),
            brand_context=content_data.get('brand_context', {})
        )
    
    async def generate_creative_variations(self, base_asset: CreativeAsset, variation_count: int = 5) -> List[CreativeAsset]:
        """Generate multiple creative variations"""
        variations = []
        
        for i in range(variation_count):
            variation_request = await self.create_variation_request(base_asset, i)
            variation = await self.create_content(variation_request)
            variations.append(variation)
        
        return variations
    
    async def create_variation_request(self, base_asset: CreativeAsset, variation_index: int) -> CreativeRequest:
        """Create a variation request from base asset"""
        # Load base asset metadata
        base_metadata = base_asset.metadata
        
        # Create variation request
        variation_request = CreativeRequest(
            request_id=f"{base_asset.request_id}_var_{variation_index}",
            client_id=self.client_id,
            content_type=base_asset.content_type,
            business_niche=base_metadata.get('business_niche', 'general'),
            target_audience=base_metadata.get('target_audience', {}),
            creative_brief=f"Variation {variation_index} of {base_metadata.get('creative_brief', '')}",
            style_preferences=base_metadata.get('style_preferences', []),
            quality_level=QualityLevel(base_metadata.get('quality_level', 'standard')),
            platform_requirements=base_metadata.get('platform_requirements', {}),
            brand_guidelines=base_metadata.get('brand_guidelines', {})
        )
        
        return variation_request
    
    async def ensure_brand_compliance(self, asset: CreativeAsset, brand_guidelines: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure content complies with brand guidelines"""
        compliance_check = await self.brand_analyzer.check_compliance(
            asset=asset,
            guidelines=brand_guidelines
        )
        
        if compliance_check['compliance_score'] < 0.8:
            # Auto-fix common compliance issues
            fixed_asset = await self.auto_fix_brand_compliance(asset, brand_guidelines, compliance_check)
            return {'compliant': True, 'fixed_asset': fixed_asset, 'original_score': compliance_check['compliance_score']}
        
        return {'compliant': True, 'asset': asset, 'score': compliance_check['compliance_score']}
    
    async def auto_fix_brand_compliance(self, asset: CreativeAsset, brand_guidelines: Dict[str, Any], compliance_check: Dict[str, Any]) -> CreativeAsset:
        """Automatically fix brand compliance issues"""
        # This is a placeholder - specific implementations in derived classes
        logger.info(f"Auto-fixing brand compliance for asset {asset.asset_id}")
        return asset
    
    async def optimize_for_virality(self, asset: CreativeAsset, platform: str, business_niche: str) -> CreativeAsset:
        """Optimize content for viral potential"""
        viral_elements = await self.ai_service.identify_viral_elements(
            platform=platform,
            business_niche=business_niche,
            current_trends=await self.get_current_viral_trends(platform)
        )
        
        optimized_asset = await self.apply_viral_optimizations(asset, viral_elements)
        return optimized_asset
    
    async def get_current_viral_trends(self, platform: str) -> Dict[str, Any]:
        """Get current viral trends for platform"""
        # This would integrate with trend analysis services
        trends = {
            'instagram': {
                'hashtags': ['trending', 'viral', 'instagood'],
                'content_types': ['reels', 'carousel'],
                'themes': ['motivational', 'educational', 'entertaining']
            },
            'tiktok': {
                'sounds': ['trending_sounds'],
                'effects': ['popular_effects'],
                'challenges': ['current_challenges']
            },
            'youtube': {
                'topics': ['trending_topics'],
                'formats': ['shorts', 'tutorials'],
                'thumbnails': ['eye_catching', 'clickbait']
            }
        }
        
        return trends.get(platform, {})
    
    async def apply_viral_optimizations(self, asset: CreativeAsset, viral_elements: Dict[str, Any]) -> CreativeAsset:
        """Apply viral optimization elements to asset"""
        # This is a placeholder - specific implementations in derived classes
        logger.info(f"Applying viral optimizations to asset {asset.asset_id}")
        asset.metadata['viral_elements'] = viral_elements
        return asset
    
    async def log_creation_error(self, error_message: str):
        """Log content creation error"""
        logger.error(f"Content creation error in {self.creator_type}: {error_message}")
        
        # Store error in database for analysis
        error_data = {
            'creator_type': self.creator_type,
            'client_id': self.client_id,
            'error_message': error_message,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to error log
        await self._save_error_log(error_data)
    
    async def _save_error_log(self, error_data: Dict[str, Any]):
        """Save error log to database"""
        # This would integrate with database
        logger.info(f"Saving error log: {error_data}")
        
    def generate_asset_id(self) -> str:
        """Generate unique asset ID"""
        return f"{self.creator_type}_{uuid.uuid4().hex[:12]}"
        
    def generate_request_id(self) -> str:
        """Generate unique request ID"""
        return f"req_{uuid.uuid4().hex[:12]}"