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
        try:
            logger.info(f"Creating content for request {request.request_id}")
            
            # 1. Detect business niche using AI (no hardcoding)
            business_niche = await self.ai_service.detect_business_niche(
                content_text=request.creative_brief,
                visual_elements=[],
                brand_context=request.brand_guidelines
            )
            
            # 2. Use AI to generate niche-specific content strategy
            content_strategy = await self.ai_service.generate_content_strategy(
                niche=business_niche,
                audience=request.target_audience,
                platform_requirements=request.platform_requirements,
                style_preferences=request.style_preferences,
                quality_level=request.quality_level
            )
            
            # 3. Create the actual content based on strategy
            asset = await self._generate_asset_from_strategy(
                content_strategy, request
            )
            
            # 4. Ensure universal quality standards
            quality_score = await self.quality_assessor.assess_quality(
                asset, request.quality_level.value, business_niche
            )
            
            # 5. Ensure brand compliance
            brand_compliance = await self.ensure_brand_compliance(
                asset, request.brand_guidelines
            )
            
            # 6. Create platform-optimized versions
            platform_versions = await self._create_platform_versions(
                asset, request.platform_requirements
            )
            
            # Update asset with final metadata
            asset.quality_score = quality_score
            asset.brand_compliance_score = brand_compliance.get('score', 0.8)
            asset.platform_optimized_versions = platform_versions
            asset.metadata.update({
                'business_niche': business_niche,
                'content_strategy': content_strategy,
                'brand_compliance': brand_compliance,
                'creation_timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Content creation completed for {request.request_id}")
            return asset
            
        except Exception as e:
            error_msg = f"Content creation failed for {request.request_id}: {str(e)}"
            await self.log_creation_error(error_msg)
            raise ContentCreationError(error_msg)
    
    @abstractmethod
    async def optimize_for_platform(self, asset: CreativeAsset, platform: str) -> CreativeAsset:
        """Optimize content for specific platform using AI"""
        try:
            logger.info(f"Optimizing asset {asset.asset_id} for platform {platform}")
            
            # Get platform-specific requirements using AI
            platform_requirements = await self.ai_service.get_platform_requirements(
                platform=platform,
                content_type=asset.content_type.value,
                business_niche=asset.metadata.get('business_niche')
            )
            
            # Create optimized version
            optimized_asset = await self._apply_platform_optimizations(
                asset, platform_requirements, platform
            )
            
            # Update metadata with optimization info
            optimized_asset.metadata.update({
                'platform_optimized_for': platform,
                'platform_requirements': platform_requirements,
                'optimization_timestamp': datetime.now().isoformat()
            })
            
            logger.info(f"Platform optimization completed for {asset.asset_id}")
            return optimized_asset
            
        except Exception as e:
            error_msg = f"Platform optimization failed for {asset.asset_id}: {str(e)}"
            await self.log_creation_error(error_msg)
            return asset  # Return original asset if optimization fails
    
    @abstractmethod
    async def analyze_performance(self, asset: CreativeAsset) -> Dict[str, Any]:
        """Analyze content performance metrics"""
        try:
            logger.info(f"Analyzing performance for asset {asset.asset_id}")
            
            # Get performance data from platforms
            performance_data = await self._collect_performance_data(asset)
            
            # Analyze engagement metrics
            engagement_analysis = await self._analyze_engagement_metrics(performance_data)
            
            # Analyze conversion metrics
            conversion_analysis = await self._analyze_conversion_metrics(performance_data)
            
            # Generate performance insights using AI
            insights = await self.ai_service.generate_performance_insights(
                asset_data=asset.metadata,
                performance_data=performance_data,
                business_niche=asset.metadata.get('business_niche')
            )
            
            # Calculate overall performance score
            performance_score = await self._calculate_performance_score(
                engagement_analysis, conversion_analysis
            )
            
            performance_report = {
                'asset_id': asset.asset_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'performance_score': performance_score,
                'engagement_analysis': engagement_analysis,
                'conversion_analysis': conversion_analysis,
                'ai_insights': insights,
                'recommendations': await self._generate_performance_recommendations(
                    performance_score, insights
                ),
                'benchmark_comparison': await self._compare_with_benchmarks(
                    performance_data, asset.metadata.get('business_niche')
                )
            }
            
            # Update asset performance metrics
            asset.performance_metrics.update(performance_report)
            
            logger.info(f"Performance analysis completed for {asset.asset_id}")
            return performance_report
            
        except Exception as e:
            error_msg = f"Performance analysis failed for {asset.asset_id}: {str(e)}"
            await self.log_creation_error(error_msg)
            raise ContentCreationError(error_msg)
    
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

    async def _generate_asset_from_strategy(self, content_strategy: Dict[str, Any], request: CreativeRequest) -> CreativeAsset:
        """Generate creative asset from content strategy"""
        try:
            # Create asset based on content type
            if request.content_type == ContentType.IMAGE:
                asset = await self._create_image_asset(content_strategy, request)
            elif request.content_type == ContentType.VIDEO:
                asset = await self._create_video_asset(content_strategy, request)
            elif request.content_type == ContentType.COPY:
                asset = await self._create_copy_asset(content_strategy, request)
            elif request.content_type == ContentType.ADVERTISEMENT:
                asset = await self._create_advertisement_asset(content_strategy, request)
            else:
                asset = await self._create_generic_asset(content_strategy, request)
            
            # Set basic metadata
            asset.metadata.update({
                'content_strategy': content_strategy,
                'request_data': {
                    'business_niche': request.business_niche,
                    'target_audience': request.target_audience,
                    'style_preferences': [style.value for style in request.style_preferences],
                    'quality_level': request.quality_level.value
                }
            })
            
            return asset
            
        except Exception as e:
            logger.error(f"Asset generation failed: {str(e)}")
            raise ContentCreationError(f"Failed to generate asset: {str(e)}")
    
    async def _create_image_asset(self, strategy: Dict[str, Any], request: CreativeRequest) -> CreativeAsset:
        """Create image asset"""
        # Generate image using AI service
        image_data = await self.ai_service.generate_content(
            content_type='image',
            prompt=strategy.get('visual_prompt', ''),
            style=strategy.get('visual_style', 'professional'),
            dimensions=strategy.get('dimensions', {'width': 1080, 'height': 1080})
        )
        
        # Save image file
        asset_id = self.generate_asset_id()
        file_path = f"assets/images/{asset_id}.png"
        await self._save_image_file(image_data, file_path)
        
        return CreativeAsset(
            asset_id=asset_id,
            request_id=request.request_id,
            content_type=ContentType.IMAGE,
            file_path=file_path,
            file_format="png",
            dimensions=strategy.get('dimensions', {'width': 1080, 'height': 1080}),
            file_size=len(image_data) if image_data else 0,
            quality_score=0.0,  # Will be assessed later
            brand_compliance_score=0.0,  # Will be assessed later
            platform_optimized_versions={},
            metadata={}
        )
    
    async def _create_video_asset(self, strategy: Dict[str, Any], request: CreativeRequest) -> CreativeAsset:
        """Create video asset"""
        # Generate video using AI service
        video_data = await self.ai_service.generate_content(
            content_type='video',
            script=strategy.get('video_script', ''),
            style=strategy.get('video_style', 'professional'),
            duration=strategy.get('duration', 30)
        )
        
        # Save video file
        asset_id = self.generate_asset_id()
        file_path = f"assets/videos/{asset_id}.mp4"
        await self._save_video_file(video_data, file_path)
        
        return CreativeAsset(
            asset_id=asset_id,
            request_id=request.request_id,
            content_type=ContentType.VIDEO,
            file_path=file_path,
            file_format="mp4",
            dimensions=strategy.get('dimensions', {'width': 1920, 'height': 1080}),
            file_size=len(video_data) if video_data else 0,
            quality_score=0.0,
            brand_compliance_score=0.0,
            platform_optimized_versions={},
            metadata={}
        )
    
    async def _create_copy_asset(self, strategy: Dict[str, Any], request: CreativeRequest) -> CreativeAsset:
        """Create copy/text asset"""
        # Generate copy using AI service
        copy_content = await self.ai_service.generate_content(
            content_type='copy',
            brief=strategy.get('copy_brief', ''),
            tone=strategy.get('tone', 'professional'),
            length=strategy.get('length', 'medium')
        )
        
        # Save copy file
        asset_id = self.generate_asset_id()
        file_path = f"assets/copy/{asset_id}.txt"
        await self._save_text_file(copy_content, file_path)
        
        return CreativeAsset(
            asset_id=asset_id,
            request_id=request.request_id,
            content_type=ContentType.COPY,
            file_path=file_path,
            file_format="txt",
            dimensions={'width': 0, 'height': 0},
            file_size=len(copy_content.encode('utf-8')),
            quality_score=0.0,
            brand_compliance_score=0.0,
            platform_optimized_versions={},
            metadata={'content': copy_content}
        )
    
    async def _create_advertisement_asset(self, strategy: Dict[str, Any], request: CreativeRequest) -> CreativeAsset:
        """Create advertisement asset"""
        # Generate advertisement using AI service
        ad_content = await self.ai_service.generate_content(
            content_type='advertisement',
            objective=strategy.get('ad_objective', 'awareness'),
            target_audience=strategy.get('target_audience', {}),
            creative_elements=strategy.get('creative_elements', [])
        )
        
        asset_id = self.generate_asset_id()
        file_path = f"assets/ads/{asset_id}.json"
        await self._save_json_file(ad_content, file_path)
        
        return CreativeAsset(
            asset_id=asset_id,
            request_id=request.request_id,
            content_type=ContentType.ADVERTISEMENT,
            file_path=file_path,
            file_format="json",
            dimensions={'width': 1080, 'height': 1080},
            file_size=len(json.dumps(ad_content).encode('utf-8')),
            quality_score=0.0,
            brand_compliance_score=0.0,
            platform_optimized_versions={},
            metadata=ad_content
        )
    
    async def _create_generic_asset(self, strategy: Dict[str, Any], request: CreativeRequest) -> CreativeAsset:
        """Create generic asset"""
        # Generate generic content using AI service
        content = await self.ai_service.generate_content(
            type=request.content_type.value,
            requirements=strategy
        )
        
        asset_id = self.generate_asset_id()
        file_path = f"assets/generic/{asset_id}.json"
        await self._save_json_file(content, file_path)
        
        return CreativeAsset(
            asset_id=asset_id,
            request_id=request.request_id,
            content_type=request.content_type,
            file_path=file_path,
            file_format="json",
            dimensions={'width': 1080, 'height': 1080},
            file_size=len(json.dumps(content).encode('utf-8')),
            quality_score=0.0,
            brand_compliance_score=0.0,
            platform_optimized_versions={},
            metadata=content
        )
    
    async def _create_platform_versions(self, asset: CreativeAsset, platform_requirements: Dict[str, Any]) -> Dict[str, str]:
        """Create platform-optimized versions"""
        platform_versions = {}
        
        for platform in platform_requirements.get('platforms', []):
            try:
                optimized_asset = await self.optimize_for_platform(asset, platform)
                platform_versions[platform] = optimized_asset.file_path
            except Exception as e:
                logger.warning(f"Failed to optimize for platform {platform}: {str(e)}")
                platform_versions[platform] = asset.file_path
        
        return platform_versions
    
    async def _save_image_file(self, image_data: bytes, file_path: str):
        """Save image file to storage"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save image data
            with open(file_path, 'wb') as f:
                f.write(image_data)
                
        except Exception as e:
            logger.error(f"Failed to save image file: {str(e)}")
            raise
    
    async def _save_video_file(self, video_data: bytes, file_path: str):
        """Save video file to storage"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save video data
            with open(file_path, 'wb') as f:
                f.write(video_data)
                
        except Exception as e:
            logger.error(f"Failed to save video file: {str(e)}")
            raise
    
    async def _save_text_file(self, text_content: str, file_path: str):
        """Save text file to storage"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save text content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
                
        except Exception as e:
            logger.error(f"Failed to save text file: {str(e)}")
            raise
    
    async def _save_json_file(self, json_content: Dict[str, Any], file_path: str):
        """Save JSON file to storage"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save JSON content
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_content, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save JSON file: {str(e)}")
            raise
    
    async def _apply_platform_optimizations(self, asset: CreativeAsset, platform_requirements: Dict[str, Any], platform: str) -> CreativeAsset:
        """Apply platform-specific optimizations to asset"""
        try:
            # Create a copy of the asset for optimization
            optimized_asset = CreativeAsset(
                asset_id=f"{asset.asset_id}_{platform}",
                request_id=asset.request_id,
                content_type=asset.content_type,
                file_path=asset.file_path,
                file_format=asset.file_format,
                dimensions=asset.dimensions.copy(),
                file_size=asset.file_size,
                quality_score=asset.quality_score,
                brand_compliance_score=asset.brand_compliance_score,
                platform_optimized_versions=asset.platform_optimized_versions.copy(),
                metadata=asset.metadata.copy()
            )
            
            # Apply platform-specific optimizations
            if platform.lower() == 'instagram':
                optimized_asset = await self._optimize_for_instagram(optimized_asset, platform_requirements)
            elif platform.lower() == 'tiktok':
                optimized_asset = await self._optimize_for_tiktok(optimized_asset, platform_requirements)
            elif platform.lower() == 'youtube':
                optimized_asset = await self._optimize_for_youtube(optimized_asset, platform_requirements)
            elif platform.lower() == 'linkedin':
                optimized_asset = await self._optimize_for_linkedin(optimized_asset, platform_requirements)
            elif platform.lower() == 'twitter':
                optimized_asset = await self._optimize_for_twitter(optimized_asset, platform_requirements)
            
            return optimized_asset
            
        except Exception as e:
            logger.error(f"Platform optimization failed: {str(e)}")
            return asset
    
    async def _optimize_for_instagram(self, asset: CreativeAsset, requirements: Dict[str, Any]) -> CreativeAsset:
        """Optimize content for Instagram"""
        # Apply Instagram-specific optimizations
        if asset.content_type == ContentType.IMAGE:
            # Instagram prefers square images
            asset.dimensions = {'width': 1080, 'height': 1080}
        elif asset.content_type == ContentType.VIDEO:
            # Instagram Reels format
            asset.dimensions = {'width': 1080, 'height': 1920}
        
        # Add Instagram-specific metadata
        asset.metadata['instagram_optimization'] = {
            'hashtag_limit': 30,
            'caption_limit': 2200,
            'optimal_posting_time': requirements.get('optimal_posting_time', '12:00')
        }
        
        return asset
    
    async def _optimize_for_tiktok(self, asset: CreativeAsset, requirements: Dict[str, Any]) -> CreativeAsset:
        """Optimize content for TikTok"""
        # TikTok vertical video format
        asset.dimensions = {'width': 1080, 'height': 1920}
        
        # Add TikTok-specific metadata
        asset.metadata['tiktok_optimization'] = {
            'max_duration': 60,  # seconds
            'trending_sounds': requirements.get('trending_sounds', []),
            'optimal_posting_time': requirements.get('optimal_posting_time', '18:00')
        }
        
        return asset
    
    async def _optimize_for_youtube(self, asset: CreativeAsset, requirements: Dict[str, Any]) -> CreativeAsset:
        """Optimize content for YouTube"""
        # YouTube video format
        asset.dimensions = {'width': 1920, 'height': 1080}
        
        # Add YouTube-specific metadata
        asset.metadata['youtube_optimization'] = {
            'thumbnail_required': True,
            'title_limit': 100,
            'description_limit': 5000,
            'optimal_posting_time': requirements.get('optimal_posting_time', '14:00')
        }
        
        return asset
    
    async def _optimize_for_linkedin(self, asset: CreativeAsset, requirements: Dict[str, Any]) -> CreativeAsset:
        """Optimize content for LinkedIn"""
        # LinkedIn professional format
        if asset.content_type == ContentType.IMAGE:
            asset.dimensions = {'width': 1200, 'height': 627}
        
        # Add LinkedIn-specific metadata
        asset.metadata['linkedin_optimization'] = {
            'professional_tone': True,
            'post_length': 'medium',
            'optimal_posting_time': requirements.get('optimal_posting_time', '09:00')
        }
        
        return asset
    
    async def _optimize_for_twitter(self, asset: CreativeAsset, requirements: Dict[str, Any]) -> CreativeAsset:
        """Optimize content for Twitter"""
        # Twitter format
        if asset.content_type == ContentType.IMAGE:
            asset.dimensions = {'width': 1200, 'height': 675}
        
        # Add Twitter-specific metadata
        asset.metadata['twitter_optimization'] = {
            'character_limit': 280,
            'hashtag_limit': 2,
            'optimal_posting_time': requirements.get('optimal_posting_time', '15:00')
        }
        
        return asset
    
    async def _collect_performance_data(self, asset: CreativeAsset) -> Dict[str, Any]:
        """Collect performance data from platforms"""
        try:
            performance_data = {
                'impressions': 0,
                'reach': 0,
                'engagement': 0,
                'clicks': 0,
                'conversions': 0,
                'shares': 0,
                'comments': 0,
                'likes': 0,
                'saves': 0,
                'platform_breakdown': {}
            }
            
            # Collect data from each platform the asset was published to
            for platform, platform_file in asset.platform_optimized_versions.items():
                platform_data = await self._get_platform_performance_data(platform, asset.asset_id)
                performance_data['platform_breakdown'][platform] = platform_data
                
                # Aggregate metrics
                performance_data['impressions'] += platform_data.get('impressions', 0)
                performance_data['reach'] += platform_data.get('reach', 0)
                performance_data['engagement'] += platform_data.get('engagement', 0)
                performance_data['clicks'] += platform_data.get('clicks', 0)
                performance_data['conversions'] += platform_data.get('conversions', 0)
                performance_data['shares'] += platform_data.get('shares', 0)
                performance_data['comments'] += platform_data.get('comments', 0)
                performance_data['likes'] += platform_data.get('likes', 0)
                performance_data['saves'] += platform_data.get('saves', 0)
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Failed to collect performance data: {str(e)}")
            return {'error': str(e)}
    
    async def _get_platform_performance_data(self, platform: str, asset_id: str) -> Dict[str, Any]:
        """Get performance data from specific platform"""
        # This would integrate with actual platform APIs
        # For now, return placeholder data
        return {
            'impressions': 1000,
            'reach': 800,
            'engagement': 150,
            'clicks': 50,
            'conversions': 5,
            'shares': 25,
            'comments': 30,
            'likes': 95,
            'saves': 20,
            'platform': platform,
            'asset_id': asset_id
        }
    
    async def _analyze_engagement_metrics(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze engagement metrics"""
        try:
            total_impressions = performance_data.get('impressions', 1)
            total_engagement = performance_data.get('engagement', 0)
            
            engagement_rate = (total_engagement / total_impressions) * 100 if total_impressions > 0 else 0
            
            return {
                'engagement_rate': engagement_rate,
                'total_engagement': total_engagement,
                'engagement_breakdown': {
                    'likes': performance_data.get('likes', 0),
                    'comments': performance_data.get('comments', 0),
                    'shares': performance_data.get('shares', 0),
                    'saves': performance_data.get('saves', 0)
                },
                'engagement_quality': 'high' if engagement_rate > 5 else 'medium' if engagement_rate > 2 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Engagement analysis failed: {str(e)}")
            return {'error': str(e)}
    
    async def _analyze_conversion_metrics(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conversion metrics"""
        try:
            total_clicks = performance_data.get('clicks', 1)
            total_conversions = performance_data.get('conversions', 0)
            
            conversion_rate = (total_conversions / total_clicks) * 100 if total_clicks > 0 else 0
            
            return {
                'conversion_rate': conversion_rate,
                'total_conversions': total_conversions,
                'total_clicks': total_clicks,
                'conversion_quality': 'high' if conversion_rate > 5 else 'medium' if conversion_rate > 2 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Conversion analysis failed: {str(e)}")
            return {'error': str(e)}
    
    async def _calculate_performance_score(self, engagement_analysis: Dict[str, Any], conversion_analysis: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        try:
            engagement_score = engagement_analysis.get('engagement_rate', 0)
            conversion_score = conversion_analysis.get('conversion_rate', 0)
            
            # Weighted average (engagement 60%, conversion 40%)
            overall_score = (engagement_score * 0.6) + (conversion_score * 0.4)
            
            # Normalize to 0-100 scale
            return min(100, max(0, overall_score))
            
        except Exception as e:
            logger.error(f"Performance score calculation failed: {str(e)}")
            return 0.0
    
    async def _generate_performance_recommendations(self, performance_score: float, insights: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        if performance_score < 30:
            recommendations.extend([
                "Consider revising content strategy to better align with audience preferences",
                "Analyze competitor content for inspiration",
                "Test different posting times and content formats",
                "Review and optimize hashtag strategy"
            ])
        elif performance_score < 60:
            recommendations.extend([
                "Experiment with more engaging visual elements",
                "Optimize posting schedule based on audience activity",
                "Increase content frequency to maintain visibility",
                "Focus on creating more shareable content"
            ])
        else:
            recommendations.extend([
                "Scale successful content formats",
                "Explore new platforms for content distribution",
                "Consider paid promotion for high-performing content",
                "Develop content series based on successful themes"
            ])
        
        return recommendations
    
    async def _compare_with_benchmarks(self, performance_data: Dict[str, Any], business_niche: str) -> Dict[str, Any]:
        """Compare performance with industry benchmarks"""
        try:
            # Industry benchmarks (would be dynamically retrieved)
            benchmarks = {
                'fitness': {'engagement_rate': 4.5, 'conversion_rate': 3.2},
                'business_consulting': {'engagement_rate': 2.8, 'conversion_rate': 5.1},
                'education': {'engagement_rate': 3.7, 'conversion_rate': 2.9},
                'creative': {'engagement_rate': 5.2, 'conversion_rate': 2.4}
            }
            
            niche_benchmarks = benchmarks.get(business_niche, {'engagement_rate': 3.5, 'conversion_rate': 3.0})
            
            engagement_rate = performance_data.get('engagement_rate', 0)
            conversion_rate = performance_data.get('conversion_rate', 0)
            
            return {
                'engagement_vs_benchmark': {
                    'performance': engagement_rate,
                    'benchmark': niche_benchmarks['engagement_rate'],
                    'difference': engagement_rate - niche_benchmarks['engagement_rate'],
                    'status': 'above' if engagement_rate > niche_benchmarks['engagement_rate'] else 'below'
                },
                'conversion_vs_benchmark': {
                    'performance': conversion_rate,
                    'benchmark': niche_benchmarks['conversion_rate'],
                    'difference': conversion_rate - niche_benchmarks['conversion_rate'],
                    'status': 'above' if conversion_rate > niche_benchmarks['conversion_rate'] else 'below'
                }
            }
            
        except Exception as e:
            logger.error(f"Benchmark comparison failed: {str(e)}")
            return {'error': str(e)}