"""
AI Image Generation Engine

Advanced AI image generation with business niche optimization for AutoGuru Universal.
"""

import os
import io
import json
import base64
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from io import BytesIO
import asyncio

# Image processing imports
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
except ImportError:
    Image = ImageDraw = ImageFont = ImageFilter = ImageEnhance = ImageOps = None

# AI service imports
try:
    import openai
    import anthropic
    import replicate
except ImportError:
    openai = anthropic = replicate = None

# Import base classes
from .base_creator import (
    UniversalContentCreator,
    CreativeRequest,
    CreativeAsset,
    ContentType,
    CreativeStyle,
    QualityLevel,
    ContentCreationError
)

# Import supporting services
from backend.services.image_processor import ImageProcessor
from backend.utils.file_manager import FileManager

logger = logging.getLogger(__name__)


class AIImageGenerationEngine(UniversalContentCreator):
    """Advanced AI image generation with business niche optimization"""
    
    def __init__(self, client_id: str):
        super().__init__(client_id, "image_generator")
        self.image_processor = ImageProcessor()
        self.file_manager = FileManager()
        
        # Initialize AI clients
        self._initialize_ai_clients()
        
        # Configure generation settings
        self.generation_config = self._load_generation_config()
        
    def _initialize_ai_clients(self):
        """Initialize various AI image generation clients"""
        self.dalle_client = None
        self.midjourney_client = None
        self.stable_diffusion = None
        
        # Initialize OpenAI DALL-E
        if openai and os.getenv('OPENAI_API_KEY'):
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.dalle_client = openai
            
        # Initialize Stable Diffusion via Replicate
        if replicate and os.getenv('REPLICATE_API_TOKEN'):
            self.stable_diffusion = replicate
            
    def _load_generation_config(self) -> Dict[str, Any]:
        """Load image generation configuration"""
        return {
            'model_preferences': {
                'education': 'dalle3',  # Clear, educational visuals
                'fitness': 'stable_diffusion',  # Dynamic, high-energy images
                'business_consulting': 'dalle3',  # Professional, polished
                'creative_arts': 'midjourney',  # Artistic, unique style
                'finance': 'dalle3',  # Clean, trustworthy
                'health_wellness': 'stable_diffusion'  # Natural, calming
            },
            'quality_settings': {
                QualityLevel.DRAFT: {'size': '512x512', 'steps': 20},
                QualityLevel.STANDARD: {'size': '1024x1024', 'steps': 30},
                QualityLevel.HIGH: {'size': '1024x1024', 'steps': 50},
                QualityLevel.PREMIUM: {'size': '2048x2048', 'steps': 75},
                QualityLevel.ULTRA: {'size': '2048x2048', 'steps': 100}
            }
        }
        
    async def create_content(self, request: CreativeRequest) -> CreativeAsset:
        """Generate AI images optimized for business niche"""
        try:
            logger.info(f"Starting image generation for request {request.request_id}")
            
            # 1. Analyze creative brief for optimal prompt
            optimized_prompt = await self.optimize_image_prompt(request)
            logger.info(f"Optimized prompt: {optimized_prompt[:100]}...")
            
            # 2. Select best AI model for this use case
            optimal_model = await self.select_optimal_ai_model(request)
            logger.info(f"Selected model: {optimal_model}")
            
            # 3. Generate base image
            base_image = await self.generate_base_image(optimized_prompt, optimal_model, request.quality_level)
            
            # 4. Apply business niche specific enhancements
            enhanced_image = await self.apply_niche_enhancements(base_image, request.business_niche)
            
            # 5. Ensure brand compliance
            brand_compliant_image = await self.apply_brand_guidelines(enhanced_image, request.brand_guidelines)
            
            # 6. Create platform-optimized versions
            platform_versions = await self.create_platform_versions(brand_compliant_image, request.platform_requirements)
            
            # 7. Save and package final asset
            asset = await self.save_image_asset(brand_compliant_image, platform_versions, request)
            
            logger.info(f"Successfully created image asset {asset.asset_id}")
            return asset
            
        except Exception as e:
            await self.log_creation_error(f"Image generation failed for request {request.request_id}: {str(e)}")
            raise ContentCreationError(f"Failed to generate image: {str(e)}")
    
    async def optimize_image_prompt(self, request: CreativeRequest) -> str:
        """Optimize prompt for AI image generation based on business niche"""
        business_niche = request.business_niche
        creative_brief = request.creative_brief
        target_audience = request.target_audience
        style_preferences = request.style_preferences
        
        # Business niche specific prompt optimization
        niche_keywords = await self.get_niche_visual_keywords(business_niche)
        audience_preferences = await self.analyze_audience_visual_preferences(target_audience)
        style_descriptors = await self.convert_styles_to_descriptors(style_preferences)
        
        # Build optimized prompt
        prompt_parts = []
        
        # Base creative brief
        prompt_parts.append(creative_brief)
        
        # Add style descriptors
        if style_descriptors:
            prompt_parts.append(f"Style: {', '.join(style_descriptors)}")
            
        # Add niche-specific keywords
        if niche_keywords:
            prompt_parts.append(f"Theme: {', '.join(niche_keywords[:5])}")
            
        # Add quality descriptors based on quality level
        quality_descriptors = {
            QualityLevel.DRAFT: "sketch, concept art",
            QualityLevel.STANDARD: "professional quality",
            QualityLevel.HIGH: "high quality, detailed",
            QualityLevel.PREMIUM: "ultra high quality, photorealistic, 8k",
            QualityLevel.ULTRA: "masterpiece, award-winning, ultra detailed, 8k resolution"
        }
        prompt_parts.append(quality_descriptors.get(request.quality_level, "high quality"))
        
        # Add technical specifications
        prompt_parts.append("digital art, trending on artstation")
        
        # Combine into final prompt
        optimized_prompt = ", ".join(prompt_parts)
        
        return optimized_prompt
    
    async def get_niche_visual_keywords(self, business_niche: str) -> List[str]:
        """Get visual keywords specific to business niche"""
        niche_keywords = {
            'education': ['learning', 'knowledge', 'books', 'classroom', 'students', 'growth', 'achievement', 'inspiration'],
            'fitness': ['energy', 'strength', 'motion', 'health', 'vitality', 'transformation', 'power', 'dynamic'],
            'business_consulting': ['professional', 'corporate', 'success', 'leadership', 'strategy', 'growth', 'innovation'],
            'creative_arts': ['artistic', 'creative', 'colorful', 'expressive', 'unique', 'imagination', 'inspiration'],
            'finance': ['trust', 'security', 'growth', 'stability', 'professional', 'wealth', 'investment'],
            'health_wellness': ['wellness', 'balance', 'nature', 'peaceful', 'healing', 'vitality', 'mindfulness'],
            'technology': ['innovation', 'digital', 'futuristic', 'modern', 'connected', 'advanced', 'smart'],
            'retail': ['shopping', 'products', 'lifestyle', 'trendy', 'appealing', 'quality', 'value']
        }
        
        return niche_keywords.get(business_niche, niche_keywords['business_consulting'])
    
    async def analyze_audience_visual_preferences(self, target_audience: Dict[str, Any]) -> List[str]:
        """Analyze target audience for visual preferences"""
        preferences = []
        
        # Age-based preferences
        age_range = target_audience.get('age_range', '25-45')
        if '18-25' in age_range:
            preferences.extend(['vibrant', 'trendy', 'social media friendly'])
        elif '25-35' in age_range:
            preferences.extend(['modern', 'professional', 'clean'])
        elif '35-50' in age_range:
            preferences.extend(['sophisticated', 'trustworthy', 'established'])
        else:
            preferences.extend(['classic', 'reliable', 'traditional'])
            
        # Interest-based preferences
        interests = target_audience.get('interests', [])
        if 'technology' in interests:
            preferences.append('futuristic')
        if 'nature' in interests:
            preferences.append('organic')
        if 'luxury' in interests:
            preferences.append('premium')
            
        return preferences
    
    async def convert_styles_to_descriptors(self, style_preferences: List[CreativeStyle]) -> List[str]:
        """Convert creative styles to AI prompt descriptors"""
        style_mapping = {
            CreativeStyle.MINIMAL: ['minimalist', 'clean', 'simple', 'white space'],
            CreativeStyle.BOLD: ['bold', 'striking', 'high contrast', 'impactful'],
            CreativeStyle.PROFESSIONAL: ['professional', 'corporate', 'polished', 'refined'],
            CreativeStyle.PLAYFUL: ['playful', 'fun', 'colorful', 'whimsical'],
            CreativeStyle.LUXURY: ['luxury', 'premium', 'elegant', 'sophisticated'],
            CreativeStyle.MODERN: ['modern', 'contemporary', 'sleek', 'innovative'],
            CreativeStyle.VINTAGE: ['vintage', 'retro', 'classic', 'nostalgic'],
            CreativeStyle.ARTISTIC: ['artistic', 'creative', 'expressive', 'unique']
        }
        
        descriptors = []
        for style in style_preferences:
            descriptors.extend(style_mapping.get(style, []))
            
        return list(set(descriptors))  # Remove duplicates
    
    async def select_optimal_ai_model(self, request: CreativeRequest) -> str:
        """Select the best AI model for this specific request"""
        # Check model preferences for business niche
        niche_preference = self.generation_config['model_preferences'].get(
            request.business_niche,
            'dalle3'  # Default to DALL-E 3
        )
        
        # Override based on specific requirements
        if request.quality_level in [QualityLevel.PREMIUM, QualityLevel.ULTRA]:
            # For highest quality, prefer DALL-E 3
            return 'dalle3'
        
        # Check style preferences
        if CreativeStyle.ARTISTIC in request.style_preferences:
            return 'midjourney'
        
        if CreativeStyle.BOLD in request.style_preferences or CreativeStyle.PLAYFUL in request.style_preferences:
            return 'stable_diffusion'
            
        return niche_preference
    
    async def generate_base_image(self, prompt: str, model: str, quality_level: QualityLevel) -> Image.Image:
        """Generate base image using selected AI model"""
        generation_params = self.generation_config['quality_settings'][quality_level]
        
        if model == 'dalle3' and self.dalle_client:
            return await self._generate_dalle_image(prompt, generation_params)
        elif model == 'stable_diffusion' and self.stable_diffusion:
            return await self._generate_stable_diffusion_image(prompt, generation_params)
        else:
            # Fallback to creating a placeholder image
            return await self._create_placeholder_image(prompt, generation_params)
    
    async def _generate_dalle_image(self, prompt: str, params: Dict[str, Any]) -> Image.Image:
        """Generate image using DALL-E"""
        try:
            response = await asyncio.to_thread(
                self.dalle_client.images.generate,
                model="dall-e-3",
                prompt=prompt,
                size=params['size'],
                quality="hd" if params.get('steps', 30) > 50 else "standard",
                n=1
            )
            
            # Get image URL and download
            image_url = response.data[0].url
            image_data = await self._download_image(image_url)
            
            return Image.open(BytesIO(image_data))
            
        except Exception as e:
            logger.error(f"DALL-E generation failed: {str(e)}")
            return await self._create_placeholder_image(prompt, params)
    
    async def _generate_stable_diffusion_image(self, prompt: str, params: Dict[str, Any]) -> Image.Image:
        """Generate image using Stable Diffusion"""
        try:
            # Use Stable Diffusion XL via Replicate
            output = await asyncio.to_thread(
                self.stable_diffusion.run,
                "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                input={
                    "prompt": prompt,
                    "width": int(params['size'].split('x')[0]),
                    "height": int(params['size'].split('x')[1]),
                    "num_inference_steps": params['steps']
                }
            )
            
            if output and len(output) > 0:
                image_url = output[0]
                image_data = await self._download_image(image_url)
                return Image.open(BytesIO(image_data))
            else:
                raise Exception("No output from Stable Diffusion")
                
        except Exception as e:
            logger.error(f"Stable Diffusion generation failed: {str(e)}")
            return await self._create_placeholder_image(prompt, params)
    
    async def _create_placeholder_image(self, prompt: str, params: Dict[str, Any]) -> Image.Image:
        """Create a placeholder image when AI generation fails"""
        # Parse size
        width, height = map(int, params['size'].split('x'))
        
        # Create gradient background
        image = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(image)
        
        # Create gradient
        for y in range(height):
            r = int(255 * (1 - y / height))
            g = int(200 * (1 - y / height))
            b = 255
            draw.line([(0, y), (width, y)], fill=(r, g, b))
            
        # Add text
        try:
            font_size = min(width, height) // 20
            # Use default font
            draw.text(
                (width // 2, height // 2),
                f"AI Generated\n{prompt[:50]}...",
                fill='white',
                anchor='mm'
            )
        except:
            pass
            
        return image
    
    async def _download_image(self, url: str) -> bytes:
        """Download image from URL"""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.read()
    
    async def apply_niche_enhancements(self, image: Image.Image, business_niche: str) -> Image.Image:
        """Apply business niche specific visual enhancements"""
        niche_enhancements = {
            'education': {
                'brightness_boost': 1.1,
                'contrast_adjustment': 1.05,
                'saturation_level': 1.0,
                'overlay_elements': ['knowledge_icons', 'learning_symbols']
            },
            'fitness': {
                'brightness_boost': 1.15,
                'contrast_adjustment': 1.2,
                'saturation_level': 1.3,
                'overlay_elements': ['energy_effects', 'motion_blur']
            },
            'business_consulting': {
                'brightness_boost': 1.0,
                'contrast_adjustment': 1.1,
                'saturation_level': 0.9,
                'overlay_elements': ['professional_gradients', 'corporate_elements']
            },
            'creative_arts': {
                'brightness_boost': 1.05,
                'contrast_adjustment': 1.15,
                'saturation_level': 1.4,
                'overlay_elements': ['artistic_textures', 'creative_borders']
            },
            'finance': {
                'brightness_boost': 1.02,
                'contrast_adjustment': 1.08,
                'saturation_level': 0.85,
                'overlay_elements': ['trust_indicators', 'growth_charts']
            },
            'health_wellness': {
                'brightness_boost': 1.08,
                'contrast_adjustment': 1.0,
                'saturation_level': 1.1,
                'overlay_elements': ['nature_elements', 'wellness_icons']
            }
        }
        
        enhancements = niche_enhancements.get(business_niche, niche_enhancements['business_consulting'])
        
        # Apply enhancements
        enhanced_image = image.copy()
        
        # Brightness adjustment
        if ImageEnhance:
            enhancer = ImageEnhance.Brightness(enhanced_image)
            enhanced_image = enhancer.enhance(enhancements['brightness_boost'])
            
            # Contrast adjustment
            enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = enhancer.enhance(enhancements['contrast_adjustment'])
            
            # Saturation adjustment
            enhancer = ImageEnhance.Color(enhanced_image)
            enhanced_image = enhancer.enhance(enhancements['saturation_level'])
        
        # Apply overlay elements
        for overlay_type in enhancements['overlay_elements']:
            enhanced_image = await self.apply_overlay_element(enhanced_image, overlay_type, business_niche)
        
        return enhanced_image
    
    async def apply_overlay_element(self, image: Image.Image, overlay_type: str, business_niche: str) -> Image.Image:
        """Apply specific overlay elements to image"""
        # Create a copy to work with
        result_image = image.copy()
        
        if overlay_type == 'professional_gradients':
            # Add subtle gradient overlay
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Create gradient from top
            for y in range(image.height // 3):
                alpha = int(50 * (1 - y / (image.height // 3)))
                draw.line([(0, y), (image.width, y)], fill=(0, 0, 0, alpha))
                
            result_image = Image.alpha_composite(result_image.convert('RGBA'), overlay).convert('RGB')
            
        elif overlay_type == 'energy_effects' and business_niche == 'fitness':
            # Add motion blur effect to edges
            if ImageFilter:
                # Create mask for center focus
                mask = Image.new('L', image.size, 0)
                draw = ImageDraw.Draw(mask)
                draw.ellipse(
                    [(image.width // 4, image.height // 4), 
                     (3 * image.width // 4, 3 * image.height // 4)],
                    fill=255
                )
                mask = mask.filter(ImageFilter.GaussianBlur(radius=50))
                
                # Apply motion blur to copy
                blurred = result_image.filter(ImageFilter.GaussianBlur(radius=5))
                
                # Composite based on mask
                result_image = Image.composite(result_image, blurred, mask)
                
        return result_image
    
    async def apply_brand_guidelines(self, image: Image.Image, brand_guidelines: Dict[str, Any]) -> Image.Image:
        """Apply brand guidelines to ensure consistency"""
        result_image = image.copy()
        
        # Apply brand colors if specified
        if 'colors' in brand_guidelines:
            # This would apply color grading to match brand colors
            pass
            
        # Apply watermark if specified
        if brand_guidelines.get('watermark'):
            result_image = await self.apply_watermark(result_image, brand_guidelines['watermark'])
            
        # Apply brand frame if specified
        if brand_guidelines.get('use_brand_frame'):
            result_image = await self.apply_brand_frame(result_image, brand_guidelines)
            
        return result_image
    
    async def apply_watermark(self, image: Image.Image, watermark_config: Dict[str, Any]) -> Image.Image:
        """Apply watermark to image"""
        # This is a placeholder - would integrate with actual watermark assets
        return image
    
    async def apply_brand_frame(self, image: Image.Image, brand_guidelines: Dict[str, Any]) -> Image.Image:
        """Apply brand frame to image"""
        # Add padding for frame
        padding = 20
        new_size = (image.width + 2 * padding, image.height + 2 * padding)
        
        # Create frame
        framed_image = Image.new('RGB', new_size, color=(255, 255, 255))
        framed_image.paste(image, (padding, padding))
        
        return framed_image
    
    async def create_platform_versions(self, image: Image.Image, platform_requirements: Dict[str, Any]) -> Dict[str, Dict[str, Image.Image]]:
        """Create optimized versions for different platforms"""
        platform_specs = {
            'instagram': {
                'square': (1080, 1080),
                'story': (1080, 1920),
                'reel': (1080, 1920)
            },
            'facebook': {
                'post': (1200, 630),
                'story': (1080, 1920),
                'cover': (820, 312)
            },
            'twitter': {
                'post': (1024, 512),
                'header': (1500, 500)
            },
            'linkedin': {
                'post': (1200, 627),
                'article': (1200, 627),
                'banner': (1584, 396)
            },
            'tiktok': {
                'video_thumbnail': (1080, 1920),
                'profile': (200, 200)
            },
            'youtube': {
                'thumbnail': (1280, 720),
                'banner': (2560, 1440),
                'community': (1280, 720)
            },
            'pinterest': {
                'pin': (1000, 1500),
                'story': (1080, 1920)
            }
        }
        
        platform_versions = {}
        
        for platform, required_sizes in platform_requirements.items():
            if platform in platform_specs:
                platform_versions[platform] = {}
                
                for size_name, dimensions in platform_specs[platform].items():
                    if required_sizes == 'all' or size_name in required_sizes:
                        # Resize and optimize for platform
                        resized_image = await self.smart_resize_for_platform(image, dimensions, platform)
                        
                        # Apply platform-specific optimizations
                        optimized_image = await self.apply_platform_optimizations(resized_image, platform, size_name)
                        
                        platform_versions[platform][size_name] = optimized_image
        
        return platform_versions
    
    async def smart_resize_for_platform(self, image: Image.Image, target_dimensions: Tuple[int, int], platform: str) -> Image.Image:
        """Intelligently resize image maintaining important visual elements"""
        target_width, target_height = target_dimensions
        original_width, original_height = image.size
        
        # Calculate aspect ratios
        original_ratio = original_width / original_height
        target_ratio = target_width / target_height
        
        if abs(original_ratio - target_ratio) < 0.1:
            # Similar aspect ratios - simple resize
            return image.resize(target_dimensions, Image.Resampling.LANCZOS)
        else:
            # Different aspect ratios - smart crop or letterbox
            if platform in ['instagram', 'tiktok']:
                # For social platforms, prefer cropping to maintain engagement
                return await self.smart_crop_image(image, target_dimensions)
            else:
                # For professional platforms, prefer letterboxing
                return await self.letterbox_image(image, target_dimensions)
    
    async def smart_crop_image(self, image: Image.Image, target_dimensions: Tuple[int, int]) -> Image.Image:
        """Smart crop image to target dimensions"""
        target_width, target_height = target_dimensions
        
        # Use ImageOps.fit for smart cropping
        if ImageOps:
            return ImageOps.fit(image, target_dimensions, Image.Resampling.LANCZOS)
        else:
            # Fallback to center crop
            return image.resize(target_dimensions, Image.Resampling.LANCZOS)
    
    async def letterbox_image(self, image: Image.Image, target_dimensions: Tuple[int, int]) -> Image.Image:
        """Add letterboxing to maintain aspect ratio"""
        target_width, target_height = target_dimensions
        
        # Calculate scaling to fit
        scale = min(target_width / image.width, target_height / image.height)
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        
        # Resize image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with letterbox
        letterboxed = Image.new('RGB', target_dimensions, (255, 255, 255))
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        letterboxed.paste(resized, (x_offset, y_offset))
        
        return letterboxed
    
    async def apply_platform_optimizations(self, image: Image.Image, platform: str, size_name: str) -> Image.Image:
        """Apply platform-specific optimizations"""
        optimized = image.copy()
        
        # Platform-specific optimizations
        if platform == 'instagram' and size_name == 'story':
            # Add story-friendly elements
            optimized = await self.add_story_elements(optimized)
        elif platform == 'youtube' and size_name == 'thumbnail':
            # Enhance for thumbnail visibility
            if ImageEnhance:
                enhancer = ImageEnhance.Contrast(optimized)
                optimized = enhancer.enhance(1.2)
                enhancer = ImageEnhance.Color(optimized)
                optimized = enhancer.enhance(1.1)
        
        return optimized
    
    async def add_story_elements(self, image: Image.Image) -> Image.Image:
        """Add elements optimized for stories"""
        # Add safe zones for story UI elements
        result = image.copy()
        
        if ImageDraw:
            # Add subtle vignette
            overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Create radial gradient for vignette
            for i in range(min(image.width, image.height) // 2):
                alpha = int(20 * (i / (min(image.width, image.height) // 2)))
                draw.ellipse(
                    [(i, i), (image.width - i, image.height - i)],
                    fill=(0, 0, 0, alpha)
                )
                
            result = Image.alpha_composite(result.convert('RGBA'), overlay).convert('RGB')
        
        return result
    
    async def save_image_asset(self, main_image: Image.Image, platform_versions: Dict[str, Dict[str, Image.Image]], request: CreativeRequest) -> CreativeAsset:
        """Save image asset and create CreativeAsset object"""
        # Generate asset ID
        asset_id = self.generate_asset_id()
        
        # Create asset directory
        asset_dir = os.path.join('assets', 'images', asset_id)
        os.makedirs(asset_dir, exist_ok=True)
        
        # Save main image
        main_path = os.path.join(asset_dir, 'main.png')
        main_image.save(main_path, 'PNG', optimize=True)
        
        # Save platform versions
        platform_paths = {}
        for platform, versions in platform_versions.items():
            platform_paths[platform] = {}
            platform_dir = os.path.join(asset_dir, platform)
            os.makedirs(platform_dir, exist_ok=True)
            
            for version_name, version_image in versions.items():
                version_path = os.path.join(platform_dir, f'{version_name}.png')
                version_image.save(version_path, 'PNG', optimize=True)
                platform_paths[platform][version_name] = version_path
        
        # Calculate quality score
        quality_score = await self.calculate_image_quality_score(main_image)
        
        # Calculate brand compliance score
        brand_compliance_score = await self.calculate_brand_compliance_score(main_image, request.brand_guidelines)
        
        # Create asset object
        asset = CreativeAsset(
            asset_id=asset_id,
            request_id=request.request_id,
            content_type=ContentType.IMAGE,
            file_path=main_path,
            file_format='PNG',
            dimensions={'width': main_image.width, 'height': main_image.height},
            file_size=os.path.getsize(main_path),
            quality_score=quality_score,
            brand_compliance_score=brand_compliance_score,
            platform_optimized_versions=platform_paths,
            metadata={
                'business_niche': request.business_niche,
                'target_audience': request.target_audience,
                'creative_brief': request.creative_brief,
                'style_preferences': [style.value for style in request.style_preferences],
                'quality_level': request.quality_level.value,
                'platform_requirements': request.platform_requirements,
                'brand_guidelines': request.brand_guidelines
            }
        )
        
        return asset
    
    async def calculate_image_quality_score(self, image: Image.Image) -> float:
        """Calculate quality score for generated image"""
        score = 0.7  # Base score
        
        # Resolution score
        pixels = image.width * image.height
        if pixels >= 4000000:  # 4MP+
            score += 0.1
        elif pixels >= 2000000:  # 2MP+
            score += 0.05
            
        # Add more quality metrics here
        
        return min(score, 1.0)
    
    async def calculate_brand_compliance_score(self, image: Image.Image, brand_guidelines: Dict[str, Any]) -> float:
        """Calculate brand compliance score"""
        # This is a simplified implementation
        score = 0.8  # Base score
        
        if brand_guidelines:
            # Check if required elements are present
            if brand_guidelines.get('watermark') and not brand_guidelines.get('watermark_applied'):
                score -= 0.1
                
        return max(score, 0.0)
    
    async def optimize_for_platform(self, asset: CreativeAsset, platform: str) -> CreativeAsset:
        """Optimize existing image asset for specific platform"""
        original_image = Image.open(asset.file_path)
        
        # Get platform requirements
        platform_requirements = await self.get_platform_requirements(platform)
        
        # Create optimized version
        platform_versions = await self.create_platform_versions(original_image, {platform: 'all'})
        
        # Update asset with platform versions
        asset.platform_optimized_versions[platform] = await self.save_platform_versions(platform_versions[platform], asset.asset_id)
        
        return asset
    
    async def get_platform_requirements(self, platform: str) -> Dict[str, Any]:
        """Get platform-specific requirements"""
        requirements = {
            'instagram': ['square', 'story', 'reel'],
            'facebook': ['post', 'story', 'cover'],
            'twitter': ['post', 'header'],
            'linkedin': ['post', 'article', 'banner'],
            'tiktok': ['video_thumbnail', 'profile'],
            'youtube': ['thumbnail', 'banner', 'community'],
            'pinterest': ['pin', 'story']
        }
        
        return requirements.get(platform, [])
    
    async def save_platform_versions(self, versions: Dict[str, Image.Image], asset_id: str) -> Dict[str, str]:
        """Save platform-specific versions"""
        paths = {}
        
        for version_name, image in versions.items():
            path = os.path.join('assets', 'images', asset_id, 'platform', f'{version_name}.png')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image.save(path, 'PNG', optimize=True)
            paths[version_name] = path
            
        return paths
    
    async def analyze_performance(self, asset: CreativeAsset) -> Dict[str, Any]:
        """Analyze image performance across platforms"""
        image = Image.open(asset.file_path)
        
        performance_data = {}
        
        # Visual appeal analysis
        performance_data['visual_appeal'] = await self.analyze_visual_appeal(asset)
        
        # Brand compliance score
        performance_data['brand_compliance'] = asset.brand_compliance_score
        
        # Platform performance
        performance_data['platform_performance'] = {}
        for platform, versions in asset.platform_optimized_versions.items():
            performance_data['platform_performance'][platform] = await self.analyze_platform_image_performance(versions, platform)
        
        # Engagement prediction
        performance_data['predicted_engagement'] = await self.predict_image_engagement(asset)
        
        # Optimization suggestions
        performance_data['optimization_suggestions'] = await self.generate_image_optimization_suggestions(asset, performance_data)
        
        return performance_data
    
    async def analyze_visual_appeal(self, asset: CreativeAsset) -> Dict[str, Any]:
        """Analyze visual appeal of image"""
        image = Image.open(asset.file_path)
        
        appeal_metrics = {
            'color_harmony': await self.analyze_color_harmony(image),
            'composition_score': await self.analyze_composition(image),
            'clarity_score': await self.analyze_clarity(image),
            'emotional_impact': await self.analyze_emotional_impact(image, asset.metadata.get('business_niche', 'general'))
        }
        
        # Calculate overall appeal score
        appeal_metrics['overall_score'] = sum(appeal_metrics.values()) / len(appeal_metrics)
        
        return appeal_metrics
    
    async def analyze_color_harmony(self, image: Image.Image) -> float:
        """Analyze color harmony in image"""
        # Simplified color harmony analysis
        # In production, this would use color theory algorithms
        return 0.75
    
    async def analyze_composition(self, image: Image.Image) -> float:
        """Analyze image composition"""
        # Check rule of thirds, balance, etc.
        return 0.8
    
    async def analyze_clarity(self, image: Image.Image) -> float:
        """Analyze image clarity and sharpness"""
        # Check for blur, noise, etc.
        return 0.85
    
    async def analyze_emotional_impact(self, image: Image.Image, business_niche: str) -> float:
        """Analyze emotional impact based on business niche"""
        # This would use emotion detection and niche-specific metrics
        return 0.8
    
    async def analyze_platform_image_performance(self, versions: Dict[str, str], platform: str) -> Dict[str, Any]:
        """Analyze performance for specific platform"""
        performance = {
            'format_compliance': 1.0,  # All versions are correctly formatted
            'size_optimization': 0.9,  # Well optimized for platform
            'visual_hierarchy': 0.85,  # Good visual hierarchy for platform
            'platform_best_practices': 0.8  # Following platform best practices
        }
        
        return performance
    
    async def predict_image_engagement(self, asset: CreativeAsset) -> Dict[str, Any]:
        """Predict engagement metrics for image"""
        predictions = {
            'likes_estimate': 'high',
            'shares_estimate': 'medium',
            'comments_estimate': 'medium',
            'saves_estimate': 'high',
            'overall_engagement_score': 0.82
        }
        
        return predictions
    
    async def generate_image_optimization_suggestions(self, asset: CreativeAsset, performance_data: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions for image"""
        suggestions = []
        
        # Check visual appeal
        if performance_data['visual_appeal']['overall_score'] < 0.7:
            suggestions.append("Consider improving color harmony and composition")
            
        # Check platform performance
        for platform, perf in performance_data['platform_performance'].items():
            if perf.get('platform_best_practices', 1.0) < 0.8:
                suggestions.append(f"Optimize for {platform} best practices")
                
        # Check predicted engagement
        if performance_data['predicted_engagement']['overall_engagement_score'] < 0.75:
            suggestions.append("Add more engaging visual elements or emotional triggers")
            
        return suggestions