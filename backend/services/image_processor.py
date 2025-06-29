"""
Image Processor Service

Handles image processing and manipulation tasks.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import io
import base64
from dataclasses import dataclass

# Image processing imports
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
    import numpy as np
except ImportError:
    Image = ImageDraw = ImageFont = ImageFilter = ImageEnhance = ImageOps = None
    np = None

logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Image metadata structure"""
    width: int
    height: int
    format: str
    mode: str
    file_size: int
    has_transparency: bool
    dominant_colors: List[str]


class ImageProcessor:
    """Processes and manipulates images"""
    
    def __init__(self):
        self.supported_formats = ['JPEG', 'PNG', 'GIF', 'BMP', 'WEBP']
        self.optimal_sizes = {
            'thumbnail': (150, 150),
            'small': (320, 240),
            'medium': (640, 480),
            'large': (1280, 960),
            'hd': (1920, 1080),
            '4k': (3840, 2160)
        }
        
    async def process_image(self, image: Image.Image, operations: List[Dict[str, Any]]) -> Image.Image:
        """Apply a series of operations to an image"""
        processed_image = image.copy()
        
        for operation in operations:
            op_type = operation.get('type')
            params = operation.get('params', {})
            
            if op_type == 'resize':
                processed_image = await self.resize_image(processed_image, **params)
            elif op_type == 'crop':
                processed_image = await self.crop_image(processed_image, **params)
            elif op_type == 'filter':
                processed_image = await self.apply_filter(processed_image, **params)
            elif op_type == 'enhance':
                processed_image = await self.enhance_image(processed_image, **params)
            elif op_type == 'overlay':
                processed_image = await self.add_overlay(processed_image, **params)
            elif op_type == 'text':
                processed_image = await self.add_text(processed_image, **params)
                
        return processed_image
        
    async def resize_image(self, image: Image.Image, width: int, height: int, maintain_aspect: bool = True) -> Image.Image:
        """Resize image with optional aspect ratio maintenance"""
        if maintain_aspect:
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            return image
        else:
            return image.resize((width, height), Image.Resampling.LANCZOS)
            
    async def crop_image(self, image: Image.Image, x: int, y: int, width: int, height: int) -> Image.Image:
        """Crop image to specified dimensions"""
        return image.crop((x, y, x + width, y + height))
        
    async def smart_crop(self, image: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """Smart crop to maintain important image content"""
        if ImageOps:
            return ImageOps.fit(image, (target_width, target_height), Image.Resampling.LANCZOS)
        else:
            # Fallback to center crop
            return await self.center_crop(image, target_width, target_height)
            
    async def center_crop(self, image: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """Crop image from center"""
        width, height = image.size
        
        # Calculate crop box
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        right = left + target_width
        bottom = top + target_height
        
        return image.crop((left, top, right, bottom))
        
    async def apply_filter(self, image: Image.Image, filter_type: str, intensity: float = 1.0) -> Image.Image:
        """Apply various filters to image"""
        if not ImageFilter:
            return image
            
        filtered_image = image.copy()
        
        if filter_type == 'blur':
            filtered_image = filtered_image.filter(ImageFilter.GaussianBlur(radius=intensity * 5))
        elif filter_type == 'sharpen':
            filtered_image = filtered_image.filter(ImageFilter.SHARPEN)
        elif filter_type == 'edge_enhance':
            filtered_image = filtered_image.filter(ImageFilter.EDGE_ENHANCE)
        elif filter_type == 'smooth':
            filtered_image = filtered_image.filter(ImageFilter.SMOOTH)
        elif filter_type == 'contour':
            filtered_image = filtered_image.filter(ImageFilter.CONTOUR)
            
        return filtered_image
        
    async def enhance_image(self, image: Image.Image, enhancement_type: str, factor: float) -> Image.Image:
        """Enhance image properties"""
        if not ImageEnhance:
            return image
            
        enhanced_image = image.copy()
        
        if enhancement_type == 'brightness':
            enhancer = ImageEnhance.Brightness(enhanced_image)
            enhanced_image = enhancer.enhance(factor)
        elif enhancement_type == 'contrast':
            enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = enhancer.enhance(factor)
        elif enhancement_type == 'color':
            enhancer = ImageEnhance.Color(enhanced_image)
            enhanced_image = enhancer.enhance(factor)
        elif enhancement_type == 'sharpness':
            enhancer = ImageEnhance.Sharpness(enhanced_image)
            enhanced_image = enhancer.enhance(factor)
            
        return enhanced_image
        
    async def add_overlay(self, base_image: Image.Image, overlay_type: str, **kwargs) -> Image.Image:
        """Add overlay effects to image"""
        result_image = base_image.copy()
        
        if overlay_type == 'gradient':
            result_image = await self._add_gradient_overlay(result_image, **kwargs)
        elif overlay_type == 'vignette':
            result_image = await self._add_vignette(result_image, **kwargs)
        elif overlay_type == 'watermark':
            result_image = await self._add_watermark(result_image, **kwargs)
            
        return result_image
        
    async def _add_gradient_overlay(self, image: Image.Image, direction: str = 'vertical', opacity: float = 0.5) -> Image.Image:
        """Add gradient overlay"""
        if not Image:
            return image
            
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        if direction == 'vertical':
            for y in range(image.height):
                alpha = int(255 * opacity * (y / image.height))
                draw.line([(0, y), (image.width, y)], fill=(0, 0, 0, alpha))
        elif direction == 'horizontal':
            for x in range(image.width):
                alpha = int(255 * opacity * (x / image.width))
                draw.line([(x, 0), (x, image.height)], fill=(0, 0, 0, alpha))
                
        return Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        
    async def _add_vignette(self, image: Image.Image, intensity: float = 0.3) -> Image.Image:
        """Add vignette effect"""
        if not Image:
            return image
            
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Create radial gradient
        center_x, center_y = image.width // 2, image.height // 2
        max_radius = min(center_x, center_y)
        
        for i in range(max_radius):
            alpha = int(255 * intensity * (i / max_radius))
            draw.ellipse(
                [(center_x - max_radius + i, center_y - max_radius + i),
                 (center_x + max_radius - i, center_y + max_radius - i)],
                fill=(0, 0, 0, alpha)
            )
            
        return Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        
    async def _add_watermark(self, image: Image.Image, watermark_text: str = "", position: str = 'bottom-right') -> Image.Image:
        """Add watermark to image"""
        if not ImageDraw:
            return image
            
        watermarked = image.copy()
        draw = ImageDraw.Draw(watermarked)
        
        # Calculate text position
        margin = 20
        if position == 'bottom-right':
            x = image.width - margin - 100  # Approximate text width
            y = image.height - margin - 30   # Approximate text height
        elif position == 'bottom-left':
            x = margin
            y = image.height - margin - 30
        elif position == 'top-right':
            x = image.width - margin - 100
            y = margin
        else:  # top-left
            x = margin
            y = margin
            
        # Add watermark text
        draw.text((x, y), watermark_text, fill=(255, 255, 255, 128))
        
        return watermarked
        
    async def add_text(self, image: Image.Image, text: str, position: Tuple[int, int], 
                      font_size: int = 20, color: str = 'white', **kwargs) -> Image.Image:
        """Add text to image"""
        if not ImageDraw:
            return image
            
        text_image = image.copy()
        draw = ImageDraw.Draw(text_image)
        
        # Add text (using default font for simplicity)
        draw.text(position, text, fill=color)
        
        return text_image
        
    async def create_collage(self, images: List[Image.Image], layout: str = 'grid', spacing: int = 10) -> Image.Image:
        """Create a collage from multiple images"""
        if not images or not Image:
            return None
            
        if layout == 'grid':
            return await self._create_grid_collage(images, spacing)
        elif layout == 'horizontal':
            return await self._create_horizontal_collage(images, spacing)
        elif layout == 'vertical':
            return await self._create_vertical_collage(images, spacing)
            
    async def _create_grid_collage(self, images: List[Image.Image], spacing: int) -> Image.Image:
        """Create grid layout collage"""
        import math
        
        num_images = len(images)
        cols = math.ceil(math.sqrt(num_images))
        rows = math.ceil(num_images / cols)
        
        # Calculate cell size (use smallest image dimensions)
        cell_width = min(img.width for img in images)
        cell_height = min(img.height for img in images)
        
        # Create canvas
        canvas_width = cols * cell_width + (cols - 1) * spacing
        canvas_height = rows * cell_height + (rows - 1) * spacing
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        
        # Place images
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            x = col * (cell_width + spacing)
            y = row * (cell_height + spacing)
            
            # Resize image to fit cell
            resized = img.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
            canvas.paste(resized, (x, y))
            
        return canvas
        
    async def _create_horizontal_collage(self, images: List[Image.Image], spacing: int) -> Image.Image:
        """Create horizontal layout collage"""
        # Calculate canvas size
        total_width = sum(img.width for img in images) + spacing * (len(images) - 1)
        max_height = max(img.height for img in images)
        
        canvas = Image.new('RGB', (total_width, max_height), 'white')
        
        # Place images
        x_offset = 0
        for img in images:
            y_offset = (max_height - img.height) // 2  # Center vertically
            canvas.paste(img, (x_offset, y_offset))
            x_offset += img.width + spacing
            
        return canvas
        
    async def _create_vertical_collage(self, images: List[Image.Image], spacing: int) -> Image.Image:
        """Create vertical layout collage"""
        # Calculate canvas size
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images) + spacing * (len(images) - 1)
        
        canvas = Image.new('RGB', (max_width, total_height), 'white')
        
        # Place images
        y_offset = 0
        for img in images:
            x_offset = (max_width - img.width) // 2  # Center horizontally
            canvas.paste(img, (x_offset, y_offset))
            y_offset += img.height + spacing
            
        return canvas
        
    async def get_image_metadata(self, image: Image.Image) -> ImageMetadata:
        """Extract metadata from image"""
        return ImageMetadata(
            width=image.width,
            height=image.height,
            format=image.format or 'Unknown',
            mode=image.mode,
            file_size=0,  # Would be calculated from actual file
            has_transparency='A' in image.mode or 'transparency' in image.info,
            dominant_colors=await self._get_dominant_colors(image)
        )
        
    async def _get_dominant_colors(self, image: Image.Image, num_colors: int = 5) -> List[str]:
        """Get dominant colors from image"""
        # Simplified implementation
        # In production, would use color clustering algorithms
        return ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']
        
    async def optimize_for_web(self, image: Image.Image, max_size: int = 1920, quality: int = 85) -> Image.Image:
        """Optimize image for web delivery"""
        optimized = image.copy()
        
        # Resize if too large
        if optimized.width > max_size or optimized.height > max_size:
            optimized.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        # Convert to RGB if necessary
        if optimized.mode not in ('RGB', 'RGBA'):
            optimized = optimized.convert('RGB')
            
        return optimized
        
    async def create_thumbnail(self, image: Image.Image, size: Tuple[int, int] = (150, 150)) -> Image.Image:
        """Create thumbnail from image"""
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        return thumbnail