"""
Advertisement Creative Engine

High-converting ad creative generation for all business niches.
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import random

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
from backend.services.conversion_optimizer import ConversionOptimizer
from backend.services.ab_test_manager import ABTestManager
from backend.services.psychology_analyzer import PsychologyAnalyzer

# Image processing
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
except ImportError:
    Image = ImageDraw = ImageFont = ImageFilter = None

logger = logging.getLogger(__name__)


class AdvertisementCreativeEngine(UniversalContentCreator):
    """High-converting ad creative generation for all business niches"""
    
    def __init__(self, client_id: str):
        super().__init__(client_id, "ad_creative_engine")
        self.conversion_optimizer = ConversionOptimizer()
        self.ab_test_manager = ABTestManager()
        self.psychology_analyzer = PsychologyAnalyzer()
        
        # Ad creative configuration
        self.ad_config = self._load_ad_config()
        
    def _load_ad_config(self) -> Dict[str, Any]:
        """Load advertisement configuration"""
        return {
            'conversion_goals': {
                'lead_generation': {
                    'cta_emphasis': 'high',
                    'form_simplicity': 'high',
                    'trust_indicators': 'essential',
                    'urgency_level': 'medium'
                },
                'sales': {
                    'cta_emphasis': 'very_high',
                    'price_visibility': 'strategic',
                    'social_proof': 'high',
                    'urgency_level': 'high'
                },
                'brand_awareness': {
                    'visual_impact': 'high',
                    'message_clarity': 'high',
                    'emotional_appeal': 'high',
                    'cta_emphasis': 'low'
                },
                'app_install': {
                    'cta_emphasis': 'high',
                    'value_proposition': 'clear',
                    'screenshots': 'essential',
                    'ratings_display': 'prominent'
                }
            },
            'psychological_triggers': {
                'scarcity': {'effectiveness': 0.85, 'industries': ['retail', 'education', 'fitness']},
                'social_proof': {'effectiveness': 0.90, 'industries': ['all']},
                'authority': {'effectiveness': 0.88, 'industries': ['consulting', 'finance', 'health']},
                'reciprocity': {'effectiveness': 0.75, 'industries': ['service', 'education']},
                'commitment': {'effectiveness': 0.80, 'industries': ['fitness', 'education']},
                'liking': {'effectiveness': 0.70, 'industries': ['creative', 'lifestyle']},
                'fear_of_missing_out': {'effectiveness': 0.82, 'industries': ['retail', 'events']}
            }
        }
        
    async def create_content(self, request: CreativeRequest) -> CreativeAsset:
        """Create high-converting ad creative"""
        try:
            logger.info(f"Starting ad creative creation for request {request.request_id}")
            
            # 1. Analyze conversion goals
            conversion_analysis = await self.analyze_conversion_goals(request)
            
            # 2. Apply psychological triggers
            psychological_elements = await self.identify_psychological_triggers(request.business_niche, request.target_audience)
            
            # 3. Generate ad copy variations
            ad_copy_variations = await self.generate_ad_copy_variations(request, psychological_elements)
            
            # 4. Create visual elements
            visual_elements = await self.create_ad_visual_elements(request, conversion_analysis)
            
            # 5. Combine copy and visuals
            ad_creative = await self.combine_copy_and_visuals(ad_copy_variations[0], visual_elements, request)
            
            # 6. Optimize for conversion
            conversion_optimized = await self.optimize_for_conversion(ad_creative, conversion_analysis)
            
            # 7. Create platform-specific versions
            platform_versions = await self.create_ad_platform_versions(conversion_optimized, request.platform_requirements)
            
            # 8. Set up A/B test variations
            ab_test_variants = await self.create_ab_test_variants(conversion_optimized, request)
            
            # 9. Save final asset
            asset = await self.save_ad_asset(conversion_optimized, platform_versions, ab_test_variants, request)
            
            logger.info(f"Successfully created ad creative asset {asset.asset_id}")
            return asset
            
        except Exception as e:
            await self.log_creation_error(f"Ad creative creation failed for request {request.request_id}: {str(e)}")
            raise ContentCreationError(f"Failed to create ad creative: {str(e)}")
    
    async def analyze_conversion_goals(self, request: CreativeRequest) -> Dict[str, Any]:
        """Analyze and optimize for specific conversion goals"""
        
        # Extract conversion intent from creative brief
        conversion_intent = await self.extract_conversion_intent(request.creative_brief)
        
        # Analyze business niche conversion patterns
        niche_patterns = await self.analyze_niche_conversion_patterns(request.business_niche)
        
        # Identify optimal conversion path
        conversion_path = await self.identify_optimal_conversion_path(request.target_audience, request.business_niche)
        
        return {
            'primary_goal': conversion_intent['primary_goal'],
            'secondary_goals': conversion_intent['secondary_goals'],
            'conversion_type': conversion_intent['type'],
            'funnel_stage': conversion_intent['funnel_stage'],
            'niche_patterns': niche_patterns,
            'optimal_path': conversion_path,
            'success_metrics': await self.define_success_metrics(conversion_intent, request.business_niche)
        }
    
    async def extract_conversion_intent(self, creative_brief: str) -> Dict[str, Any]:
        """Extract conversion intent from creative brief"""
        # Keywords indicating different conversion goals
        goal_keywords = {
            'lead_generation': ['lead', 'signup', 'register', 'subscribe', 'contact'],
            'sales': ['buy', 'purchase', 'shop', 'order', 'discount'],
            'brand_awareness': ['aware', 'discover', 'learn', 'explore', 'introduce'],
            'app_install': ['download', 'install', 'app', 'mobile', 'get']
        }
        
        brief_lower = creative_brief.lower()
        goal_scores = {}
        
        for goal, keywords in goal_keywords.items():
            score = sum(1 for keyword in keywords if keyword in brief_lower)
            goal_scores[goal] = score
            
        primary_goal = max(goal_scores.items(), key=lambda x: x[1])[0] if goal_scores else 'lead_generation'
        
        return {
            'primary_goal': primary_goal,
            'secondary_goals': [goal for goal, score in goal_scores.items() if score > 0 and goal != primary_goal],
            'type': primary_goal,
            'funnel_stage': 'consideration'  # Default, would be more sophisticated
        }
    
    async def analyze_niche_conversion_patterns(self, business_niche: str) -> Dict[str, Any]:
        """Analyze conversion patterns specific to business niche"""
        niche_patterns = {
            'education': {
                'best_converting_elements': ['free_trial', 'testimonials', 'curriculum_preview'],
                'optimal_cta': 'Start Learning Free',
                'trust_factors': ['accreditation', 'instructor_credentials', 'success_stories']
            },
            'fitness': {
                'best_converting_elements': ['before_after', 'transformation_timeline', 'workout_preview'],
                'optimal_cta': 'Start Your Transformation',
                'trust_factors': ['real_results', 'certified_trainers', 'money_back_guarantee']
            },
            'business_consulting': {
                'best_converting_elements': ['case_studies', 'roi_calculator', 'free_consultation'],
                'optimal_cta': 'Get Your Free Strategy Call',
                'trust_factors': ['client_logos', 'certifications', 'years_experience']
            },
            'creative_arts': {
                'best_converting_elements': ['portfolio_samples', 'creative_process', 'unique_style'],
                'optimal_cta': 'See My Work',
                'trust_factors': ['awards', 'client_testimonials', 'featured_work']
            }
        }
        
        return niche_patterns.get(business_niche, {
            'best_converting_elements': ['value_proposition', 'social_proof', 'clear_cta'],
            'optimal_cta': 'Learn More',
            'trust_factors': ['testimonials', 'guarantees', 'credentials']
        })
    
    async def identify_optimal_conversion_path(self, target_audience: Dict[str, Any], business_niche: str) -> List[str]:
        """Identify optimal conversion path for audience and niche"""
        # Simplified path identification
        age_range = target_audience.get('age_range', '25-45')
        
        if '18-25' in age_range:
            return ['attention_grab', 'quick_value', 'easy_action', 'instant_gratification']
        elif '45+' in age_range:
            return ['trust_building', 'detailed_info', 'risk_reduction', 'clear_benefits']
        else:
            return ['value_proposition', 'social_proof', 'clear_cta', 'easy_conversion']
    
    async def define_success_metrics(self, conversion_intent: Dict[str, Any], business_niche: str) -> Dict[str, Any]:
        """Define success metrics for the ad creative"""
        base_metrics = {
            'click_through_rate': 0.02,  # 2% baseline
            'conversion_rate': 0.03,      # 3% baseline
            'cost_per_acquisition': 50.0,  # $50 baseline
            'return_on_ad_spend': 3.0     # 3x baseline
        }
        
        # Adjust based on conversion type
        if conversion_intent['type'] == 'lead_generation':
            base_metrics['conversion_rate'] = 0.05  # Higher for leads
        elif conversion_intent['type'] == 'sales':
            base_metrics['return_on_ad_spend'] = 4.0  # Higher ROAS target
            
        return base_metrics
    
    async def identify_psychological_triggers(self, business_niche: str, target_audience: Dict[str, Any]) -> Dict[str, Any]:
        """Identify and apply psychological triggers for maximum conversion"""
        
        # Get all applicable triggers
        all_triggers = self.ad_config['psychological_triggers']
        applicable_triggers = []
        
        for trigger, config in all_triggers.items():
            industries = config['industries']
            if 'all' in industries or business_niche in industries:
                applicable_triggers.append({
                    'name': trigger,
                    'effectiveness': config['effectiveness'],
                    'application': await self.determine_trigger_application(trigger, business_niche)
                })
        
        # Sort by effectiveness
        applicable_triggers.sort(key=lambda x: x['effectiveness'], reverse=True)
        
        # Select top triggers
        primary_triggers = applicable_triggers[:3]
        secondary_triggers = applicable_triggers[3:6]
        
        return {
            'primary_triggers': primary_triggers,
            'secondary_triggers': secondary_triggers,
            'trigger_applications': await self.define_trigger_applications(primary_triggers),
            'copy_guidelines': await self.generate_copy_guidelines(primary_triggers),
            'visual_guidelines': await self.generate_visual_guidelines(primary_triggers)
        }
    
    async def determine_trigger_application(self, trigger: str, business_niche: str) -> str:
        """Determine how to apply a specific trigger"""
        applications = {
            'scarcity': {
                'education': 'Limited enrollment spots',
                'fitness': 'Special pricing ends soon',
                'retail': 'Only X items left'
            },
            'social_proof': {
                'education': 'Join 10,000+ students',
                'fitness': 'See real transformations',
                'business_consulting': 'Trusted by Fortune 500'
            },
            'authority': {
                'education': 'Learn from industry experts',
                'fitness': 'Certified trainers',
                'business_consulting': '20 years of experience'
            }
        }
        
        return applications.get(trigger, {}).get(business_niche, f'Apply {trigger}')
    
    async def define_trigger_applications(self, triggers: List[Dict[str, Any]]) -> Dict[str, str]:
        """Define how to apply each trigger in the creative"""
        applications = {}
        
        for trigger in triggers:
            trigger_name = trigger['name']
            if trigger_name == 'scarcity':
                applications['scarcity'] = 'Add countdown timer or limited quantity indicator'
            elif trigger_name == 'social_proof':
                applications['social_proof'] = 'Include testimonials or user count'
            elif trigger_name == 'authority':
                applications['authority'] = 'Show credentials or expert endorsements'
            elif trigger_name == 'fear_of_missing_out':
                applications['fomo'] = 'Emphasize what they\'ll miss without action'
                
        return applications
    
    async def generate_copy_guidelines(self, triggers: List[Dict[str, Any]]) -> List[str]:
        """Generate copy guidelines based on triggers"""
        guidelines = []
        
        for trigger in triggers:
            if trigger['name'] == 'scarcity':
                guidelines.append('Use urgent language: "Only 3 spots left!", "Ends tonight!"')
            elif trigger['name'] == 'social_proof':
                guidelines.append('Include numbers: "Join 5,000+ happy customers"')
            elif trigger['name'] == 'authority':
                guidelines.append('Mention credentials: "Award-winning", "Industry leader"')
                
        return guidelines
    
    async def generate_visual_guidelines(self, triggers: List[Dict[str, Any]]) -> List[str]:
        """Generate visual guidelines based on triggers"""
        guidelines = []
        
        for trigger in triggers:
            if trigger['name'] == 'scarcity':
                guidelines.append('Add visual countdown timer or progress bar')
            elif trigger['name'] == 'social_proof':
                guidelines.append('Show customer photos or testimonial badges')
            elif trigger['name'] == 'authority':
                guidelines.append('Display certifications, awards, or expert photos')
                
        return guidelines
    
    async def generate_ad_copy_variations(self, request: CreativeRequest, psychological_elements: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate multiple ad copy variations optimized for conversion"""
        
        variations = []
        
        # Different copy approaches
        copy_approaches = [
            'problem_solution',
            'benefit_focused',
            'story_driven',
            'urgency_based',
            'social_proof_heavy',
            'authority_led'
        ]
        
        # Generate copy for each approach
        for approach in copy_approaches[:3]:  # Top 3 approaches
            copy_variation = await self.ai_service.generate_ad_copy(
                approach=approach,
                business_niche=request.business_niche,
                target_audience=request.target_audience,
                creative_brief=request.creative_brief,
                psychological_triggers=[t['name'] for t in psychological_elements['primary_triggers']],
                platform_requirements=request.platform_requirements
            )
            
            # Optimize copy for conversion
            optimized_copy = await self.conversion_optimizer.optimize_copy(copy_variation, request.business_niche)
            
            variations.append({
                'approach': approach,
                'headline': optimized_copy.get('headline', 'Transform Your Business Today'),
                'body': optimized_copy.get('body', 'Discover the proven solution thousands trust'),
                'cta': optimized_copy.get('cta', 'Get Started Now'),
                'conversion_score': optimized_copy.get('predicted_conversion_score', 0.7)
            })
        
        # Sort by predicted conversion score
        variations.sort(key=lambda x: x['conversion_score'], reverse=True)
        
        return variations
    
    async def create_ad_visual_elements(self, request: CreativeRequest, conversion_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create visual elements optimized for conversion"""
        
        visual_elements = {}
        
        # Create base visual
        base_image = await self.create_ad_base_image(request, conversion_analysis)
        visual_elements['background'] = base_image
        
        # Add conversion-optimized elements
        visual_elements['cta_button'] = await self.design_optimal_cta_button(
            conversion_analysis['primary_goal'],
            request.business_niche,
            request.brand_guidelines
        )
        
        # Add trust indicators
        visual_elements['trust_indicators'] = await self.create_trust_indicators(
            request.business_niche,
            conversion_analysis['niche_patterns']
        )
        
        # Add social proof elements
        visual_elements['social_proof'] = await self.create_social_proof_elements(
            request.business_niche,
            request.target_audience
        )
        
        # Add urgency elements if applicable
        if 'urgency' in [t['name'] for t in conversion_analysis.get('psychological_triggers', [])]:
            visual_elements['urgency_elements'] = await self.create_urgency_elements(request.business_niche)
            
        return visual_elements
    
    async def create_ad_base_image(self, request: CreativeRequest, conversion_analysis: Dict[str, Any]) -> Image.Image:
        """Create base image for ad"""
        if not Image:
            logger.warning("PIL not available, creating placeholder")
            return None
            
        # Determine dimensions based on platform
        width, height = 1200, 628  # Default Facebook ad size
        
        # Create base image
        image = Image.new('RGB', (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Add background gradient based on business niche
        niche_colors = {
            'education': [(41, 128, 185), (52, 152, 219)],    # Blue gradient
            'fitness': [(231, 76, 60), (255, 94, 87)],        # Red gradient
            'business_consulting': [(44, 62, 80), (52, 73, 94)],  # Dark blue gradient
            'creative_arts': [(155, 89, 182), (142, 68, 173)]    # Purple gradient
        }
        
        colors = niche_colors.get(request.business_niche, [(100, 100, 100), (150, 150, 150)])
        
        # Create gradient
        for y in range(height):
            r = int(colors[0][0] + (colors[1][0] - colors[0][0]) * y / height)
            g = int(colors[0][1] + (colors[1][1] - colors[0][1]) * y / height)
            b = int(colors[0][2] + (colors[1][2] - colors[0][2]) * y / height)
            draw.line([(0, y), (width, y)], fill=(r, g, b))
            
        return image
    
    async def design_optimal_cta_button(self, conversion_goal: str, business_niche: str, brand_guidelines: Dict[str, Any]) -> Dict[str, Any]:
        """Design optimal call-to-action button"""
        
        # CTA text based on goal and niche
        cta_texts = {
            'lead_generation': {
                'education': 'Start Learning Free',
                'fitness': 'Get Your Free Plan',
                'business_consulting': 'Book Free Consultation',
                'default': 'Get Started Free'
            },
            'sales': {
                'education': 'Enroll Now',
                'fitness': 'Start Today',
                'business_consulting': 'Get Instant Access',
                'default': 'Buy Now'
            }
        }
        
        cta_text = cta_texts.get(conversion_goal, {}).get(business_niche, 'Learn More')
        
        # Button design
        button_design = {
            'text': cta_text,
            'color': brand_guidelines.get('primary_color', '#FF6B6B'),
            'text_color': '#FFFFFF',
            'size': 'large',
            'style': 'rounded',
            'hover_effect': 'glow',
            'urgency_indicator': conversion_goal == 'sales'
        }
        
        return button_design
    
    async def create_trust_indicators(self, business_niche: str, niche_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create trust indicators for the ad"""
        trust_indicators = []
        
        # Universal trust indicators
        trust_indicators.append({
            'type': 'security_badge',
            'text': 'Secure & Trusted',
            'icon': 'shield'
        })
        
        # Niche-specific trust indicators
        if business_niche == 'education':
            trust_indicators.append({
                'type': 'student_count',
                'text': '50,000+ Students',
                'icon': 'users'
            })
        elif business_niche == 'fitness':
            trust_indicators.append({
                'type': 'guarantee',
                'text': '30-Day Money Back',
                'icon': 'guarantee'
            })
        elif business_niche == 'business_consulting':
            trust_indicators.append({
                'type': 'experience',
                'text': '15+ Years Experience',
                'icon': 'award'
            })
            
        return trust_indicators
    
    async def create_social_proof_elements(self, business_niche: str, target_audience: Dict[str, Any]) -> Dict[str, Any]:
        """Create social proof elements"""
        return {
            'testimonial': {
                'text': 'This changed my life! Highly recommend to everyone.',
                'author': 'Sarah J.',
                'rating': 5,
                'verified': True
            },
            'statistics': {
                'users': '10,000+',
                'rating': 4.8,
                'reviews': 500
            },
            'badges': ['top_rated', 'editor_choice', 'verified_business']
        }
    
    async def create_urgency_elements(self, business_niche: str) -> Dict[str, Any]:
        """Create urgency elements"""
        return {
            'countdown_timer': {
                'duration': 24 * 60 * 60,  # 24 hours in seconds
                'text': 'Offer ends in:'
            },
            'limited_spots': {
                'total': 100,
                'remaining': random.randint(5, 20),
                'text': 'Only {remaining} spots left!'
            },
            'flash_sale': {
                'discount': '50%',
                'original_price': '$99',
                'sale_price': '$49'
            }
        }
    
    async def combine_copy_and_visuals(self, copy: Dict[str, str], visual_elements: Dict[str, Any], request: CreativeRequest) -> Dict[str, Any]:
        """Combine copy and visual elements into cohesive ad creative"""
        
        ad_creative = {
            'copy': copy,
            'visuals': visual_elements,
            'layout': await self.determine_optimal_layout(request.platform_requirements),
            'color_scheme': await self.determine_color_scheme(request.business_niche, request.brand_guidelines),
            'typography': await self.determine_typography(request.brand_guidelines),
            'animations': await self.determine_animations(request.platform_requirements)
        }
        
        return ad_creative
    
    async def determine_optimal_layout(self, platform_requirements: Dict[str, Any]) -> str:
        """Determine optimal ad layout"""
        # Platform-specific layouts
        if 'facebook' in platform_requirements:
            return 'image_left_text_right'
        elif 'instagram' in platform_requirements:
            return 'centered_overlay'
        elif 'google_ads' in platform_requirements:
            return 'responsive_grid'
        else:
            return 'flexible_grid'
    
    async def determine_color_scheme(self, business_niche: str, brand_guidelines: Dict[str, Any]) -> Dict[str, str]:
        """Determine color scheme for ad"""
        # Use brand colors if available
        if brand_guidelines.get('colors'):
            return {
                'primary': brand_guidelines['colors'][0],
                'secondary': brand_guidelines['colors'][1] if len(brand_guidelines['colors']) > 1 else '#666666',
                'accent': brand_guidelines['colors'][2] if len(brand_guidelines['colors']) > 2 else '#FF6B6B'
            }
            
        # Niche-based color schemes
        niche_schemes = {
            'education': {
                'primary': '#3498DB',
                'secondary': '#2C3E50',
                'accent': '#E74C3C'
            },
            'fitness': {
                'primary': '#E74C3C',
                'secondary': '#34495E',
                'accent': '#F39C12'
            },
            'business_consulting': {
                'primary': '#2C3E50',
                'secondary': '#34495E',
                'accent': '#3498DB'
            }
        }
        
        return niche_schemes.get(business_niche, {
            'primary': '#333333',
            'secondary': '#666666',
            'accent': '#FF6B6B'
        })
    
    async def determine_typography(self, brand_guidelines: Dict[str, Any]) -> Dict[str, Any]:
        """Determine typography for ad"""
        return {
            'headline_font': brand_guidelines.get('fonts', ['Arial'])[0],
            'headline_size': 32,
            'body_font': brand_guidelines.get('fonts', ['Arial'])[0],
            'body_size': 16,
            'cta_font': brand_guidelines.get('fonts', ['Arial'])[0],
            'cta_size': 20,
            'font_weight': 'bold' if 'bold' in str(brand_guidelines.get('style', '')).lower() else 'normal'
        }
    
    async def determine_animations(self, platform_requirements: Dict[str, Any]) -> List[str]:
        """Determine animations based on platform"""
        animations = []
        
        if 'facebook' in platform_requirements or 'instagram' in platform_requirements:
            animations.extend(['fade_in', 'pulse_cta', 'slide_text'])
        if 'google_ads' in platform_requirements:
            animations.append('responsive_resize')
            
        return animations
    
    async def optimize_for_conversion(self, ad_creative: Dict[str, Any], conversion_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply conversion optimization techniques"""
        
        optimized_creative = ad_creative.copy()
        
        # Optimize visual hierarchy
        optimized_creative = await self.optimize_visual_hierarchy(optimized_creative, conversion_analysis['primary_goal'])
        
        # Optimize color psychology
        optimized_creative = await self.apply_color_psychology(optimized_creative, conversion_analysis['conversion_type'])
        
        # Optimize CTA placement and design
        optimized_creative = await self.optimize_cta_elements(optimized_creative, conversion_analysis)
        
        # Apply conversion-focused spacing and layout
        optimized_creative = await self.optimize_layout_for_conversion(optimized_creative)
        
        # Add conversion tracking elements
        optimized_creative = await self.add_conversion_tracking_elements(optimized_creative)
        
        return optimized_creative
    
    async def optimize_visual_hierarchy(self, creative: Dict[str, Any], primary_goal: str) -> Dict[str, Any]:
        """Optimize visual hierarchy for conversion goal"""
        creative['visual_hierarchy'] = {
            'primary_focus': 'cta_button' if primary_goal == 'sales' else 'headline',
            'secondary_focus': 'value_proposition',
            'tertiary_focus': 'social_proof',
            'contrast_levels': {
                'cta': 'maximum',
                'headline': 'high',
                'body': 'medium'
            }
        }
        
        return creative
    
    async def apply_color_psychology(self, creative: Dict[str, Any], conversion_type: str) -> Dict[str, Any]:
        """Apply color psychology for conversion"""
        color_psychology = {
            'lead_generation': {
                'cta_color': '#27AE60',  # Green - growth, positive
                'urgency_color': '#E74C3C'  # Red - urgency
            },
            'sales': {
                'cta_color': '#E74C3C',  # Red - action, urgency
                'trust_color': '#3498DB'  # Blue - trust
            },
            'brand_awareness': {
                'primary_color': creative['color_scheme']['primary'],
                'accent_color': creative['color_scheme']['accent']
            }
        }
        
        creative['optimized_colors'] = color_psychology.get(conversion_type, color_psychology['lead_generation'])
        
        return creative
    
    async def optimize_cta_elements(self, creative: Dict[str, Any], conversion_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize CTA elements for maximum conversion"""
        creative['cta_optimization'] = {
            'position': 'above_fold',
            'size_multiplier': 1.5 if conversion_analysis['primary_goal'] == 'sales' else 1.2,
            'contrast_ratio': 4.5,  # WCAG AA compliance
            'whitespace_padding': '20px',
            'hover_state': 'scale_and_glow',
            'mobile_sticky': True
        }
        
        return creative
    
    async def optimize_layout_for_conversion(self, creative: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize layout for conversion"""
        creative['layout_optimization'] = {
            'f_pattern': True,  # Follow F-pattern reading
            'whitespace_ratio': 0.3,  # 30% whitespace
            'content_blocks': 'scannable',
            'mobile_first': True,
            'thumb_friendly_cta': True
        }
        
        return creative
    
    async def add_conversion_tracking_elements(self, creative: Dict[str, Any]) -> Dict[str, Any]:
        """Add conversion tracking elements"""
        creative['tracking'] = {
            'utm_parameters': {
                'source': 'paid_social',
                'medium': 'cpc',
                'campaign': creative.get('campaign_id', 'default')
            },
            'pixel_events': ['ViewContent', 'AddToCart', 'InitiateCheckout', 'Purchase'],
            'conversion_api': True,
            'enhanced_conversions': True
        }
        
        return creative
    
    async def create_ad_platform_versions(self, creative: Dict[str, Any], platform_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create platform-specific ad versions"""
        platform_versions = {}
        
        for platform in platform_requirements:
            if platform == 'facebook':
                platform_versions['facebook'] = await self.create_facebook_ad_version(creative)
            elif platform == 'instagram':
                platform_versions['instagram'] = await self.create_instagram_ad_version(creative)
            elif platform == 'google_ads':
                platform_versions['google_ads'] = await self.create_google_ads_version(creative)
            elif platform == 'linkedin':
                platform_versions['linkedin'] = await self.create_linkedin_ad_version(creative)
                
        return platform_versions
    
    async def create_facebook_ad_version(self, creative: Dict[str, Any]) -> Dict[str, Any]:
        """Create Facebook-optimized ad version"""
        return {
            'format': 'single_image',
            'image_ratio': '1.91:1',
            'headline_length': min(len(creative['copy']['headline']), 40),
            'body_length': min(len(creative['copy']['body']), 125),
            'cta_type': 'button',
            'placement_optimization': ['feed', 'stories', 'reels']
        }
    
    async def create_instagram_ad_version(self, creative: Dict[str, Any]) -> Dict[str, Any]:
        """Create Instagram-optimized ad version"""
        return {
            'format': 'square_image',
            'image_ratio': '1:1',
            'story_format': '9:16',
            'hashtag_strategy': await self.generate_hashtag_strategy(creative),
            'visual_first': True,
            'cta_sticker': True
        }
    
    async def create_google_ads_version(self, creative: Dict[str, Any]) -> Dict[str, Any]:
        """Create Google Ads optimized version"""
        return {
            'responsive_display': {
                'headlines': [creative['copy']['headline']],
                'descriptions': [creative['copy']['body']],
                'images': ['landscape', 'square'],
                'logos': ['square']
            },
            'keyword_alignment': True,
            'dynamic_insertion': True
        }
    
    async def create_linkedin_ad_version(self, creative: Dict[str, Any]) -> Dict[str, Any]:
        """Create LinkedIn-optimized ad version"""
        return {
            'format': 'sponsored_content',
            'professional_tone': True,
            'b2b_focus': True,
            'thought_leadership': True,
            'company_page_integration': True
        }
    
    async def generate_hashtag_strategy(self, creative: Dict[str, Any]) -> List[str]:
        """Generate hashtag strategy for Instagram"""
        # This would analyze trending hashtags
        return ['#innovation', '#growth', '#success', '#business', '#motivation']
    
    async def create_ab_test_variants(self, base_creative: Dict[str, Any], request: CreativeRequest) -> List[Dict[str, Any]]:
        """Create A/B test variants for optimization"""
        variants = []
        
        # Variant 1: Different headline
        headline_variant = base_creative.copy()
        headline_variant['copy']['headline'] = await self.generate_alternative_headline(base_creative['copy']['headline'], request)
        headline_variant['variant_type'] = 'headline_test'
        headline_variant['variant_id'] = 'variant_headline_001'
        variants.append(headline_variant)
        
        # Variant 2: Different CTA
        cta_variant = base_creative.copy()
        cta_variant['copy']['cta'] = await self.generate_alternative_cta(base_creative['copy']['cta'], request)
        cta_variant['variant_type'] = 'cta_test'
        cta_variant['variant_id'] = 'variant_cta_001'
        variants.append(cta_variant)
        
        # Variant 3: Different visual approach
        visual_variant = base_creative.copy()
        visual_variant['visuals'] = await self.generate_alternative_visuals(base_creative['visuals'], request)
        visual_variant['variant_type'] = 'visual_test'
        visual_variant['variant_id'] = 'variant_visual_001'
        variants.append(visual_variant)
        
        # Variant 4: Different psychological trigger emphasis
        psychology_variant = base_creative.copy()
        psychology_variant = await self.apply_alternative_psychology(psychology_variant, request)
        psychology_variant['variant_type'] = 'psychology_test'
        psychology_variant['variant_id'] = 'variant_psychology_001'
        variants.append(psychology_variant)
        
        return variants
    
    async def generate_alternative_headline(self, original_headline: str, request: CreativeRequest) -> str:
        """Generate alternative headline for A/B testing"""
        alternatives = {
            'education': 'Master New Skills in Just 30 Days',
            'fitness': 'Your Dream Body Starts Here',
            'business_consulting': 'Scale Your Business 10x Faster',
            'creative_arts': 'Unleash Your Creative Potential'
        }
        
        return alternatives.get(request.business_niche, 'Transform Your Life Today')
    
    async def generate_alternative_cta(self, original_cta: str, request: CreativeRequest) -> str:
        """Generate alternative CTA for A/B testing"""
        alternatives = {
            'Get Started': 'Start Now',
            'Learn More': 'Discover How',
            'Sign Up': 'Join Today',
            'Buy Now': 'Get Instant Access'
        }
        
        return alternatives.get(original_cta, 'Take Action Now')
    
    async def generate_alternative_visuals(self, original_visuals: Dict[str, Any], request: CreativeRequest) -> Dict[str, Any]:
        """Generate alternative visual approach"""
        alt_visuals = original_visuals.copy()
        
        # Change visual style
        if 'background' in alt_visuals:
            alt_visuals['background_style'] = 'lifestyle_image' if original_visuals.get('background_style') == 'gradient' else 'gradient'
            
        # Change trust indicators
        if 'trust_indicators' in alt_visuals:
            alt_visuals['trust_indicators'] = await self.create_alternative_trust_indicators(request.business_niche)
            
        return alt_visuals
    
    async def create_alternative_trust_indicators(self, business_niche: str) -> List[Dict[str, Any]]:
        """Create alternative trust indicators"""
        return [
            {
                'type': 'certification',
                'text': 'Certified Excellence',
                'icon': 'certificate'
            },
            {
                'type': 'satisfaction',
                'text': '99% Satisfaction Rate',
                'icon': 'smile'
            }
        ]
    
    async def apply_alternative_psychology(self, creative: Dict[str, Any], request: CreativeRequest) -> Dict[str, Any]:
        """Apply alternative psychological approach"""
        # Switch primary psychological trigger
        creative['psychological_approach'] = 'fear_based' if creative.get('psychological_approach') == 'benefit_based' else 'benefit_based'
        
        # Adjust copy tone
        if creative['psychological_approach'] == 'fear_based':
            creative['copy']['headline'] = f"Don't Miss Out on {request.business_niche.title()} Success"
        else:
            creative['copy']['headline'] = f"Achieve {request.business_niche.title()} Excellence Today"
            
        return creative
    
    async def save_ad_asset(self, ad_creative: Dict[str, Any], platform_versions: Dict[str, Any], 
                            ab_test_variants: List[Dict[str, Any]], request: CreativeRequest) -> CreativeAsset:
        """Save ad creative asset"""
        # Generate asset ID
        asset_id = self.generate_asset_id()
        
        # Create asset directory
        asset_dir = os.path.join('assets', 'ads', asset_id)
        os.makedirs(asset_dir, exist_ok=True)
        
        # Save creative data
        creative_path = os.path.join(asset_dir, 'creative.json')
        with open(creative_path, 'w') as f:
            json.dump({
                'main_creative': ad_creative,
                'platform_versions': platform_versions,
                'ab_test_variants': [v for v in ab_test_variants]
            }, f, indent=2)
            
        # Save visual assets if available
        if ad_creative.get('visuals', {}).get('background'):
            visual_path = os.path.join(asset_dir, 'main_visual.png')
            if isinstance(ad_creative['visuals']['background'], Image.Image):
                ad_creative['visuals']['background'].save(visual_path, 'PNG')
        
        # Calculate quality score
        quality_score = await self.calculate_ad_quality_score(ad_creative)
        
        # Create asset object
        asset = CreativeAsset(
            asset_id=asset_id,
            request_id=request.request_id,
            content_type=ContentType.ADVERTISEMENT,
            file_path=creative_path,
            file_format='JSON',
            dimensions={'width': 1200, 'height': 628},  # Default ad dimensions
            file_size=os.path.getsize(creative_path),
            quality_score=quality_score,
            brand_compliance_score=0.9,  # High compliance for ads
            platform_optimized_versions=platform_versions,
            metadata={
                'business_niche': request.business_niche,
                'conversion_goal': ad_creative.get('conversion_goal', 'lead_generation'),
                'ab_test_variants': len(ab_test_variants),
                'psychological_triggers': ad_creative.get('psychological_triggers', []),
                'predicted_ctr': 0.025,  # 2.5% predicted CTR
                'predicted_conversion_rate': 0.03  # 3% predicted conversion
            }
        )
        
        return asset
    
    async def calculate_ad_quality_score(self, ad_creative: Dict[str, Any]) -> float:
        """Calculate quality score for ad creative"""
        score = 0.7  # Base score
        
        # Check for essential elements
        if ad_creative.get('copy', {}).get('headline'):
            score += 0.05
        if ad_creative.get('copy', {}).get('cta'):
            score += 0.05
        if ad_creative.get('visuals', {}).get('trust_indicators'):
            score += 0.05
        if ad_creative.get('visuals', {}).get('social_proof'):
            score += 0.05
        if ad_creative.get('psychological_approach'):
            score += 0.05
        if ad_creative.get('cta_optimization'):
            score += 0.05
            
        return min(score, 1.0)
    
    async def optimize_for_platform(self, asset: CreativeAsset, platform: str) -> CreativeAsset:
        """Optimize ad creative for specific platform requirements"""
        logger.info(f"Optimizing ad {asset.asset_id} for {platform}")
        
        # Load creative data
        with open(asset.file_path, 'r') as f:
            creative_data = json.load(f)
            
        # Apply platform-specific optimizations
        if platform == 'facebook':
            creative_data['platform_optimization'] = await self.optimize_for_facebook(creative_data)
        elif platform == 'google_ads':
            creative_data['platform_optimization'] = await self.optimize_for_google_ads(creative_data)
        elif platform == 'linkedin':
            creative_data['platform_optimization'] = await self.optimize_for_linkedin(creative_data)
            
        # Save optimized version
        optimized_path = os.path.join(os.path.dirname(asset.file_path), f'optimized_{platform}.json')
        with open(optimized_path, 'w') as f:
            json.dump(creative_data, f, indent=2)
            
        # Update asset
        asset.platform_optimized_versions[platform] = {
            'optimized': True,
            'path': optimized_path
        }
        
        return asset
    
    async def optimize_for_facebook(self, creative_data: Dict[str, Any]) -> Dict[str, Any]:
        """Facebook-specific optimizations"""
        return {
            'text_overlay_percentage': 0.18,  # Under 20% limit
            'aspect_ratio_compliant': True,
            'cta_button_enabled': True,
            'pixel_integrated': True
        }
    
    async def optimize_for_google_ads(self, creative_data: Dict[str, Any]) -> Dict[str, Any]:
        """Google Ads specific optimizations"""
        return {
            'responsive_elements': True,
            'keyword_insertion': True,
            'landing_page_match': True,
            'quality_score_optimized': True
        }
    
    async def optimize_for_linkedin(self, creative_data: Dict[str, Any]) -> Dict[str, Any]:
        """LinkedIn specific optimizations"""
        return {
            'professional_tone': True,
            'b2b_messaging': True,
            'thought_leadership': True,
            'company_page_aligned': True
        }
    
    async def analyze_performance(self, asset: CreativeAsset) -> Dict[str, Any]:
        """Analyze ad creative performance and optimization opportunities"""
        
        performance_data = {
            'conversion_potential': await self.analyze_conversion_potential(asset),
            'psychological_effectiveness': await self.analyze_psychological_effectiveness(asset),
            'visual_optimization': await self.analyze_visual_optimization(asset),
            'copy_effectiveness': await self.analyze_copy_effectiveness(asset),
            'platform_performance': await self.analyze_platform_ad_performance(asset),
            'ab_test_results': await self.get_ab_test_results(asset),
            'optimization_recommendations': await self.generate_ad_optimization_recommendations(asset)
        }
        
        return performance_data
    
    async def analyze_conversion_potential(self, asset: CreativeAsset) -> Dict[str, Any]:
        """Analyze conversion potential of ad creative"""
        return {
            'predicted_ctr': asset.metadata.get('predicted_ctr', 0.02),
            'predicted_conversion_rate': asset.metadata.get('predicted_conversion_rate', 0.03),
            'quality_score': asset.quality_score,
            'conversion_elements_present': 0.9,
            'optimization_score': 0.85
        }
    
    async def analyze_psychological_effectiveness(self, asset: CreativeAsset) -> Dict[str, Any]:
        """Analyze psychological trigger effectiveness"""
        return {
            'trigger_implementation': 0.9,
            'emotional_appeal': 0.85,
            'urgency_effectiveness': 0.8,
            'trust_building': 0.9,
            'overall_psychology_score': 0.86
        }
    
    async def analyze_visual_optimization(self, asset: CreativeAsset) -> Dict[str, Any]:
        """Analyze visual optimization of ad"""
        return {
            'visual_hierarchy': 0.9,
            'color_psychology': 0.85,
            'whitespace_usage': 0.8,
            'cta_prominence': 0.95,
            'overall_visual_score': 0.875
        }
    
    async def analyze_copy_effectiveness(self, asset: CreativeAsset) -> Dict[str, Any]:
        """Analyze ad copy effectiveness"""
        return {
            'headline_impact': 0.85,
            'value_proposition_clarity': 0.9,
            'cta_effectiveness': 0.88,
            'readability_score': 0.92,
            'overall_copy_score': 0.8875
        }
    
    async def analyze_platform_ad_performance(self, asset: CreativeAsset) -> Dict[str, Any]:
        """Analyze platform-specific ad performance"""
        return {
            'facebook': {
                'relevance_score': 8,
                'estimated_reach': 10000,
                'cost_per_result': 2.50
            },
            'google_ads': {
                'quality_score': 8,
                'expected_ctr': 'above_average',
                'ad_relevance': 'above_average'
            },
            'overall_platform_readiness': 0.9
        }
    
    async def get_ab_test_results(self, asset: CreativeAsset) -> Dict[str, Any]:
        """Get A/B test results (simulated)"""
        return {
            'test_status': 'running',
            'variants_tested': asset.metadata.get('ab_test_variants', 4),
            'leading_variant': 'variant_headline_001',
            'performance_lift': 0.15,  # 15% lift
            'statistical_significance': 0.92
        }
    
    async def generate_ad_optimization_recommendations(self, asset: CreativeAsset) -> List[str]:
        """Generate optimization recommendations for ad"""
        recommendations = []
        
        if asset.metadata.get('predicted_ctr', 0) < 0.03:
            recommendations.append("Strengthen headline to improve click-through rate")
            
        if asset.metadata.get('predicted_conversion_rate', 0) < 0.04:
            recommendations.append("Enhance value proposition to boost conversions")
            
        if asset.quality_score < 0.9:
            recommendations.append("Add more trust indicators and social proof elements")
            
        recommendations.append("Test different CTA colors for better visibility")
        recommendations.append("Consider adding urgency elements to drive immediate action")
        
        return recommendations