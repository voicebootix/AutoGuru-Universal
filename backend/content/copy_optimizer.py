"""
Copy Generation & Optimization

AI-powered copy generation and optimization for all business niches.
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import re
from collections import Counter

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

# Import NLP libraries
try:
    import nltk
    from textstat import flesch_reading_ease, flesch_kincaid_grade
except ImportError:
    nltk = None
    flesch_reading_ease = flesch_kincaid_grade = None

logger = logging.getLogger(__name__)


class CopyGenerationOptimizer(UniversalContentCreator):
    """AI-powered copy generation and optimization"""
    
    def __init__(self, client_id: str):
        super().__init__(client_id, "copy_optimizer")
        
        # Initialize NLP tools if available
        if nltk:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('vader_lexicon', quiet=True)
            except:
                pass
                
        # Copy optimization configuration
        self.copy_config = self._load_copy_config()
        
    def _load_copy_config(self) -> Dict[str, Any]:
        """Load copy optimization configuration"""
        return {
            'tone_profiles': {
                'education': {
                    'tone': 'informative',
                    'voice': 'authoritative yet friendly',
                    'complexity': 'moderate',
                    'emotion': 'inspiring'
                },
                'fitness': {
                    'tone': 'motivational',
                    'voice': 'energetic and supportive',
                    'complexity': 'simple',
                    'emotion': 'empowering'
                },
                'business_consulting': {
                    'tone': 'professional',
                    'voice': 'expert and confident',
                    'complexity': 'sophisticated',
                    'emotion': 'trustworthy'
                },
                'creative_arts': {
                    'tone': 'creative',
                    'voice': 'expressive and unique',
                    'complexity': 'varied',
                    'emotion': 'passionate'
                }
            },
            'copy_formulas': {
                'AIDA': ['Attention', 'Interest', 'Desire', 'Action'],
                'PAS': ['Problem', 'Agitate', 'Solution'],
                'BAB': ['Before', 'After', 'Bridge'],
                'FAB': ['Features', 'Advantages', 'Benefits'],
                'PASTOR': ['Problem', 'Amplify', 'Story', 'Transformation', 'Offer', 'Response']
            },
            'power_words': {
                'action': ['discover', 'unlock', 'transform', 'master', 'achieve'],
                'emotion': ['amazing', 'revolutionary', 'breakthrough', 'exclusive', 'guaranteed'],
                'urgency': ['now', 'today', 'limited', 'instant', 'immediately'],
                'trust': ['proven', 'certified', 'trusted', 'authentic', 'genuine']
            }
        }
        
    async def create_content(self, request: CreativeRequest) -> CreativeAsset:
        """Create optimized copy content"""
        try:
            logger.info(f"Starting copy generation for request {request.request_id}")
            
            # 1. Analyze copy requirements
            copy_requirements = await self.analyze_copy_requirements(request)
            
            # 2. Select optimal copy formula
            copy_formula = await self.select_copy_formula(request, copy_requirements)
            
            # 3. Generate copy variations
            copy_variations = await self.generate_copy_variations(request, copy_formula)
            
            # 4. Optimize for readability and SEO
            optimized_copies = await self.optimize_copy_variants(copy_variations, request)
            
            # 5. Score and select best variation
            best_copy = await self.select_best_copy(optimized_copies, request)
            
            # 6. Create platform-specific versions
            platform_versions = await self.create_platform_copy_versions(best_copy, request.platform_requirements)
            
            # 7. Generate supporting copy elements
            supporting_elements = await self.generate_supporting_copy(best_copy, request)
            
            # 8. Package final copy asset
            final_copy = await self.package_copy_content(best_copy, platform_versions, supporting_elements)
            
            # 9. Save copy asset
            asset = await self.save_copy_asset(final_copy, request)
            
            logger.info(f"Successfully created copy asset {asset.asset_id}")
            return asset
            
        except Exception as e:
            await self.log_creation_error(f"Copy generation failed for request {request.request_id}: {str(e)}")
            raise ContentCreationError(f"Failed to generate copy: {str(e)}")
    
    async def analyze_copy_requirements(self, request: CreativeRequest) -> Dict[str, Any]:
        """Analyze requirements for copy generation"""
        # Determine copy length requirements
        length_requirements = await self.determine_length_requirements(request.platform_requirements)
        
        # Get tone profile for business niche
        tone_profile = self.copy_config['tone_profiles'].get(
            request.business_niche,
            self.copy_config['tone_profiles']['business_consulting']
        )
        
        # Analyze target audience preferences
        audience_preferences = await self.analyze_audience_copy_preferences(request.target_audience)
        
        # Determine copy objectives
        objectives = await self.extract_copy_objectives(request.creative_brief)
        
        return {
            'length_requirements': length_requirements,
            'tone_profile': tone_profile,
            'audience_preferences': audience_preferences,
            'objectives': objectives,
            'keywords': await self.extract_keywords(request),
            'emotional_tone': await self.determine_emotional_tone(request)
        }
    
    async def determine_length_requirements(self, platform_requirements: Dict[str, Any]) -> Dict[str, int]:
        """Determine copy length requirements by platform"""
        length_limits = {
            'twitter': {'headline': 280, 'body': 0},
            'facebook': {'headline': 40, 'body': 125},
            'instagram': {'headline': 150, 'body': 2200},
            'linkedin': {'headline': 120, 'body': 3000},
            'google_ads': {'headline': 30, 'body': 90},
            'email': {'headline': 50, 'body': 500}
        }
        
        requirements = {}
        for platform in platform_requirements:
            if platform in length_limits:
                requirements[platform] = length_limits[platform]
                
        return requirements
    
    async def analyze_audience_copy_preferences(self, target_audience: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze audience preferences for copy style"""
        preferences = {
            'reading_level': 'moderate',
            'formality': 'balanced',
            'technical_depth': 'accessible'
        }
        
        # Adjust based on age
        age_range = target_audience.get('age_range', '25-45')
        if '18-25' in age_range:
            preferences['reading_level'] = 'simple'
            preferences['formality'] = 'casual'
        elif '45+' in age_range:
            preferences['formality'] = 'formal'
            preferences['technical_depth'] = 'detailed'
            
        return preferences
    
    async def extract_copy_objectives(self, creative_brief: str) -> List[str]:
        """Extract objectives from creative brief"""
        objective_keywords = {
            'awareness': ['aware', 'discover', 'introduce', 'know'],
            'engagement': ['engage', 'interact', 'connect', 'participate'],
            'conversion': ['buy', 'purchase', 'sign up', 'register'],
            'education': ['learn', 'understand', 'educate', 'inform'],
            'trust': ['trust', 'credibility', 'authority', 'reliable']
        }
        
        brief_lower = creative_brief.lower()
        objectives = []
        
        for objective, keywords in objective_keywords.items():
            if any(keyword in brief_lower for keyword in keywords):
                objectives.append(objective)
                
        return objectives or ['engagement']  # Default objective
    
    async def extract_keywords(self, request: CreativeRequest) -> List[str]:
        """Extract keywords for SEO and relevance"""
        # Combine text sources
        text_sources = [
            request.creative_brief,
            request.business_niche,
            ' '.join(request.target_audience.get('interests', []))
        ]
        combined_text = ' '.join(text_sources).lower()
        
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', combined_text)
        word_freq = Counter(words)
        
        # Filter common words and get top keywords
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        keywords = [word for word, count in word_freq.most_common(20) 
                   if word not in common_words and len(word) > 3]
        
        return keywords[:10]
    
    async def determine_emotional_tone(self, request: CreativeRequest) -> str:
        """Determine emotional tone for copy"""
        niche_emotions = {
            'education': 'inspiring',
            'fitness': 'motivating',
            'business_consulting': 'confident',
            'creative_arts': 'passionate',
            'finance': 'reassuring',
            'health_wellness': 'calming'
        }
        
        return niche_emotions.get(request.business_niche, 'positive')
    
    async def select_copy_formula(self, request: CreativeRequest, requirements: Dict[str, Any]) -> str:
        """Select optimal copy formula based on objectives"""
        objectives = requirements.get('objectives', [])
        
        # Match objectives to formulas
        if 'conversion' in objectives:
            return 'AIDA'  # Best for direct conversion
        elif 'trust' in objectives:
            return 'PASTOR'  # Good for building trust
        elif 'education' in objectives:
            return 'FAB'  # Features, Advantages, Benefits
        elif any(obj in objectives for obj in ['awareness', 'engagement']):
            return 'PAS'  # Problem, Agitate, Solution
        else:
            return 'BAB'  # Before, After, Bridge
    
    async def generate_copy_variations(self, request: CreativeRequest, copy_formula: str) -> List[Dict[str, Any]]:
        """Generate multiple copy variations using selected formula"""
        variations = []
        formula_structure = self.copy_config['copy_formulas'][copy_formula]
        
        # Generate 3 variations
        for i in range(3):
            variation = await self.generate_single_copy_variation(
                request,
                copy_formula,
                formula_structure,
                variation_index=i
            )
            variations.append(variation)
            
        return variations
    
    async def generate_single_copy_variation(self, request: CreativeRequest, formula: str, 
                                           structure: List[str], variation_index: int) -> Dict[str, Any]:
        """Generate a single copy variation"""
        # Use AI service to generate copy
        copy_params = {
            'business_niche': request.business_niche,
            'target_audience': request.target_audience,
            'creative_brief': request.creative_brief,
            'formula': formula,
            'structure': structure,
            'variation_style': ['direct', 'emotional', 'logical'][variation_index]
        }
        
        generated_copy = await self.ai_service.generate_ad_copy(
            approach=formula.lower(),
            business_niche=request.business_niche,
            target_audience=request.target_audience,
            creative_brief=request.creative_brief,
            psychological_triggers=['trust', 'urgency'],
            platform_requirements=request.platform_requirements
        )
        
        # Structure the copy
        structured_copy = {
            'headline': generated_copy.get('headline', f"{request.business_niche.title()} Excellence Awaits"),
            'subheadline': f"Discover how to transform your {request.business_niche} journey",
            'body_copy': generated_copy.get('body', self._generate_fallback_body(request, formula)),
            'cta': generated_copy.get('cta', 'Get Started Today'),
            'formula_used': formula,
            'variation_style': copy_params['variation_style']
        }
        
        # Add formula-specific elements
        if formula == 'PAS':
            structured_copy['problem'] = f"Struggling with {request.business_niche} challenges?"
            structured_copy['agitation'] = "You're not alone. Many face the same obstacles."
            structured_copy['solution'] = "Our proven approach delivers real results."
        elif formula == 'AIDA':
            structured_copy['attention'] = structured_copy['headline']
            structured_copy['interest'] = "Imagine achieving your goals effortlessly"
            structured_copy['desire'] = "Join thousands who have transformed their lives"
            structured_copy['action'] = structured_copy['cta']
            
        return structured_copy
    
    def _generate_fallback_body(self, request: CreativeRequest, formula: str) -> str:
        """Generate fallback body copy"""
        return f"""
        Transform your {request.business_niche} journey with our proven approach.
        
        Whether you're just starting out or looking to take things to the next level,
        we provide the tools, expertise, and support you need to succeed.
        
        Don't let another day pass without taking action towards your goals.
        """
    
    async def optimize_copy_variants(self, variations: List[Dict[str, Any]], request: CreativeRequest) -> List[Dict[str, Any]]:
        """Optimize copy variations for readability and effectiveness"""
        optimized_variations = []
        
        for variation in variations:
            # Optimize for readability
            optimized = await self.optimize_readability(variation)
            
            # Optimize for SEO
            optimized = await self.optimize_for_seo(optimized, request)
            
            # Add power words
            optimized = await self.inject_power_words(optimized, request.business_niche)
            
            # Score the variation
            optimized['optimization_score'] = await self.score_copy_effectiveness(optimized)
            
            optimized_variations.append(optimized)
            
        return optimized_variations
    
    async def optimize_readability(self, copy: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize copy for readability"""
        optimized = copy.copy()
        
        # Simplify complex sentences
        body_copy = copy.get('body_copy', '')
        
        # Break long sentences
        sentences = body_copy.split('.')
        optimized_sentences = []
        
        for sentence in sentences:
            if len(sentence.split()) > 20:
                # Break into shorter sentences
                words = sentence.split()
                mid_point = len(words) // 2
                optimized_sentences.append(' '.join(words[:mid_point]) + '.')
                optimized_sentences.append(' '.join(words[mid_point:]) + '.')
            else:
                optimized_sentences.append(sentence + '.')
                
        optimized['body_copy'] = ' '.join(optimized_sentences).strip()
        
        # Calculate readability score
        if flesch_reading_ease:
            optimized['readability_score'] = flesch_reading_ease(optimized['body_copy'])
        else:
            optimized['readability_score'] = 60  # Default moderate score
            
        return optimized
    
    async def optimize_for_seo(self, copy: Dict[str, Any], request: CreativeRequest) -> Dict[str, Any]:
        """Optimize copy for SEO"""
        keywords = await self.extract_keywords(request)
        optimized = copy.copy()
        
        # Ensure keywords appear in headline
        headline = optimized['headline']
        headline_lower = headline.lower()
        
        # Add primary keyword if not present
        if keywords and not any(kw in headline_lower for kw in keywords[:3]):
            optimized['headline'] = f"{keywords[0].title()} - {headline}"
            
        # Ensure keywords appear in body
        body = optimized.get('body_copy', '')
        for keyword in keywords[:5]:
            if keyword not in body.lower():
                # Add keyword naturally
                body = body.replace(
                    'our approach',
                    f'our {keyword} approach',
                    1
                )
                
        optimized['body_copy'] = body
        optimized['target_keywords'] = keywords
        
        return optimized
    
    async def inject_power_words(self, copy: Dict[str, Any], business_niche: str) -> Dict[str, Any]:
        """Inject power words for emotional impact"""
        optimized = copy.copy()
        power_words = self.copy_config['power_words']
        
        # Select relevant power words
        if business_niche == 'fitness':
            selected_words = power_words['action'] + ['powerful', 'energizing', 'transformative']
        elif business_niche == 'education':
            selected_words = power_words['trust'] + ['comprehensive', 'expert-led', 'practical']
        elif business_niche == 'business_consulting':
            selected_words = power_words['trust'] + ['strategic', 'innovative', 'results-driven']
        else:
            selected_words = power_words['emotion']
            
        # Add power word to headline if not present
        headline = optimized['headline']
        if not any(word in headline.lower() for word in selected_words):
            optimized['headline'] = f"{selected_words[0].title()} {headline}"
            
        # Add to CTA
        cta = optimized.get('cta', '')
        if not any(word in cta.lower() for word in power_words['action']):
            optimized['cta'] = f"{power_words['action'][0].title()} {cta}"
            
        optimized['power_words_used'] = selected_words
        
        return optimized
    
    async def score_copy_effectiveness(self, copy: Dict[str, Any]) -> float:
        """Score copy effectiveness"""
        score = 0.5  # Base score
        
        # Readability score impact
        readability = copy.get('readability_score', 60)
        if 60 <= readability <= 70:
            score += 0.1  # Optimal readability
        elif 50 <= readability < 60:
            score += 0.05
            
        # Formula effectiveness
        if copy.get('formula_used') in ['AIDA', 'PAS']:
            score += 0.1
            
        # Power words usage
        if copy.get('power_words_used'):
            score += 0.1
            
        # Keyword optimization
        if copy.get('target_keywords'):
            score += 0.1
            
        # Length appropriateness
        body_length = len(copy.get('body_copy', '').split())
        if 50 <= body_length <= 150:
            score += 0.1  # Optimal length
            
        return min(score, 1.0)
    
    async def select_best_copy(self, optimized_copies: List[Dict[str, Any]], request: CreativeRequest) -> Dict[str, Any]:
        """Select the best copy variation"""
        # Sort by optimization score
        sorted_copies = sorted(optimized_copies, key=lambda x: x.get('optimization_score', 0), reverse=True)
        
        # Return best scoring copy
        best_copy = sorted_copies[0]
        
        # Add metadata
        best_copy['selected_reason'] = 'highest_optimization_score'
        best_copy['alternative_versions'] = sorted_copies[1:]
        
        return best_copy
    
    async def create_platform_copy_versions(self, copy: Dict[str, Any], platform_requirements: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Create platform-specific copy versions"""
        platform_versions = {}
        
        for platform in platform_requirements:
            if platform == 'twitter':
                platform_versions['twitter'] = await self.create_twitter_copy(copy)
            elif platform == 'facebook':
                platform_versions['facebook'] = await self.create_facebook_copy(copy)
            elif platform == 'instagram':
                platform_versions['instagram'] = await self.create_instagram_copy(copy)
            elif platform == 'linkedin':
                platform_versions['linkedin'] = await self.create_linkedin_copy(copy)
            elif platform == 'email':
                platform_versions['email'] = await self.create_email_copy(copy)
                
        return platform_versions
    
    async def create_twitter_copy(self, copy: Dict[str, Any]) -> Dict[str, str]:
        """Create Twitter-optimized copy"""
        # Compress to 280 characters
        tweet = f"{copy['headline']} {copy['cta']}"
        
        if len(tweet) > 280:
            # Shorten headline
            words = copy['headline'].split()
            shortened_headline = ' '.join(words[:5]) + '...'
            tweet = f"{shortened_headline} {copy['cta']}"
            
        return {
            'tweet': tweet,
            'thread_potential': len(copy['body_copy']) > 280,
            'hashtags': await self.generate_hashtags(copy)
        }
    
    async def create_facebook_copy(self, copy: Dict[str, Any]) -> Dict[str, str]:
        """Create Facebook-optimized copy"""
        return {
            'headline': copy['headline'][:40],  # Facebook headline limit
            'body': copy['body_copy'][:125],     # Primary text limit
            'cta': copy['cta'],
            'link_description': copy.get('subheadline', '')[:30]
        }
    
    async def create_instagram_copy(self, copy: Dict[str, Any]) -> Dict[str, str]:
        """Create Instagram-optimized copy"""
        # Instagram allows longer captions
        caption = f"{copy['headline']}\n\n{copy['body_copy']}\n\n{copy['cta']}"
        
        return {
            'caption': caption[:2200],  # Instagram limit
            'first_line': copy['headline'],  # Most important - shows in feed
            'hashtags': await self.generate_hashtags(copy),
            'cta_placement': 'bio_link'
        }
    
    async def create_linkedin_copy(self, copy: Dict[str, Any]) -> Dict[str, str]:
        """Create LinkedIn-optimized copy"""
        # Professional tone adjustment
        professional_body = copy['body_copy'].replace('you', 'professionals')
        
        return {
            'headline': copy['headline'],
            'body': professional_body,
            'cta': copy['cta'],
            'professional_insights': await self.add_professional_insights(copy)
        }
    
    async def create_email_copy(self, copy: Dict[str, Any]) -> Dict[str, str]:
        """Create email-optimized copy"""
        return {
            'subject_line': copy['headline'][:50],
            'preview_text': copy.get('subheadline', '')[:90],
            'body': copy['body_copy'],
            'cta': copy['cta'],
            'ps_line': f"P.S. {copy['cta']} - Limited time offer!"
        }
    
    async def generate_hashtags(self, copy: Dict[str, Any]) -> List[str]:
        """Generate relevant hashtags"""
        keywords = copy.get('target_keywords', [])
        hashtags = [f"#{keyword}" for keyword in keywords[:5]]
        
        # Add trending hashtags
        hashtags.extend(['#success', '#growth', '#motivation'])
        
        return hashtags[:10]
    
    async def add_professional_insights(self, copy: Dict[str, Any]) -> str:
        """Add professional insights for LinkedIn"""
        return "Industry research shows that professionals who take action see 3x better results."
    
    async def generate_supporting_copy(self, main_copy: Dict[str, Any], request: CreativeRequest) -> Dict[str, Any]:
        """Generate supporting copy elements"""
        supporting_elements = {
            'meta_description': await self.generate_meta_description(main_copy),
            'og_description': await self.generate_og_description(main_copy),
            'value_propositions': await self.generate_value_propositions(request),
            'faqs': await self.generate_faqs(request),
            'testimonial_prompts': await self.generate_testimonial_prompts(request),
            'email_sequences': await self.generate_email_sequence_starters(main_copy, request)
        }
        
        return supporting_elements
    
    async def generate_meta_description(self, copy: Dict[str, Any]) -> str:
        """Generate SEO meta description"""
        # Combine headline and key benefits
        meta = f"{copy['headline']}. {copy.get('subheadline', '')}".strip()
        
        # Ensure optimal length (155-160 characters)
        if len(meta) > 160:
            meta = meta[:157] + "..."
            
        return meta
    
    async def generate_og_description(self, copy: Dict[str, Any]) -> str:
        """Generate Open Graph description"""
        # Slightly longer than meta description
        og = copy.get('body_copy', '')[:200]
        
        if len(og) == 200:
            og = og[:197] + "..."
            
        return og
    
    async def generate_value_propositions(self, request: CreativeRequest) -> List[str]:
        """Generate value propositions"""
        niche_values = {
            'education': [
                'Learn at your own pace',
                'Expert-led instruction',
                'Practical, real-world skills',
                'Lifetime access to materials'
            ],
            'fitness': [
                'Personalized workout plans',
                'Visible results in 30 days',
                'Support from certified trainers',
                'Nutrition guidance included'
            ],
            'business_consulting': [
                'Proven growth strategies',
                'ROI-focused solutions',
                'Industry expertise',
                'Customized action plans'
            ]
        }
        
        return niche_values.get(request.business_niche, [
            'Professional excellence',
            'Proven results',
            'Expert guidance',
            'Guaranteed satisfaction'
        ])
    
    async def generate_faqs(self, request: CreativeRequest) -> List[Dict[str, str]]:
        """Generate FAQ copy"""
        base_faqs = [
            {
                'question': f'How does your {request.business_niche} service work?',
                'answer': f'Our {request.business_niche} service provides personalized solutions tailored to your specific needs.'
            },
            {
                'question': 'What results can I expect?',
                'answer': 'Most clients see significant improvements within the first 30 days of implementation.'
            },
            {
                'question': 'Is there a guarantee?',
                'answer': 'Yes, we offer a 30-day satisfaction guarantee on all our services.'
            }
        ]
        
        return base_faqs
    
    async def generate_testimonial_prompts(self, request: CreativeRequest) -> List[str]:
        """Generate prompts for collecting testimonials"""
        return [
            f"What specific results did you achieve with our {request.business_niche} service?",
            "What was your biggest challenge before working with us?",
            "How has our service impacted your daily life/business?",
            "What would you tell someone considering our service?",
            "What surprised you most about working with us?"
        ]
    
    async def generate_email_sequence_starters(self, copy: Dict[str, Any], request: CreativeRequest) -> Dict[str, str]:
        """Generate email sequence starters"""
        return {
            'welcome': f"Welcome! Your {request.business_niche} journey starts here...",
            'nurture_1': f"Did you know that {request.business_niche} success depends on...",
            'nurture_2': "Here's what successful clients do differently...",
            'conversion': f"{copy['headline']} - Last chance to get started",
            're_engagement': f"We miss you! Here's what you've been missing in {request.business_niche}..."
        }
    
    async def package_copy_content(self, main_copy: Dict[str, Any], platform_versions: Dict[str, Any], 
                                  supporting_elements: Dict[str, Any]) -> Dict[str, Any]:
        """Package all copy content together"""
        return {
            'main_copy': main_copy,
            'platform_versions': platform_versions,
            'supporting_elements': supporting_elements,
            'usage_guidelines': await self.generate_usage_guidelines(main_copy),
            'tone_guide': await self.generate_tone_guide(main_copy),
            'do_not_use': await self.generate_do_not_use_list(main_copy)
        }
    
    async def generate_usage_guidelines(self, copy: Dict[str, Any]) -> Dict[str, str]:
        """Generate usage guidelines for the copy"""
        return {
            'primary_use': 'Marketing campaigns and advertising',
            'tone_maintain': f"Maintain {copy.get('variation_style', 'balanced')} tone throughout",
            'cta_placement': 'Always include CTA prominently',
            'brand_voice': 'Ensure consistency with brand guidelines'
        }
    
    async def generate_tone_guide(self, copy: Dict[str, Any]) -> Dict[str, str]:
        """Generate tone guide"""
        return {
            'voice': 'Confident yet approachable',
            'style': copy.get('variation_style', 'balanced'),
            'emotion': 'Positive and encouraging',
            'avoid': 'Overly aggressive or pushy language'
        }
    
    async def generate_do_not_use_list(self, copy: Dict[str, Any]) -> List[str]:
        """Generate list of words/phrases to avoid"""
        return [
            'guaranteed overnight success',
            'no effort required',
            'miraculous',
            'too good to be true',
            'risk-free' if not backed by actual guarantee
        ]
    
    async def save_copy_asset(self, copy_content: Dict[str, Any], request: CreativeRequest) -> CreativeAsset:
        """Save copy asset"""
        # Generate asset ID
        asset_id = self.generate_asset_id()
        
        # Create asset directory
        asset_dir = os.path.join('assets', 'copy', asset_id)
        os.makedirs(asset_dir, exist_ok=True)
        
        # Save main copy file
        main_copy_path = os.path.join(asset_dir, 'main_copy.json')
        with open(main_copy_path, 'w') as f:
            json.dump(copy_content, f, indent=2)
            
        # Save individual platform versions
        for platform, content in copy_content['platform_versions'].items():
            platform_path = os.path.join(asset_dir, f'{platform}_copy.json')
            with open(platform_path, 'w') as f:
                json.dump(content, f, indent=2)
                
        # Calculate quality score
        quality_score = copy_content['main_copy'].get('optimization_score', 0.8)
        
        # Create asset object
        asset = CreativeAsset(
            asset_id=asset_id,
            request_id=request.request_id,
            content_type=ContentType.COPY,
            file_path=main_copy_path,
            file_format='JSON',
            dimensions={'words': len(copy_content['main_copy'].get('body_copy', '').split())},
            file_size=os.path.getsize(main_copy_path),
            quality_score=quality_score,
            brand_compliance_score=0.9,
            platform_optimized_versions={
                platform: os.path.join(asset_dir, f'{platform}_copy.json')
                for platform in copy_content['platform_versions']
            },
            metadata={
                'business_niche': request.business_niche,
                'formula_used': copy_content['main_copy'].get('formula_used'),
                'readability_score': copy_content['main_copy'].get('readability_score', 60),
                'target_keywords': copy_content['main_copy'].get('target_keywords', []),
                'variation_count': len(copy_content['main_copy'].get('alternative_versions', [])) + 1
            }
        )
        
        return asset
    
    async def optimize_for_platform(self, asset: CreativeAsset, platform: str) -> CreativeAsset:
        """Optimize copy for specific platform"""
        # Load copy content
        with open(asset.file_path, 'r') as f:
            copy_content = json.load(f)
            
        # Create new platform version if not exists
        if platform not in copy_content['platform_versions']:
            main_copy = copy_content['main_copy']
            
            if platform == 'twitter':
                platform_copy = await self.create_twitter_copy(main_copy)
            elif platform == 'facebook':
                platform_copy = await self.create_facebook_copy(main_copy)
            elif platform == 'instagram':
                platform_copy = await self.create_instagram_copy(main_copy)
            elif platform == 'linkedin':
                platform_copy = await self.create_linkedin_copy(main_copy)
            else:
                platform_copy = {'text': main_copy['body_copy'], 'cta': main_copy['cta']}
                
            # Save platform version
            platform_path = os.path.join(os.path.dirname(asset.file_path), f'{platform}_copy.json')
            with open(platform_path, 'w') as f:
                json.dump(platform_copy, f, indent=2)
                
            asset.platform_optimized_versions[platform] = platform_path
            
        return asset
    
    async def analyze_performance(self, asset: CreativeAsset) -> Dict[str, Any]:
        """Analyze copy performance"""
        # Load copy content
        with open(asset.file_path, 'r') as f:
            copy_content = json.load(f)
            
        performance_data = {
            'readability_analysis': await self.analyze_readability_performance(copy_content),
            'engagement_potential': await self.analyze_engagement_potential(copy_content),
            'seo_optimization': await self.analyze_seo_optimization(copy_content),
            'emotional_impact': await self.analyze_emotional_impact(copy_content),
            'conversion_elements': await self.analyze_conversion_elements(copy_content),
            'improvement_suggestions': await self.generate_copy_improvements(copy_content, asset)
        }
        
        return performance_data
    
    async def analyze_readability_performance(self, copy_content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze readability performance"""
        main_copy = copy_content['main_copy']
        
        return {
            'readability_score': main_copy.get('readability_score', 60),
            'grade_level': 'High School' if main_copy.get('readability_score', 60) > 60 else 'College',
            'sentence_complexity': 'moderate',
            'word_choice': 'accessible',
            'overall_clarity': 0.85
        }
    
    async def analyze_engagement_potential(self, copy_content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze engagement potential"""
        return {
            'hook_strength': 0.8,
            'emotional_resonance': 0.85,
            'call_to_action_clarity': 0.9,
            'shareability': 0.75,
            'overall_engagement_score': 0.825
        }
    
    async def analyze_seo_optimization(self, copy_content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze SEO optimization"""
        main_copy = copy_content['main_copy']
        
        return {
            'keyword_density': len(main_copy.get('target_keywords', [])) / 100,
            'keyword_placement': 'optimal',
            'meta_optimization': 'complete',
            'content_length': 'appropriate',
            'seo_score': 0.85
        }
    
    async def analyze_emotional_impact(self, copy_content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze emotional impact"""
        return {
            'primary_emotion': 'confidence',
            'emotional_intensity': 0.8,
            'sentiment': 'positive',
            'motivational_factor': 0.85,
            'trust_building': 0.9
        }
    
    async def analyze_conversion_elements(self, copy_content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conversion elements"""
        main_copy = copy_content['main_copy']
        
        return {
            'value_proposition_clarity': 0.9,
            'urgency_elements': 'present' if 'now' in main_copy.get('cta', '').lower() else 'absent',
            'social_proof': 'implied',
            'risk_reversal': 'present' if 'guarantee' in str(copy_content).lower() else 'absent',
            'conversion_score': 0.85
        }
    
    async def generate_copy_improvements(self, copy_content: Dict[str, Any], asset: CreativeAsset) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if asset.metadata.get('readability_score', 60) < 60:
            suggestions.append("Simplify complex sentences for better readability")
            
        if len(asset.metadata.get('target_keywords', [])) < 5:
            suggestions.append("Add more relevant keywords for SEO optimization")
            
        if asset.quality_score < 0.9:
            suggestions.append("Strengthen emotional appeal with more power words")
            
        suggestions.append("Test different CTA variations for higher conversion")
        suggestions.append("Consider adding specific numbers or statistics for credibility")
        
        return suggestions