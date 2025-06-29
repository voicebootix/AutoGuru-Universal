"""
Quality Assessor Service

Assesses quality of generated content across different dimensions.
"""

import logging
from typing import Dict, List, Any, Optional
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Quality assessment dimensions"""
    TECHNICAL = "technical"
    AESTHETIC = "aesthetic"
    ENGAGEMENT = "engagement"
    CLARITY = "clarity"
    RELEVANCE = "relevance"
    UNIQUENESS = "uniqueness"


class QualityAssessor:
    """Assesses content quality across multiple dimensions"""
    
    def __init__(self):
        self.quality_weights = {
            QualityDimension.TECHNICAL: 0.25,
            QualityDimension.AESTHETIC: 0.20,
            QualityDimension.ENGAGEMENT: 0.20,
            QualityDimension.CLARITY: 0.15,
            QualityDimension.RELEVANCE: 0.15,
            QualityDimension.UNIQUENESS: 0.05
        }
        
    async def assess_quality(self, content: Any, content_type: str, business_niche: str) -> Dict[str, Any]:
        """Comprehensive quality assessment of content"""
        dimension_scores = {}
        
        # Assess each quality dimension
        dimension_scores[QualityDimension.TECHNICAL] = await self._assess_technical_quality(content, content_type)
        dimension_scores[QualityDimension.AESTHETIC] = await self._assess_aesthetic_quality(content, business_niche)
        dimension_scores[QualityDimension.ENGAGEMENT] = await self._assess_engagement_potential(content, business_niche)
        dimension_scores[QualityDimension.CLARITY] = await self._assess_clarity(content, content_type)
        dimension_scores[QualityDimension.RELEVANCE] = await self._assess_relevance(content, business_niche)
        dimension_scores[QualityDimension.UNIQUENESS] = await self._assess_uniqueness(content)
        
        # Calculate weighted overall score
        overall_score = sum(
            score * self.quality_weights[dimension]
            for dimension, score in dimension_scores.items()
        )
        
        # Identify areas for improvement
        improvement_areas = []
        for dimension, score in dimension_scores.items():
            if score < 0.7:
                improvement_areas.append({
                    'dimension': dimension.value,
                    'current_score': score,
                    'improvement_priority': 'high' if self.quality_weights[dimension] > 0.2 else 'medium'
                })
        
        return {
            'overall_quality_score': overall_score,
            'dimension_scores': {dim.value: score for dim, score in dimension_scores.items()},
            'quality_grade': self._calculate_quality_grade(overall_score),
            'improvement_areas': improvement_areas,
            'recommendations': await self._generate_quality_recommendations(dimension_scores, business_niche)
        }
        
    async def _assess_technical_quality(self, content: Any, content_type: str) -> float:
        """Assess technical quality based on content type"""
        score = 0.8  # Base score
        
        if content_type == 'image':
            # Check resolution, format, compression
            if hasattr(content, 'size'):
                width, height = content.size
                if width >= 1920 and height >= 1080:
                    score += 0.1
                elif width >= 1280 and height >= 720:
                    score += 0.05
                    
        elif content_type == 'video':
            # Check frame rate, resolution, codec
            score = 0.75  # Placeholder
            
        elif content_type == 'copy':
            # Check grammar, spelling, structure
            score = 0.85  # Placeholder
            
        return min(score, 1.0)
        
    async def _assess_aesthetic_quality(self, content: Any, business_niche: str) -> float:
        """Assess aesthetic appeal based on business niche"""
        # Niche-specific aesthetic standards
        niche_aesthetics = {
            'creative_arts': 0.9,  # High aesthetic standards
            'fitness': 0.8,        # Dynamic, energetic aesthetics
            'business_consulting': 0.75,  # Professional, clean aesthetics
            'education': 0.7,      # Clear, educational aesthetics
            'finance': 0.75,       # Trust-inspiring aesthetics
            'health_wellness': 0.85  # Calming, natural aesthetics
        }
        
        base_score = niche_aesthetics.get(business_niche, 0.75)
        
        # Additional aesthetic checks would go here
        # For now, return base score with slight variation
        return base_score
        
    async def _assess_engagement_potential(self, content: Any, business_niche: str) -> float:
        """Assess potential for user engagement"""
        # Simplified engagement scoring
        engagement_factors = {
            'has_hook': 0.2,
            'emotional_appeal': 0.3,
            'call_to_action': 0.2,
            'shareability': 0.3
        }
        
        # Calculate engagement score
        score = 0.0
        
        # Check for engagement factors (simplified)
        score += engagement_factors['has_hook'] * 0.8  # Assume most content has hooks
        score += engagement_factors['emotional_appeal'] * 0.75
        score += engagement_factors['call_to_action'] * 0.9
        score += engagement_factors['shareability'] * 0.7
        
        return score
        
    async def _assess_clarity(self, content: Any, content_type: str) -> float:
        """Assess content clarity and comprehension"""
        # Base clarity score
        clarity_score = 0.8
        
        if content_type == 'copy':
            # Would analyze readability, sentence structure, etc.
            clarity_score = 0.85
        elif content_type == 'image':
            # Would analyze visual hierarchy, contrast, etc.
            clarity_score = 0.8
        elif content_type == 'video':
            # Would analyze pacing, transitions, etc.
            clarity_score = 0.75
            
        return clarity_score
        
    async def _assess_relevance(self, content: Any, business_niche: str) -> float:
        """Assess relevance to business niche and target audience"""
        # Simplified relevance scoring
        return 0.85  # Placeholder - would analyze content against niche keywords
        
    async def _assess_uniqueness(self, content: Any) -> float:
        """Assess content uniqueness and originality"""
        # Simplified uniqueness scoring
        return 0.8  # Placeholder - would check against existing content
        
    def _calculate_quality_grade(self, score: float) -> str:
        """Calculate quality grade from score"""
        if score >= 0.9:
            return 'A+'
        elif score >= 0.85:
            return 'A'
        elif score >= 0.8:
            return 'B+'
        elif score >= 0.75:
            return 'B'
        elif score >= 0.7:
            return 'C+'
        elif score >= 0.65:
            return 'C'
        else:
            return 'D'
            
    async def _generate_quality_recommendations(self, dimension_scores: Dict[QualityDimension, float], business_niche: str) -> List[str]:
        """Generate specific recommendations for quality improvement"""
        recommendations = []
        
        # Technical quality recommendations
        if dimension_scores[QualityDimension.TECHNICAL] < 0.8:
            recommendations.append("Improve technical specifications (resolution, format, compression)")
            
        # Aesthetic recommendations
        if dimension_scores[QualityDimension.AESTHETIC] < 0.8:
            recommendations.append(f"Enhance visual appeal for {business_niche} audience expectations")
            
        # Engagement recommendations
        if dimension_scores[QualityDimension.ENGAGEMENT] < 0.7:
            recommendations.append("Add stronger hooks and calls-to-action to boost engagement")
            
        # Clarity recommendations
        if dimension_scores[QualityDimension.CLARITY] < 0.75:
            recommendations.append("Simplify message and improve visual/textual hierarchy")
            
        # Relevance recommendations
        if dimension_scores[QualityDimension.RELEVANCE] < 0.8:
            recommendations.append(f"Better align content with {business_niche} industry trends and audience needs")
            
        return recommendations
        
    async def compare_quality(self, content_a: Any, content_b: Any, content_type: str, business_niche: str) -> Dict[str, Any]:
        """Compare quality between two pieces of content"""
        # Assess both pieces
        quality_a = await self.assess_quality(content_a, content_type, business_niche)
        quality_b = await self.assess_quality(content_b, content_type, business_niche)
        
        # Compare scores
        comparison = {
            'content_a_score': quality_a['overall_quality_score'],
            'content_b_score': quality_b['overall_quality_score'],
            'winner': 'A' if quality_a['overall_quality_score'] > quality_b['overall_quality_score'] else 'B',
            'score_difference': abs(quality_a['overall_quality_score'] - quality_b['overall_quality_score']),
            'dimension_comparison': {}
        }
        
        # Compare each dimension
        for dimension in QualityDimension:
            dim_value = dimension.value
            comparison['dimension_comparison'][dim_value] = {
                'content_a': quality_a['dimension_scores'][dim_value],
                'content_b': quality_b['dimension_scores'][dim_value],
                'winner': 'A' if quality_a['dimension_scores'][dim_value] > quality_b['dimension_scores'][dim_value] else 'B'
            }
            
        return comparison
        
    async def suggest_quality_improvements(self, content: Any, target_score: float, content_type: str, business_niche: str) -> Dict[str, Any]:
        """Suggest specific improvements to reach target quality score"""
        current_assessment = await self.assess_quality(content, content_type, business_niche)
        current_score = current_assessment['overall_quality_score']
        
        if current_score >= target_score:
            return {
                'improvements_needed': False,
                'current_score': current_score,
                'message': 'Content already meets target quality score'
            }
            
        # Calculate improvements needed
        score_gap = target_score - current_score
        improvements = []
        
        # Prioritize improvements by weight and current score
        for dimension, score in current_assessment['dimension_scores'].items():
            dimension_enum = QualityDimension(dimension)
            weight = self.quality_weights[dimension_enum]
            potential_impact = weight * (1.0 - score)
            
            if score < 0.9:  # Room for improvement
                improvements.append({
                    'dimension': dimension,
                    'current_score': score,
                    'potential_impact': potential_impact,
                    'action': f"Improve {dimension} quality",
                    'priority': 'high' if potential_impact > 0.05 else 'medium'
                })
        
        # Sort by potential impact
        improvements.sort(key=lambda x: x['potential_impact'], reverse=True)
        
        return {
            'improvements_needed': True,
            'current_score': current_score,
            'target_score': target_score,
            'score_gap': score_gap,
            'improvements': improvements[:5],  # Top 5 improvements
            'estimated_effort': self._estimate_improvement_effort(score_gap, improvements)
        }
        
    def _estimate_improvement_effort(self, score_gap: float, improvements: List[Dict[str, Any]]) -> str:
        """Estimate effort required for improvements"""
        if score_gap < 0.05:
            return 'minimal'
        elif score_gap < 0.1:
            return 'low'
        elif score_gap < 0.2:
            return 'medium'
        else:
            return 'high'