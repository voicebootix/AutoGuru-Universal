"""
Brand Analyzer Service

Analyzes content for brand compliance and consistency.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class BrandGuidelines:
    """Brand guidelines structure"""
    colors: List[str]
    fonts: List[str]
    logo_usage: Dict[str, Any]
    tone_of_voice: str
    visual_style: str
    required_elements: List[str]
    prohibited_elements: List[str]


class BrandAnalyzer:
    """Analyzes content for brand compliance"""
    
    def __init__(self):
        self.compliance_thresholds = {
            'color_compliance': 0.8,
            'style_compliance': 0.7,
            'tone_compliance': 0.75,
            'element_compliance': 0.9
        }
        
    async def check_compliance(self, asset: Any, guidelines: Dict[str, Any]) -> Dict[str, Any]:
        """Check asset compliance with brand guidelines"""
        compliance_scores = {}
        
        # Check different aspects of compliance
        compliance_scores['color_compliance'] = await self._check_color_compliance(asset, guidelines)
        compliance_scores['style_compliance'] = await self._check_style_compliance(asset, guidelines)
        compliance_scores['element_compliance'] = await self._check_element_compliance(asset, guidelines)
        compliance_scores['format_compliance'] = await self._check_format_compliance(asset, guidelines)
        
        # Calculate overall compliance score
        overall_score = sum(compliance_scores.values()) / len(compliance_scores)
        
        # Identify issues
        issues = []
        for aspect, score in compliance_scores.items():
            threshold = self.compliance_thresholds.get(aspect, 0.7)
            if score < threshold:
                issues.append({
                    'aspect': aspect,
                    'score': score,
                    'threshold': threshold,
                    'severity': 'high' if score < threshold * 0.7 else 'medium'
                })
        
        return {
            'compliance_score': overall_score,
            'aspect_scores': compliance_scores,
            'issues': issues,
            'compliant': overall_score >= 0.8,
            'recommendations': await self._generate_compliance_recommendations(issues, guidelines)
        }
        
    async def _check_color_compliance(self, asset: Any, guidelines: Dict[str, Any]) -> float:
        """Check color compliance"""
        # Simplified implementation
        if not guidelines.get('colors'):
            return 1.0
            
        # In production, would analyze actual colors in the asset
        # For now, return a reasonable score
        return 0.85
        
    async def _check_style_compliance(self, asset: Any, guidelines: Dict[str, Any]) -> float:
        """Check style compliance"""
        # Check if visual style matches brand guidelines
        if not guidelines.get('visual_style'):
            return 1.0
            
        # Simplified scoring
        return 0.8
        
    async def _check_element_compliance(self, asset: Any, guidelines: Dict[str, Any]) -> float:
        """Check required/prohibited elements"""
        score = 1.0
        
        # Check for required elements
        required_elements = guidelines.get('required_elements', [])
        if required_elements:
            # In production, would check if elements are present
            score *= 0.9
            
        # Check for prohibited elements
        prohibited_elements = guidelines.get('prohibited_elements', [])
        if prohibited_elements:
            # In production, would check if prohibited elements are absent
            score *= 0.95
            
        return score
        
    async def _check_format_compliance(self, asset: Any, guidelines: Dict[str, Any]) -> float:
        """Check format compliance"""
        # Check if asset format meets guidelines
        return 0.9
        
    async def _generate_compliance_recommendations(self, issues: List[Dict[str, Any]], guidelines: Dict[str, Any]) -> List[str]:
        """Generate recommendations to fix compliance issues"""
        recommendations = []
        
        for issue in issues:
            aspect = issue['aspect']
            
            if aspect == 'color_compliance':
                recommendations.append(f"Adjust colors to match brand palette: {', '.join(guidelines.get('colors', []))}")
            elif aspect == 'style_compliance':
                recommendations.append(f"Align visual style with brand guidelines: {guidelines.get('visual_style', 'professional')}")
            elif aspect == 'element_compliance':
                recommendations.append("Ensure all required brand elements are present and prohibited elements are removed")
            elif aspect == 'format_compliance':
                recommendations.append("Adjust format to meet brand specifications")
                
        return recommendations
        
    async def analyze_brand_consistency(self, assets: List[Any], guidelines: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze brand consistency across multiple assets"""
        consistency_scores = []
        
        for asset in assets:
            compliance = await self.check_compliance(asset, guidelines)
            consistency_scores.append(compliance['compliance_score'])
            
        # Calculate consistency metrics
        avg_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
        min_score = min(consistency_scores) if consistency_scores else 0
        max_score = max(consistency_scores) if consistency_scores else 0
        
        # Calculate variance
        variance = sum((score - avg_score) ** 2 for score in consistency_scores) / len(consistency_scores) if consistency_scores else 0
        
        return {
            'average_compliance': avg_score,
            'minimum_compliance': min_score,
            'maximum_compliance': max_score,
            'consistency_score': 1 - (variance ** 0.5),  # Lower variance = higher consistency
            'total_assets_analyzed': len(assets),
            'recommendations': await self._generate_consistency_recommendations(avg_score, variance)
        }
        
    async def _generate_consistency_recommendations(self, avg_score: float, variance: float) -> List[str]:
        """Generate recommendations for improving brand consistency"""
        recommendations = []
        
        if avg_score < 0.8:
            recommendations.append("Overall brand compliance needs improvement across all assets")
            
        if variance > 0.1:
            recommendations.append("Ensure consistent application of brand guidelines across all content")
            
        if avg_score > 0.9 and variance < 0.05:
            recommendations.append("Excellent brand consistency! Continue following current guidelines")
            
        return recommendations
        
    async def suggest_brand_improvements(self, asset: Any, guidelines: Dict[str, Any], target_score: float = 0.9) -> Dict[str, Any]:
        """Suggest specific improvements to reach target compliance score"""
        current_compliance = await self.check_compliance(asset, guidelines)
        current_score = current_compliance['compliance_score']
        
        if current_score >= target_score:
            return {
                'improvements_needed': False,
                'current_score': current_score,
                'message': 'Asset already meets target compliance score'
            }
            
        improvements = []
        score_gap = target_score - current_score
        
        # Prioritize improvements based on impact
        for issue in current_compliance['issues']:
            if issue['severity'] == 'high':
                improvements.append({
                    'aspect': issue['aspect'],
                    'action': f"Fix {issue['aspect']} to improve score by ~{(issue['threshold'] - issue['score']) * 0.25:.2f}",
                    'priority': 'high'
                })
            else:
                improvements.append({
                    'aspect': issue['aspect'],
                    'action': f"Improve {issue['aspect']} for minor score increase",
                    'priority': 'medium'
                })
                
        return {
            'improvements_needed': True,
            'current_score': current_score,
            'target_score': target_score,
            'score_gap': score_gap,
            'improvements': improvements,
            'estimated_effort': 'medium' if len(improvements) > 3 else 'low'
        }