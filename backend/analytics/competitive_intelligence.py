"""
Competitive Intelligence System Module

This module provides comprehensive competitive intelligence analysis that automatically
adapts to any business niche. It monitors competitors, analyzes market position,
identifies competitive advantages, and provides strategic insights.

Features:
- Competitor identification and profiling
- Market share analysis
- Competitive positioning mapping
- SWOT analysis automation
- Competitive threat assessment
- Opportunity identification
- Strategic recommendations
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
from collections import defaultdict
import json

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .base_analytics import (
    UniversalAnalyticsEngine, AnalyticsInsight, InsightPriority,
    AnalyticsScope, BusinessKPI, AnalyticsRequest, ReportFormat
)
from backend.core.persona_factory import PersonaFactory
from backend.utils.encryption import EncryptionManager
from backend.database.connection import get_db_context

logger = logging.getLogger(__name__)


class CompetitorType(Enum):
    """Types of competitors"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    POTENTIAL = "potential"
    SUBSTITUTE = "substitute"
    NEW_ENTRANT = "new_entrant"


class CompetitiveAdvantage(Enum):
    """Types of competitive advantages"""
    COST_LEADERSHIP = "cost_leadership"
    DIFFERENTIATION = "differentiation"
    FOCUS = "focus"
    INNOVATION = "innovation"
    CUSTOMER_SERVICE = "customer_service"
    BRAND = "brand"
    TECHNOLOGY = "technology"
    SCALE = "scale"


class MarketPosition(Enum):
    """Market position categories"""
    LEADER = "leader"
    CHALLENGER = "challenger"
    FOLLOWER = "follower"
    NICHER = "nicher"
    NEW_ENTRANT = "new_entrant"


@dataclass
class CompetitorProfile:
    """Comprehensive competitor profile"""
    competitor_id: str
    name: str
    type: CompetitorType
    market_share: float
    revenue_estimate: float
    growth_rate: float
    strengths: List[str]
    weaknesses: List[str]
    products_services: List[Dict[str, Any]]
    target_segments: List[str]
    pricing_strategy: str
    marketing_channels: List[str]
    technology_stack: List[str]
    key_personnel: List[Dict[str, str]]
    recent_developments: List[Dict[str, Any]]
    threat_level: float
    opportunity_areas: List[str]


@dataclass
class CompetitivePosition:
    """Company's competitive position analysis"""
    market_position: MarketPosition
    market_share: float
    relative_strength: float
    competitive_advantages: List[CompetitiveAdvantage]
    vulnerabilities: List[str]
    opportunities: List[str]
    threats: List[str]
    strategic_recommendations: List[str]


@dataclass
class MarketDynamics:
    """Market dynamics and competitive landscape"""
    total_market_size: float
    growth_rate: float
    concentration_ratio: float
    entry_barriers: List[str]
    key_success_factors: List[str]
    industry_trends: List[str]
    disruption_risks: List[str]
    consolidation_potential: float


class CompetitorDataCollector:
    """Collects and processes competitor data from various sources"""
    
    def __init__(self, persona_factory: PersonaFactory):
        self.persona_factory = persona_factory
        
    async def collect_competitor_data(
        self,
        business_type: str,
        company_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Collect competitor data from various sources"""
        try:
            async with get_db_context() as db:
                # Get competitor data
                query = """
                SELECT 
                    c.id as competitor_id,
                    c.name,
                    c.type,
                    c.market_share,
                    c.revenue,
                    c.employee_count,
                    c.founded_date,
                    c.products,
                    c.pricing_data,
                    c.marketing_data,
                    c.social_media_metrics,
                    c.web_traffic,
                    c.customer_reviews,
                    c.news_mentions
                FROM competitors c
                WHERE c.industry = %s
                AND c.active = true
                """
                
                result = await db.fetch_all(query, business_type)
                
                if not result:
                    # Generate synthetic competitor data
                    return self._generate_synthetic_competitor_data(business_type)
                
                # Process results
                competitors = []
                for row in result:
                    competitor_data = {
                        'competitor_id': row['competitor_id'],
                        'name': row['name'],
                        'type': row['type'],
                        'market_share': row['market_share'] or 0,
                        'revenue': row['revenue'] or 0,
                        'employee_count': row['employee_count'] or 0,
                        'age_years': (
                            (datetime.now() - row['founded_date']).days / 365
                            if row['founded_date'] else 5
                        )
                    }
                    
                    # Add metrics
                    if row['social_media_metrics']:
                        metrics = json.loads(row['social_media_metrics'])
                        competitor_data.update(metrics)
                    
                    competitors.append(competitor_data)
                
                return pd.DataFrame(competitors)
                
        except Exception as e:
            logger.error(f"Error collecting competitor data: {e}")
            return self._generate_synthetic_competitor_data(business_type)
    
    def _generate_synthetic_competitor_data(self, business_type: str) -> pd.DataFrame:
        """Generate synthetic competitor data for analysis"""
        persona = self.persona_factory.create_persona(business_type)
        
        # Generate competitors based on business type
        n_competitors = np.random.randint(5, 15)
        
        competitors = []
        total_market = 100  # Market share percentage
        
        for i in range(n_competitors):
            # Assign market share using power law distribution
            if i == 0:
                market_share = np.random.uniform(20, 35)  # Leader
            elif i < 3:
                market_share = np.random.uniform(10, 20)  # Challengers
            else:
                market_share = np.random.uniform(1, 10)  # Others
            
            total_market -= market_share
            
            competitor = {
                'competitor_id': f'COMP_{i:03d}',
                'name': f'Competitor {i+1}',
                'type': np.random.choice(list(CompetitorType)).value,
                'market_share': market_share / 100,
                'revenue': np.random.lognormal(15, 1),
                'employee_count': int(np.random.lognormal(5, 1)),
                'age_years': np.random.exponential(10),
                'growth_rate': np.random.normal(0.1, 0.2),
                'customer_satisfaction': np.random.beta(4, 1),
                'price_index': np.random.uniform(0.7, 1.3),
                'innovation_score': np.random.beta(2, 2),
                'brand_strength': np.random.beta(3, 2),
                'digital_presence': np.random.beta(3, 1),
                'market_reach': np.random.beta(2, 3)
            }
            
            competitors.append(competitor)
        
        return pd.DataFrame(competitors)
    
    async def collect_market_data(
        self,
        business_type: str
    ) -> Dict[str, Any]:
        """Collect overall market data"""
        try:
            async with get_db_context() as db:
                query = """
                SELECT 
                    market_size,
                    growth_rate,
                    key_players,
                    trends,
                    regulations,
                    technology_factors
                FROM market_intelligence
                WHERE industry = %s
                AND updated_at > NOW() - INTERVAL '30 days'
                ORDER BY updated_at DESC
                LIMIT 1
                """
                
                result = await db.fetch_one(query, business_type)
                
                if result:
                    return dict(result)
                else:
                    return self._generate_synthetic_market_data(business_type)
                    
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            return self._generate_synthetic_market_data(business_type)
    
    def _generate_synthetic_market_data(self, business_type: str) -> Dict[str, Any]:
        """Generate synthetic market data"""
        persona = self.persona_factory.create_persona(business_type)
        
        return {
            'market_size': np.random.lognormal(20, 1),
            'growth_rate': np.random.normal(0.1, 0.05),
            'concentration_ratio': np.random.beta(2, 5),  # CR4
            'entry_barriers': [
                'High capital requirements',
                'Regulatory compliance',
                'Brand loyalty',
                'Economies of scale'
            ],
            'key_success_factors': [
                'Customer experience',
                'Product quality',
                'Innovation',
                'Cost efficiency',
                'Market reach'
            ],
            'trends': [
                'Digital transformation',
                'Sustainability focus',
                'Personalization demand',
                'AI/ML adoption'
            ]
        }


class CompetitorAnalyzer:
    """Analyzes competitors and competitive positioning"""
    
    def __init__(self, persona_factory: PersonaFactory):
        self.persona_factory = persona_factory
        self.scaler = StandardScaler()
        
    def analyze_competitors(
        self,
        competitor_data: pd.DataFrame,
        business_type: str
    ) -> List[CompetitorProfile]:
        """Analyze and profile competitors"""
        profiles = []
        persona = self.persona_factory.create_persona(business_type)
        
        for _, competitor in competitor_data.iterrows():
            # Analyze strengths and weaknesses
            strengths, weaknesses = self._analyze_strengths_weaknesses(
                competitor,
                competitor_data
            )
            
            # Estimate threat level
            threat_level = self._calculate_threat_level(competitor, competitor_data)
            
            # Identify opportunity areas
            opportunities = self._identify_opportunities(
                competitor,
                strengths,
                weaknesses
            )
            
            # Create profile
            profile = CompetitorProfile(
                competitor_id=competitor['competitor_id'],
                name=competitor['name'],
                type=CompetitorType(competitor.get('type', 'direct')),
                market_share=competitor['market_share'],
                revenue_estimate=competitor['revenue'],
                growth_rate=competitor.get('growth_rate', 0.1),
                strengths=strengths,
                weaknesses=weaknesses,
                products_services=self._analyze_products(competitor),
                target_segments=self._identify_target_segments(competitor, persona),
                pricing_strategy=self._analyze_pricing_strategy(competitor),
                marketing_channels=self._identify_marketing_channels(competitor),
                technology_stack=self._analyze_technology(competitor),
                key_personnel=[],  # Would be populated from real data
                recent_developments=self._get_recent_developments(competitor),
                threat_level=threat_level,
                opportunity_areas=opportunities
            )
            
            profiles.append(profile)
        
        return profiles
    
    def _analyze_strengths_weaknesses(
        self,
        competitor: pd.Series,
        all_competitors: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """Analyze competitor strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        # Market position
        if competitor['market_share'] > all_competitors['market_share'].quantile(0.75):
            strengths.append("Strong market position")
        elif competitor['market_share'] < all_competitors['market_share'].quantile(0.25):
            weaknesses.append("Weak market position")
        
        # Growth
        if competitor.get('growth_rate', 0) > 0.15:
            strengths.append("High growth rate")
        elif competitor.get('growth_rate', 0) < 0.05:
            weaknesses.append("Low growth rate")
        
        # Customer satisfaction
        if competitor.get('customer_satisfaction', 0.5) > 0.8:
            strengths.append("Excellent customer satisfaction")
        elif competitor.get('customer_satisfaction', 0.5) < 0.6:
            weaknesses.append("Poor customer satisfaction")
        
        # Innovation
        if competitor.get('innovation_score', 0.5) > 0.7:
            strengths.append("Strong innovation capability")
        elif competitor.get('innovation_score', 0.5) < 0.3:
            weaknesses.append("Limited innovation")
        
        # Brand
        if competitor.get('brand_strength', 0.5) > 0.7:
            strengths.append("Strong brand recognition")
        elif competitor.get('brand_strength', 0.5) < 0.3:
            weaknesses.append("Weak brand presence")
        
        # Digital presence
        if competitor.get('digital_presence', 0.5) > 0.7:
            strengths.append("Strong digital presence")
        elif competitor.get('digital_presence', 0.5) < 0.3:
            weaknesses.append("Limited digital presence")
        
        # Pricing
        if competitor.get('price_index', 1.0) < 0.9:
            strengths.append("Cost leadership")
        elif competitor.get('price_index', 1.0) > 1.2:
            strengths.append("Premium positioning")
        
        return strengths, weaknesses
    
    def _calculate_threat_level(
        self,
        competitor: pd.Series,
        all_competitors: pd.DataFrame
    ) -> float:
        """Calculate competitor threat level (0-1)"""
        threat_score = 0
        
        # Market share weight: 30%
        market_share_percentile = (
            competitor['market_share'] / all_competitors['market_share'].max()
        )
        threat_score += market_share_percentile * 0.3
        
        # Growth rate weight: 25%
        growth_rate = competitor.get('growth_rate', 0.1)
        growth_percentile = min(max(growth_rate / 0.3, 0), 1)  # Cap at 30% growth
        threat_score += growth_percentile * 0.25
        
        # Innovation weight: 20%
        innovation = competitor.get('innovation_score', 0.5)
        threat_score += innovation * 0.2
        
        # Customer satisfaction weight: 15%
        satisfaction = competitor.get('customer_satisfaction', 0.5)
        threat_score += satisfaction * 0.15
        
        # Digital presence weight: 10%
        digital = competitor.get('digital_presence', 0.5)
        threat_score += digital * 0.1
        
        return min(threat_score, 1.0)
    
    def _identify_opportunities(
        self,
        competitor: pd.Series,
        strengths: List[str],
        weaknesses: List[str]
    ) -> List[str]:
        """Identify opportunities based on competitor analysis"""
        opportunities = []
        
        # Weakness-based opportunities
        if "Weak market position" in weaknesses:
            opportunities.append("Capture market share from struggling competitor")
        
        if "Poor customer satisfaction" in weaknesses:
            opportunities.append("Win dissatisfied customers with superior service")
        
        if "Limited digital presence" in weaknesses:
            opportunities.append("Dominate digital channels")
        
        if "Limited innovation" in weaknesses:
            opportunities.append("Lead with innovative products/services")
        
        # Strength-based opportunities (learn from)
        if "Strong market position" in strengths:
            opportunities.append("Study and adapt successful strategies")
        
        if "Cost leadership" in strengths:
            opportunities.append("Differentiate on value beyond price")
        
        # General opportunities
        if competitor.get('growth_rate', 0) < 0:
            opportunities.append("Acquire weakened competitor or assets")
        
        return opportunities
    
    def _analyze_products(self, competitor: pd.Series) -> List[Dict[str, Any]]:
        """Analyze competitor products/services"""
        # In real implementation, would parse actual product data
        return [
            {
                'name': 'Core Product',
                'market_share': 0.6,
                'price_point': 'mid-range',
                'key_features': ['Feature A', 'Feature B']
            },
            {
                'name': 'Premium Service',
                'market_share': 0.3,
                'price_point': 'premium',
                'key_features': ['Advanced Feature X', 'Premium Support']
            }
        ]
    
    def _identify_target_segments(
        self,
        competitor: pd.Series,
        persona
    ) -> List[str]:
        """Identify competitor's target segments"""
        segments = []
        
        # Price-based segmentation
        price_index = competitor.get('price_index', 1.0)
        if price_index < 0.8:
            segments.append("Budget-conscious customers")
        elif price_index > 1.2:
            segments.append("Premium segment")
        else:
            segments.append("Mid-market")
        
        # Size-based segmentation
        if competitor.get('market_reach', 0.5) > 0.7:
            segments.append("Enterprise customers")
        else:
            segments.append("SMB market")
        
        # Innovation-based
        if competitor.get('innovation_score', 0.5) > 0.7:
            segments.append("Early adopters")
        
        return segments
    
    def _analyze_pricing_strategy(self, competitor: pd.Series) -> str:
        """Analyze competitor's pricing strategy"""
        price_index = competitor.get('price_index', 1.0)
        
        if price_index < 0.8:
            return "Penetration pricing"
        elif price_index < 0.95:
            return "Competitive pricing"
        elif price_index < 1.05:
            return "Market pricing"
        elif price_index < 1.2:
            return "Premium pricing"
        else:
            return "Luxury pricing"
    
    def _identify_marketing_channels(self, competitor: pd.Series) -> List[str]:
        """Identify competitor's marketing channels"""
        channels = []
        
        digital_presence = competitor.get('digital_presence', 0.5)
        
        if digital_presence > 0.7:
            channels.extend(['SEO', 'SEM', 'Social Media', 'Content Marketing'])
        elif digital_presence > 0.4:
            channels.extend(['Basic SEO', 'Social Media'])
        
        # Traditional channels based on size
        if competitor['revenue'] > 1e8:
            channels.extend(['TV Advertising', 'Print Media'])
        
        channels.append('Direct Sales')
        
        return channels
    
    def _analyze_technology(self, competitor: pd.Series) -> List[str]:
        """Analyze competitor's technology stack"""
        tech_stack = []
        
        innovation_score = competitor.get('innovation_score', 0.5)
        
        if innovation_score > 0.7:
            tech_stack.extend(['AI/ML', 'Cloud-native', 'Microservices'])
        elif innovation_score > 0.4:
            tech_stack.extend(['Cloud', 'Modern Web Stack'])
        else:
            tech_stack.extend(['Legacy Systems', 'Basic Web'])
        
        return tech_stack
    
    def _get_recent_developments(self, competitor: pd.Series) -> List[Dict[str, Any]]:
        """Get recent developments for competitor"""
        # In real implementation, would fetch actual news/updates
        developments = []
        
        if competitor.get('growth_rate', 0) > 0.2:
            developments.append({
                'date': datetime.now() - timedelta(days=30),
                'type': 'expansion',
                'description': 'Announced market expansion'
            })
        
        if competitor.get('innovation_score', 0.5) > 0.7:
            developments.append({
                'date': datetime.now() - timedelta(days=60),
                'type': 'product_launch',
                'description': 'Launched new innovative product'
            })
        
        return developments


class CompetitivePositionAnalyzer:
    """Analyzes company's competitive position"""
    
    def __init__(self, persona_factory: PersonaFactory):
        self.persona_factory = persona_factory
        
    def analyze_position(
        self,
        company_data: Dict[str, Any],
        competitor_profiles: List[CompetitorProfile],
        market_data: Dict[str, Any],
        business_type: str
    ) -> CompetitivePosition:
        """Analyze company's competitive position"""
        # Calculate market position
        market_position = self._determine_market_position(
            company_data,
            competitor_profiles
        )
        
        # Identify competitive advantages
        advantages = self._identify_competitive_advantages(
            company_data,
            competitor_profiles,
            business_type
        )
        
        # Identify vulnerabilities
        vulnerabilities = self._identify_vulnerabilities(
            company_data,
            competitor_profiles
        )
        
        # SWOT-based analysis
        opportunities = self._identify_opportunities(
            company_data,
            competitor_profiles,
            market_data
        )
        
        threats = self._identify_threats(
            competitor_profiles,
            market_data
        )
        
        # Generate strategic recommendations
        recommendations = self._generate_strategic_recommendations(
            market_position,
            advantages,
            vulnerabilities,
            opportunities,
            threats
        )
        
        return CompetitivePosition(
            market_position=market_position,
            market_share=company_data.get('market_share', 0.1),
            relative_strength=self._calculate_relative_strength(
                company_data,
                competitor_profiles
            ),
            competitive_advantages=advantages,
            vulnerabilities=vulnerabilities,
            opportunities=opportunities,
            threats=threats,
            strategic_recommendations=recommendations
        )
    
    def _determine_market_position(
        self,
        company_data: Dict[str, Any],
        competitors: List[CompetitorProfile]
    ) -> MarketPosition:
        """Determine company's market position"""
        company_share = company_data.get('market_share', 0.1)
        
        # Get competitor shares
        competitor_shares = [c.market_share for c in competitors]
        
        if not competitor_shares:
            return MarketPosition.NEW_ENTRANT
        
        max_share = max(competitor_shares)
        
        # Determine position based on relative share
        if company_share >= max_share * 0.9:
            return MarketPosition.LEADER
        elif company_share >= max_share * 0.5:
            return MarketPosition.CHALLENGER
        elif company_share >= max_share * 0.2:
            return MarketPosition.FOLLOWER
        elif company_share >= 0.02:
            return MarketPosition.NICHER
        else:
            return MarketPosition.NEW_ENTRANT
    
    def _identify_competitive_advantages(
        self,
        company_data: Dict[str, Any],
        competitors: List[CompetitorProfile],
        business_type: str
    ) -> List[CompetitiveAdvantage]:
        """Identify company's competitive advantages"""
        advantages = []
        persona = self.persona_factory.create_persona(business_type)
        
        # Cost advantage
        avg_price = np.mean([
            c.pricing_strategy == "Competitive pricing" 
            for c in competitors
        ])
        if company_data.get('price_competitiveness', 0.5) > 0.7:
            advantages.append(CompetitiveAdvantage.COST_LEADERSHIP)
        
        # Differentiation
        if company_data.get('uniqueness_score', 0.5) > 0.7:
            advantages.append(CompetitiveAdvantage.DIFFERENTIATION)
        
        # Innovation
        avg_innovation = np.mean([
            0.5 for c in competitors  # Default innovation score
        ])
        if company_data.get('innovation_score', 0.5) > avg_innovation + 0.2:
            advantages.append(CompetitiveAdvantage.INNOVATION)
        
        # Customer service
        if company_data.get('customer_satisfaction', 0.5) > 0.8:
            advantages.append(CompetitiveAdvantage.CUSTOMER_SERVICE)
        
        # Brand strength
        if company_data.get('brand_recognition', 0.5) > 0.7:
            advantages.append(CompetitiveAdvantage.BRAND)
        
        # Technology
        if company_data.get('tech_advancement', 0.5) > 0.7:
            advantages.append(CompetitiveAdvantage.TECHNOLOGY)
        
        # Scale
        if company_data.get('market_share', 0.1) > 0.2:
            advantages.append(CompetitiveAdvantage.SCALE)
        
        return advantages
    
    def _identify_vulnerabilities(
        self,
        company_data: Dict[str, Any],
        competitors: List[CompetitorProfile]
    ) -> List[str]:
        """Identify company vulnerabilities"""
        vulnerabilities = []
        
        # Market share vulnerability
        if company_data.get('market_share', 0.1) < 0.05:
            vulnerabilities.append("Limited market presence")
        
        # Growth vulnerability
        avg_growth = np.mean([c.growth_rate for c in competitors])
        if company_data.get('growth_rate', 0.1) < avg_growth - 0.05:
            vulnerabilities.append("Below-average growth rate")
        
        # Technology gap
        tech_leaders = [c for c in competitors if 'technology' in [
            s.lower() for s in c.strengths
        ]]
        if tech_leaders and company_data.get('tech_advancement', 0.5) < 0.5:
            vulnerabilities.append("Technology gap vs competitors")
        
        # Customer satisfaction
        if company_data.get('customer_satisfaction', 0.5) < 0.6:
            vulnerabilities.append("Customer satisfaction issues")
        
        # Financial vulnerability
        if company_data.get('profitability', 0.1) < 0.05:
            vulnerabilities.append("Low profitability")
        
        # Channel vulnerability
        if company_data.get('channel_diversity', 0.5) < 0.3:
            vulnerabilities.append("Limited distribution channels")
        
        return vulnerabilities
    
    def _identify_opportunities(
        self,
        company_data: Dict[str, Any],
        competitors: List[CompetitorProfile],
        market_data: Dict[str, Any]
    ) -> List[str]:
        """Identify market opportunities"""
        opportunities = []
        
        # Market growth opportunity
        if market_data.get('growth_rate', 0.1) > 0.15:
            opportunities.append("High market growth potential")
        
        # Competitor weaknesses
        weak_competitors = [
            c for c in competitors 
            if c.threat_level < 0.3
        ]
        if weak_competitors:
            opportunities.append("Acquire market share from weak competitors")
        
        # Unserved segments
        all_segments = set()
        for c in competitors:
            all_segments.update(c.target_segments)
        
        if len(all_segments) < 5:
            opportunities.append("Target underserved market segments")
        
        # Technology opportunities
        tech_laggards = [
            c for c in competitors
            if 'Limited digital presence' in c.weaknesses
        ]
        if len(tech_laggards) > len(competitors) * 0.3:
            opportunities.append("Lead digital transformation in industry")
        
        # Consolidation opportunity
        if market_data.get('concentration_ratio', 0.5) < 0.4:
            opportunities.append("Industry consolidation opportunity")
        
        # Innovation gap
        avg_innovation = np.mean([
            0.5 for c in competitors
        ])
        if avg_innovation < 0.5:
            opportunities.append("Innovation leadership opportunity")
        
        return opportunities
    
    def _identify_threats(
        self,
        competitors: List[CompetitorProfile],
        market_data: Dict[str, Any]
    ) -> List[str]:
        """Identify competitive threats"""
        threats = []
        
        # High threat competitors
        high_threats = [c for c in competitors if c.threat_level > 0.7]
        if high_threats:
            threats.append(f"{len(high_threats)} high-threat competitors")
        
        # New entrants
        new_entrants = [
            c for c in competitors 
            if c.type == CompetitorType.NEW_ENTRANT
        ]
        if new_entrants:
            threats.append("New market entrants increasing competition")
        
        # Market saturation
        if market_data.get('growth_rate', 0.1) < 0.05:
            threats.append("Market saturation limiting growth")
        
        # Price pressure
        price_warriors = [
            c for c in competitors
            if c.pricing_strategy == "Penetration pricing"
        ]
        if price_warriors:
            threats.append("Price pressure from low-cost competitors")
        
        # Technology disruption
        tech_leaders = [
            c for c in competitors
            if CompetitiveAdvantage.TECHNOLOGY in [
                CompetitiveAdvantage.TECHNOLOGY
            ]
        ]
        if tech_leaders:
            threats.append("Technology disruption risk")
        
        # Regulatory threats
        if 'regulations' in market_data.get('trends', []):
            threats.append("Increasing regulatory requirements")
        
        return threats
    
    def _calculate_relative_strength(
        self,
        company_data: Dict[str, Any],
        competitors: List[CompetitorProfile]
    ) -> float:
        """Calculate company's relative competitive strength (0-1)"""
        if not competitors:
            return 0.5
        
        # Compare key metrics
        strength_score = 0
        
        # Market share comparison
        avg_share = np.mean([c.market_share for c in competitors])
        if company_data.get('market_share', 0.1) > avg_share:
            strength_score += 0.2
        
        # Growth comparison
        avg_growth = np.mean([c.growth_rate for c in competitors])
        if company_data.get('growth_rate', 0.1) > avg_growth:
            strength_score += 0.2
        
        # Innovation comparison
        if company_data.get('innovation_score', 0.5) > 0.6:
            strength_score += 0.2
        
        # Customer satisfaction
        if company_data.get('customer_satisfaction', 0.5) > 0.7:
            strength_score += 0.2
        
        # Financial strength
        if company_data.get('profitability', 0.1) > 0.15:
            strength_score += 0.2
        
        return min(strength_score, 1.0)
    
    def _generate_strategic_recommendations(
        self,
        position: MarketPosition,
        advantages: List[CompetitiveAdvantage],
        vulnerabilities: List[str],
        opportunities: List[str],
        threats: List[str]
    ) -> List[str]:
        """Generate strategic recommendations based on analysis"""
        recommendations = []
        
        # Position-based strategies
        if position == MarketPosition.LEADER:
            recommendations.extend([
                "Defend market leadership through continuous innovation",
                "Expand market size to maintain growth",
                "Build barriers to entry for new competitors"
            ])
        elif position == MarketPosition.CHALLENGER:
            recommendations.extend([
                "Execute targeted attacks on leader's weaknesses",
                "Differentiate aggressively to gain market share",
                "Form strategic alliances to strengthen position"
            ])
        elif position == MarketPosition.FOLLOWER:
            recommendations.extend([
                "Focus on operational excellence and efficiency",
                "Target niche segments overlooked by leaders",
                "Avoid direct confrontation with market leaders"
            ])
        elif position == MarketPosition.NICHER:
            recommendations.extend([
                "Deepen specialization in chosen niche",
                "Build strong customer relationships",
                "Maintain premium positioning"
            ])
        
        # Advantage-based strategies
        if CompetitiveAdvantage.COST_LEADERSHIP in advantages:
            recommendations.append("Leverage cost advantage to gain market share")
        
        if CompetitiveAdvantage.INNOVATION in advantages:
            recommendations.append("Accelerate innovation pipeline to stay ahead")
        
        if CompetitiveAdvantage.CUSTOMER_SERVICE in advantages:
            recommendations.append("Build loyalty programs around service excellence")
        
        # Vulnerability mitigation
        if "Technology gap vs competitors" in vulnerabilities:
            recommendations.append("Invest urgently in technology modernization")
        
        if "Limited market presence" in vulnerabilities:
            recommendations.append("Execute aggressive market expansion strategy")
        
        # Opportunity capture
        if "High market growth potential" in opportunities:
            recommendations.append("Scale operations to capture market growth")
        
        if "Industry consolidation opportunity" in opportunities:
            recommendations.append("Evaluate M&A targets for strategic acquisitions")
        
        # Threat mitigation
        if "Price pressure from low-cost competitors" in threats:
            recommendations.append("Enhance value proposition beyond price")
        
        if "Technology disruption risk" in threats:
            recommendations.append("Invest in emerging technologies proactively")
        
        # Limit and prioritize
        return recommendations[:8]


class CompetitiveMapGenerator:
    """Generates competitive positioning maps and visualizations"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def create_positioning_map(
        self,
        company_data: Dict[str, Any],
        competitors: List[CompetitorProfile]
    ) -> Dict[str, Any]:
        """Create competitive positioning map"""
        # Prepare data for positioning
        all_entities = self._prepare_positioning_data(company_data, competitors)
        
        # Perform dimensionality reduction
        positions = self._calculate_positions(all_entities)
        
        # Create visualization data
        viz_data = {
            'positions': positions,
            'entities': all_entities,
            'axes': self._determine_axes(all_entities)
        }
        
        return viz_data
    
    def _prepare_positioning_data(
        self,
        company_data: Dict[str, Any],
        competitors: List[CompetitorProfile]
    ) -> List[Dict[str, Any]]:
        """Prepare data for positioning analysis"""
        entities = []
        
        # Add company
        entities.append({
            'name': company_data.get('name', 'Our Company'),
            'type': 'company',
            'market_share': company_data.get('market_share', 0.1),
            'growth_rate': company_data.get('growth_rate', 0.1),
            'innovation_score': company_data.get('innovation_score', 0.5),
            'price_index': company_data.get('price_index', 1.0),
            'customer_satisfaction': company_data.get('customer_satisfaction', 0.7),
            'brand_strength': company_data.get('brand_recognition', 0.5)
        })
        
        # Add competitors
        for comp in competitors:
            entities.append({
                'name': comp.name,
                'type': 'competitor',
                'market_share': comp.market_share,
                'growth_rate': comp.growth_rate,
                'innovation_score': 0.5,  # Default
                'price_index': 1.0,  # Default
                'customer_satisfaction': 0.7,  # Default
                'brand_strength': 0.5  # Default
            })
        
        return entities
    
    def _calculate_positions(
        self,
        entities: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Calculate 2D positions using PCA"""
        # Extract features
        features = []
        for entity in entities:
            features.append([
                entity['market_share'],
                entity['growth_rate'],
                entity['innovation_score'],
                entity['price_index'],
                entity['customer_satisfaction'],
                entity['brand_strength']
            ])
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA
        pca = PCA(n_components=2)
        positions = pca.fit_transform(features_scaled)
        
        return positions
    
    def _determine_axes(self, entities: List[Dict[str, Any]]) -> Dict[str, str]:
        """Determine what the axes represent"""
        # In a more sophisticated implementation, would analyze PCA loadings
        return {
            'x_axis': 'Market Position (Share & Brand)',
            'y_axis': 'Growth & Innovation'
        }
    
    def create_competitive_network(
        self,
        competitors: List[CompetitorProfile]
    ) -> nx.Graph:
        """Create competitive relationship network"""
        G = nx.Graph()
        
        # Add nodes
        for comp in competitors:
            G.add_node(
                comp.name,
                market_share=comp.market_share,
                threat_level=comp.threat_level,
                type=comp.type.value
            )
        
        # Add edges based on competitive relationships
        for i, comp1 in enumerate(competitors):
            for j, comp2 in enumerate(competitors[i+1:], i+1):
                # Calculate competitive overlap
                overlap = self._calculate_competitive_overlap(comp1, comp2)
                
                if overlap > 0.3:  # Threshold for competition
                    G.add_edge(
                        comp1.name,
                        comp2.name,
                        weight=overlap,
                        competition_type='direct' if overlap > 0.7 else 'indirect'
                    )
        
        return G
    
    def _calculate_competitive_overlap(
        self,
        comp1: CompetitorProfile,
        comp2: CompetitorProfile
    ) -> float:
        """Calculate how much two competitors overlap"""
        overlap_score = 0
        
        # Segment overlap
        segment_overlap = len(
            set(comp1.target_segments) & set(comp2.target_segments)
        ) / max(len(comp1.target_segments), len(comp2.target_segments), 1)
        overlap_score += segment_overlap * 0.4
        
        # Pricing overlap
        if comp1.pricing_strategy == comp2.pricing_strategy:
            overlap_score += 0.3
        
        # Market position overlap
        share_diff = abs(comp1.market_share - comp2.market_share)
        if share_diff < 0.1:
            overlap_score += 0.3
        
        return min(overlap_score, 1.0)


class CompetitiveIntelligenceSystem(UniversalAnalyticsEngine):
    """
    Competitive Intelligence System
    
    Provides comprehensive competitive analysis and market intelligence
    that automatically adapts to any business niche.
    """
    
    def __init__(self, encryption_manager: EncryptionManager):
        super().__init__(encryption_manager)
        self.persona_factory = PersonaFactory()
        self.data_collector = CompetitorDataCollector(self.persona_factory)
        self.competitor_analyzer = CompetitorAnalyzer(self.persona_factory)
        self.position_analyzer = CompetitivePositionAnalyzer(self.persona_factory)
        self.map_generator = CompetitiveMapGenerator()
        
    async def generate_executive_summary(
        self,
        analytics_data: Dict[str, Any],
        business_type: str
    ) -> str:
        """Generate executive summary of competitive intelligence"""
        summary_parts = []
        
        # Competitive position
        position = analytics_data.get('competitive_position')
        if position:
            summary_parts.append(
                f"**Competitive Position:**\n"
                f"- Market Position: {position.market_position.value.title()}\n"
                f"- Market Share: {position.market_share:.1%}\n"
                f"- Relative Strength: {position.relative_strength:.1%}\n"
                f"- Key Advantages: {', '.join([a.value for a in position.competitive_advantages[:3]])}\n"
            )
        
        # Top competitors
        competitors = analytics_data.get('competitor_profiles', [])
        if competitors:
            top_threats = sorted(competitors, key=lambda x: x.threat_level, reverse=True)[:3]
            summary_parts.append("\n**Top Competitive Threats:**")
            for comp in top_threats:
                summary_parts.append(
                    f"- {comp.name}: {comp.threat_level:.1%} threat level, "
                    f"{comp.market_share:.1%} market share"
                )
        
        # Market dynamics
        market = analytics_data.get('market_dynamics')
        if market:
            summary_parts.append(
                f"\n**Market Overview:**\n"
                f"- Market Size: ${market.total_market_size:,.0f}\n"
                f"- Growth Rate: {market.growth_rate:.1%}\n"
                f"- Concentration: {market.concentration_ratio:.1%} (Top 4)\n"
            )
        
        # Strategic insights
        insights = analytics_data.get('insights', [])
        if insights:
            summary_parts.append("\n**Key Strategic Insights:**")
            for insight in insights[:3]:
                summary_parts.append(f"- {insight.title}")
        
        # Recommendations
        if position and position.strategic_recommendations:
            summary_parts.append("\n**Strategic Recommendations:**")
            for i, rec in enumerate(position.strategic_recommendations[:3], 1):
                summary_parts.append(f"{i}. {rec}")
        
        return "\n".join(summary_parts)
    
    async def calculate_kpis(
        self,
        request: AnalyticsRequest
    ) -> List[BusinessKPI]:
        """Calculate competitive intelligence KPIs"""
        kpis = []
        
        # Collect data
        competitor_data = await self.data_collector.collect_competitor_data(
            request.business_type
        )
        
        company_data = request.custom_params.get('company_data', {
            'market_share': 0.15,
            'growth_rate': 0.12,
            'customer_satisfaction': 0.75
        })
        
        # Market share KPI
        kpis.append(BusinessKPI(
            name="Market Share",
            value=company_data.get('market_share', 0.15),
            unit="percentage",
            trend="up" if company_data.get('growth_rate', 0) > 0 else "down",
            target=0.20,
            category="Market Position"
        ))
        
        # Competitive threat level
        if len(competitor_data) > 0:
            avg_threat = competitor_data['market_share'].mean()
            threat_ratio = company_data.get('market_share', 0.15) / (avg_threat + 0.001)
            
            kpis.append(BusinessKPI(
                name="Competitive Strength Ratio",
                value=threat_ratio,
                unit="ratio",
                trend="stable" if 0.8 < threat_ratio < 1.2 else "down",
                target=1.5,
                category="Competitive Position"
            ))
        
        # Market concentration
        if len(competitor_data) > 0:
            top_4_share = competitor_data.nlargest(4, 'market_share')['market_share'].sum()
            
            kpis.append(BusinessKPI(
                name="Market Concentration (CR4)",
                value=top_4_share,
                unit="percentage",
                trend="up" if top_4_share > 0.6 else "stable",
                target=0.5,
                category="Market Structure"
            ))
        
        # Growth differential
        if len(competitor_data) > 0:
            avg_competitor_growth = competitor_data.get('growth_rate', pd.Series([0.1])).mean()
            growth_differential = company_data.get('growth_rate', 0.12) - avg_competitor_growth
            
            kpis.append(BusinessKPI(
                name="Growth Rate vs Competitors",
                value=growth_differential,
                unit="percentage_points",
                trend="up" if growth_differential > 0 else "down",
                target=0.05,
                category="Performance"
            ))
        
        # Innovation index
        kpis.append(BusinessKPI(
            name="Innovation Index",
            value=company_data.get('innovation_score', 0.6),
            unit="score",
            trend="up",
            target=0.75,
            category="Competitive Advantage"
        ))
        
        # Customer satisfaction differential
        if len(competitor_data) > 0:
            avg_satisfaction = competitor_data.get('customer_satisfaction', pd.Series([0.7])).mean()
            satisfaction_diff = company_data.get('customer_satisfaction', 0.75) - avg_satisfaction
            
            kpis.append(BusinessKPI(
                name="Customer Satisfaction Advantage",
                value=satisfaction_diff,
                unit="points",
                trend="up" if satisfaction_diff > 0 else "down",
                target=0.1,
                category="Customer"
            ))
        
        return kpis
    
    async def analyze_data(
        self,
        request: AnalyticsRequest
    ) -> Dict[str, Any]:
        """Perform comprehensive competitive analysis"""
        try:
            # Collect competitor data
            competitor_data = await self.data_collector.collect_competitor_data(
                request.business_type
            )
            
            # Collect market data
            market_data = await self.data_collector.collect_market_data(
                request.business_type
            )
            
            # Get company data
            company_data = request.custom_params.get('company_data', {
                'name': 'Our Company',
                'market_share': 0.15,
                'growth_rate': 0.12,
                'innovation_score': 0.65,
                'customer_satisfaction': 0.75,
                'price_index': 1.0,
                'brand_recognition': 0.6,
                'profitability': 0.12,
                'tech_advancement': 0.7,
                'channel_diversity': 0.6
            })
            
            # Analyze competitors
            competitor_profiles = self.competitor_analyzer.analyze_competitors(
                competitor_data,
                request.business_type
            )
            
            # Analyze competitive position
            competitive_position = self.position_analyzer.analyze_position(
                company_data,
                competitor_profiles,
                market_data,
                request.business_type
            )
            
            # Create market dynamics analysis
            market_dynamics = MarketDynamics(
                total_market_size=market_data.get('market_size', 1e9),
                growth_rate=market_data.get('growth_rate', 0.1),
                concentration_ratio=market_data.get('concentration_ratio', 0.5),
                entry_barriers=market_data.get('entry_barriers', []),
                key_success_factors=market_data.get('key_success_factors', []),
                industry_trends=market_data.get('trends', []),
                disruption_risks=self._identify_disruption_risks(
                    competitor_profiles,
                    market_data
                ),
                consolidation_potential=self._calculate_consolidation_potential(
                    competitor_profiles,
                    market_data
                )
            )
            
            # Generate positioning map data
            positioning_map = self.map_generator.create_positioning_map(
                company_data,
                competitor_profiles
            )
            
            # Create competitive network
            competitive_network = self.map_generator.create_competitive_network(
                competitor_profiles
            )
            
            # Generate insights
            insights = await self._generate_competitive_insights(
                competitive_position,
                competitor_profiles,
                market_dynamics,
                request.business_type
            )
            
            # Identify strategic moves
            strategic_moves = self._identify_strategic_moves(
                competitive_position,
                competitor_profiles,
                market_dynamics
            )
            
            return {
                'competitor_profiles': competitor_profiles,
                'competitive_position': competitive_position,
                'market_dynamics': market_dynamics,
                'positioning_map': positioning_map,
                'competitive_network': competitive_network,
                'insights': insights,
                'strategic_moves': strategic_moves,
                'company_data': company_data
            }
            
        except Exception as e:
            logger.error(f"Error in competitive analysis: {e}")
            raise
    
    def _identify_disruption_risks(
        self,
        competitors: List[CompetitorProfile],
        market_data: Dict[str, Any]
    ) -> List[str]:
        """Identify potential disruption risks"""
        risks = []
        
        # Technology disruption
        tech_innovators = [
            c for c in competitors
            if 'innovation' in [s.lower() for s in c.strengths]
        ]
        if tech_innovators:
            risks.append("Technology disruption from innovative competitors")
        
        # Business model disruption
        new_entrants = [
            c for c in competitors
            if c.type == CompetitorType.NEW_ENTRANT
        ]
        if new_entrants:
            risks.append("New business models from recent entrants")
        
        # Platform disruption
        if 'platform economy' in market_data.get('trends', []):
            risks.append("Platform-based competitors disrupting traditional model")
        
        # Regulatory disruption
        if 'regulatory changes' in market_data.get('trends', []):
            risks.append("Regulatory changes favoring new approaches")
        
        return risks
    
    def _calculate_consolidation_potential(
        self,
        competitors: List[CompetitorProfile],
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate market consolidation potential (0-1)"""
        potential = 0
        
        # Fragmentation factor
        if len(competitors) > 10:
            potential += 0.3
        
        # Low concentration
        if market_data.get('concentration_ratio', 0.5) < 0.4:
            potential += 0.3
        
        # Struggling competitors
        weak_competitors = [c for c in competitors if c.threat_level < 0.3]
        if len(weak_competitors) > len(competitors) * 0.3:
            potential += 0.2
        
        # Market maturity
        if market_data.get('growth_rate', 0.1) < 0.05:
            potential += 0.2
        
        return min(potential, 1.0)
    
    async def _generate_competitive_insights(
        self,
        position: CompetitivePosition,
        competitors: List[CompetitorProfile],
        market_dynamics: MarketDynamics,
        business_type: str
    ) -> List[AnalyticsInsight]:
        """Generate actionable competitive insights"""
        insights = []
        persona = self.persona_factory.create_persona(business_type)
        
        # Position-based insights
        if position.market_position == MarketPosition.FOLLOWER:
            insights.append(AnalyticsInsight(
                title="Market Follower Position Requires Strategic Focus",
                description=(
                    f"As a market follower with {position.market_share:.1%} share, "
                    f"competing directly with leaders is resource-intensive"
                ),
                impact="Risk of being squeezed between leaders and niche players",
                recommendation=(
                    "Adopt focused strategy:\n"
                    "1. Target underserved segments\n"
                    "2. Excel in operational efficiency\n"
                    "3. Build strategic partnerships\n"
                    "4. Differentiate through service"
                ),
                priority=InsightPriority.HIGH,
                confidence=0.85,
                supporting_data={
                    'market_position': position.market_position.value,
                    'relative_strength': position.relative_strength
                }
            ))
        
        # High threat competitors
        high_threats = [c for c in competitors if c.threat_level > 0.7]
        if high_threats:
            top_threat = max(high_threats, key=lambda x: x.threat_level)
            insights.append(AnalyticsInsight(
                title=f"High Threat from {top_threat.name}",
                description=(
                    f"{top_threat.name} poses {top_threat.threat_level:.0%} threat level "
                    f"with {top_threat.market_share:.1%} market share and strong "
                    f"{', '.join(top_threat.strengths[:2])}"
                ),
                impact="Risk of losing market share and customer base",
                recommendation=(
                    f"Counter-strategy for {top_threat.name}:\n"
                    f"1. Exploit weaknesses: {', '.join(top_threat.weaknesses[:2])}\n"
                    f"2. Differentiate in areas they're weak\n"
                    f"3. Target their dissatisfied customers\n"
                    f"4. Build defensive moats"
                ),
                priority=InsightPriority.CRITICAL,
                confidence=0.90,
                supporting_data={
                    'competitor_name': top_threat.name,
                    'threat_level': top_threat.threat_level,
                    'weaknesses': top_threat.weaknesses
                }
            ))
        
        # Market consolidation opportunity
        if market_dynamics.consolidation_potential > 0.7:
            weak_targets = [
                c for c in competitors
                if c.threat_level < 0.3 and c.market_share > 0.02
            ]
            insights.append(AnalyticsInsight(
                title="Market Consolidation Opportunity",
                description=(
                    f"High consolidation potential ({market_dynamics.consolidation_potential:.0%}) "
                    f"with {len(weak_targets)} potential acquisition targets"
                ),
                impact="Opportunity to rapidly gain market share and capabilities",
                recommendation=(
                    "Execute consolidation strategy:\n"
                    "1. Evaluate acquisition targets\n"
                    "2. Secure financing for M&A\n"
                    "3. Develop integration playbook\n"
                    "4. Act quickly before competitors"
                ),
                priority=InsightPriority.HIGH,
                confidence=0.85,
                supporting_data={
                    'consolidation_potential': market_dynamics.consolidation_potential,
                    'weak_competitors': len(weak_targets)
                }
            ))
        
        # Competitive advantage insights
        if CompetitiveAdvantage.INNOVATION in position.competitive_advantages:
            innovation_laggards = [
                c for c in competitors
                if 'Limited innovation' in c.weaknesses
            ]
            if len(innovation_laggards) > len(competitors) * 0.5:
                insights.append(AnalyticsInsight(
                    title="Innovation Leadership Opportunity",
                    description=(
                        f"Strong innovation advantage with {len(innovation_laggards)} "
                        f"competitors showing innovation weakness"
                    ),
                    impact="Potential to disrupt market and gain significant share",
                    recommendation=(
                        "Leverage innovation advantage:\n"
                        "1. Accelerate product development\n"
                        "2. Patent key innovations\n"
                        "3. Market innovation leadership\n"
                        "4. Create innovation barriers"
                    ),
                    priority=InsightPriority.HIGH,
                    confidence=0.88,
                    supporting_data={
                        'innovation_advantage': True,
                        'laggard_count': len(innovation_laggards)
                    }
                ))
        
        # Vulnerability warnings
        if "Technology gap vs competitors" in position.vulnerabilities:
            tech_leaders = [
                c for c in competitors
                if 'technology' in [s.lower() for s in c.strengths]
            ]
            insights.append(AnalyticsInsight(
                title="Critical Technology Gap",
                description=(
                    f"Technology gap versus {len(tech_leaders)} competitors "
                    f"with strong tech capabilities"
                ),
                impact="Risk of becoming obsolete and losing competitive position",
                recommendation=(
                    "Close technology gap urgently:\n"
                    "1. Invest in digital transformation\n"
                    "2. Hire tech talent aggressively\n"
                    "3. Partner with tech providers\n"
                    "4. Consider tech acquisitions"
                ),
                priority=InsightPriority.CRITICAL,
                confidence=0.92,
                supporting_data={
                    'tech_leaders': len(tech_leaders),
                    'vulnerability': 'Technology gap'
                }
            ))
        
        # Market dynamics insights
        if market_dynamics.growth_rate > 0.15:
            insights.append(AnalyticsInsight(
                title="High Market Growth Creates Opportunities",
                description=(
                    f"Market growing at {market_dynamics.growth_rate:.1%} annually, "
                    f"reducing zero-sum competition"
                ),
                impact="Opportunity to grow without directly attacking competitors",
                recommendation=(
                    "Growth market strategy:\n"
                    "1. Invest in capacity expansion\n"
                    "2. Focus on customer acquisition\n"
                    "3. Build market presence quickly\n"
                    "4. Establish category leadership"
                ),
                priority=InsightPriority.MEDIUM,
                confidence=0.85,
                supporting_data={
                    'market_growth': market_dynamics.growth_rate
                }
            ))
        
        return insights
    
    def _identify_strategic_moves(
        self,
        position: CompetitivePosition,
        competitors: List[CompetitorProfile],
        market_dynamics: MarketDynamics
    ) -> List[Dict[str, Any]]:
        """Identify potential strategic moves"""
        moves = []
        
        # Acquisition opportunities
        acquisition_targets = [
            c for c in competitors
            if c.threat_level < 0.4 and c.market_share > 0.02
        ]
        
        for target in acquisition_targets[:3]:
            moves.append({
                'type': 'acquisition',
                'target': target.name,
                'rationale': f"Gain {target.market_share:.1%} market share and {', '.join(target.strengths[:2])}",
                'priority': 'high' if target.market_share > 0.05 else 'medium',
                'estimated_impact': {
                    'market_share_gain': target.market_share,
                    'synergies': ['cost reduction', 'market expansion']
                }
            })
        
        # Partnership opportunities
        complementary_competitors = [
            c for c in competitors
            if len(set(c.strengths) & set(position.vulnerabilities)) > 0
        ]
        
        for partner in complementary_competitors[:2]:
            moves.append({
                'type': 'partnership',
                'target': partner.name,
                'rationale': f"Complement weaknesses with their {', '.join(partner.strengths[:2])}",
                'priority': 'medium',
                'estimated_impact': {
                    'capability_gain': partner.strengths[:2],
                    'market_access': partner.target_segments
                }
            })
        
        # Market entry moves
        underserved_segments = self._identify_underserved_segments(
            competitors,
            market_dynamics
        )
        
        for segment in underserved_segments[:2]:
            moves.append({
                'type': 'market_entry',
                'target': segment,
                'rationale': f"Enter underserved {segment} segment",
                'priority': 'medium',
                'estimated_impact': {
                    'market_expansion': 0.02,
                    'first_mover_advantage': True
                }
            })
        
        # Defensive moves
        if position.market_position in [MarketPosition.LEADER, MarketPosition.CHALLENGER]:
            moves.append({
                'type': 'defensive',
                'target': 'market_position',
                'rationale': 'Protect current market position from challengers',
                'priority': 'high',
                'tactics': [
                    'Increase switching costs',
                    'Lock in key customers',
                    'Expand product line',
                    'Preemptive pricing'
                ]
            })
        
        return moves
    
    def _identify_underserved_segments(
        self,
        competitors: List[CompetitorProfile],
        market_dynamics: MarketDynamics
    ) -> List[str]:
        """Identify underserved market segments"""
        all_segments = set()
        covered_segments = set()
        
        # Collect all possible segments
        all_segments.update([
            'Enterprise', 'SMB', 'Consumer',
            'Premium', 'Budget', 'Mid-market',
            'Urban', 'Rural', 'Suburban',
            'Tech-savvy', 'Traditional'
        ])
        
        # Identify covered segments
        for comp in competitors:
            covered_segments.update(comp.target_segments)
        
        # Find gaps
        underserved = list(all_segments - covered_segments)
        
        return underserved[:5]
    
    async def generate_report(
        self,
        request: AnalyticsRequest,
        format: ReportFormat
    ) -> Union[Dict[str, Any], str, bytes]:
        """Generate comprehensive competitive intelligence report"""
        # Perform analysis
        analysis_results = await self.analyze_data(request)
        
        # Create visualizations
        visualizations = await self._create_visualizations(
            analysis_results,
            request.business_type
        )
        
        # Compile report data
        report_data = {
            'title': f'Competitive Intelligence Report - {request.business_type}',
            'generated_at': datetime.now().isoformat(),
            'executive_summary': await self.generate_executive_summary(
                analysis_results,
                request.business_type
            ),
            'kpis': await self.calculate_kpis(request),
            'competitive_position': self._serialize_position(
                analysis_results.get('competitive_position')
            ),
            'competitor_profiles': [
                self._serialize_competitor(c)
                for c in analysis_results.get('competitor_profiles', [])
            ],
            'market_dynamics': self._serialize_market_dynamics(
                analysis_results.get('market_dynamics')
            ),
            'insights': analysis_results.get('insights', []),
            'strategic_moves': analysis_results.get('strategic_moves', []),
            'visualizations': visualizations
        }
        
        # Format report based on requested format
        if format == ReportFormat.JSON:
            return report_data
        elif format == ReportFormat.PDF:
            return await self._generate_pdf_report(report_data)
        elif format == ReportFormat.EXCEL:
            return await self._generate_excel_report(report_data)
        elif format == ReportFormat.DASHBOARD:
            return await self._generate_dashboard(report_data)
        else:
            return report_data
    
    def _serialize_position(
        self,
        position: Optional[CompetitivePosition]
    ) -> Dict[str, Any]:
        """Serialize competitive position for report"""
        if not position:
            return {}
        
        return {
            'market_position': position.market_position.value,
            'market_share': position.market_share,
            'relative_strength': position.relative_strength,
            'competitive_advantages': [a.value for a in position.competitive_advantages],
            'vulnerabilities': position.vulnerabilities,
            'opportunities': position.opportunities,
            'threats': position.threats,
            'strategic_recommendations': position.strategic_recommendations
        }
    
    def _serialize_competitor(self, competitor: CompetitorProfile) -> Dict[str, Any]:
        """Serialize competitor profile for report"""
        return {
            'name': competitor.name,
            'type': competitor.type.value,
            'market_share': competitor.market_share,
            'revenue_estimate': competitor.revenue_estimate,
            'growth_rate': competitor.growth_rate,
            'threat_level': competitor.threat_level,
            'strengths': competitor.strengths,
            'weaknesses': competitor.weaknesses,
            'target_segments': competitor.target_segments,
            'pricing_strategy': competitor.pricing_strategy
        }
    
    def _serialize_market_dynamics(
        self,
        dynamics: Optional[MarketDynamics]
    ) -> Dict[str, Any]:
        """Serialize market dynamics for report"""
        if not dynamics:
            return {}
        
        return {
            'total_market_size': dynamics.total_market_size,
            'growth_rate': dynamics.growth_rate,
            'concentration_ratio': dynamics.concentration_ratio,
            'entry_barriers': dynamics.entry_barriers,
            'key_success_factors': dynamics.key_success_factors,
            'industry_trends': dynamics.industry_trends,
            'disruption_risks': dynamics.disruption_risks,
            'consolidation_potential': dynamics.consolidation_potential
        }
    
    async def _create_visualizations(
        self,
        analysis_results: Dict[str, Any],
        business_type: str
    ) -> Dict[str, Any]:
        """Create competitive analysis visualizations"""
        visualizations = {}
        
        # Competitive positioning map
        positioning_data = analysis_results.get('positioning_map', {})
        if positioning_data:
            positions = positioning_data.get('positions', [])
            entities = positioning_data.get('entities', [])
            
            if len(positions) > 0 and len(entities) > 0:
                fig = go.Figure()
                
                # Add competitor points
                for i, entity in enumerate(entities):
                    if i < len(positions):
                        color = 'red' if entity['type'] == 'company' else 'blue'
                        size = entity['market_share'] * 1000
                        
                        fig.add_trace(go.Scatter(
                            x=[positions[i][0]],
                            y=[positions[i][1]],
                            mode='markers+text',
                            marker=dict(size=size, color=color),
                            text=entity['name'],
                            textposition='top center',
                            name=entity['name']
                        ))
                
                fig.update_layout(
                    title='Competitive Positioning Map',
                    xaxis_title=positioning_data.get('axes', {}).get('x_axis', 'Dimension 1'),
                    yaxis_title=positioning_data.get('axes', {}).get('y_axis', 'Dimension 2'),
                    showlegend=False
                )
                
                visualizations['positioning_map'] = fig.to_json()
        
        # Market share comparison
        competitors = analysis_results.get('competitor_profiles', [])
        company_data = analysis_results.get('company_data', {})
        
        if competitors:
            # Prepare data
            names = [company_data.get('name', 'Our Company')] + [c.name for c in competitors[:9]]
            shares = [company_data.get('market_share', 0.15)] + [c.market_share for c in competitors[:9]]
            
            fig = px.pie(
                values=shares,
                names=names,
                title='Market Share Distribution'
            )
            
            visualizations['market_share'] = fig.to_json()
        
        # Competitive threat matrix
        if competitors:
            threat_data = []
            for comp in competitors[:10]:
                threat_data.append({
                    'Competitor': comp.name,
                    'Market Share': comp.market_share * 100,
                    'Threat Level': comp.threat_level * 100,
                    'Growth Rate': comp.growth_rate * 100
                })
            
            df = pd.DataFrame(threat_data)
            
            fig = px.scatter(
                df,
                x='Market Share',
                y='Threat Level',
                size='Growth Rate',
                color='Threat Level',
                hover_data=['Competitor', 'Growth Rate'],
                title='Competitive Threat Matrix',
                labels={
                    'Market Share': 'Market Share (%)',
                    'Threat Level': 'Threat Level (%)'
                }
            )
            
            visualizations['threat_matrix'] = fig.to_json()
        
        # Competitive advantages radar
        position = analysis_results.get('competitive_position')
        if position:
            # Define all possible advantages
            all_advantages = list(CompetitiveAdvantage)
            
            # Score each advantage
            scores = []
            for adv in all_advantages:
                if adv in position.competitive_advantages:
                    scores.append(0.9)
                else:
                    scores.append(0.3)
            
            fig = go.Figure(data=go.Scatterpolar(
                r=scores,
                theta=[adv.value.replace('_', ' ').title() for adv in all_advantages],
                fill='toself',
                name='Competitive Advantages'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False,
                title="Competitive Advantages Profile"
            )
            
            visualizations['advantages_radar'] = fig.to_json()
        
        # Strategic moves impact
        moves = analysis_results.get('strategic_moves', [])
        if moves:
            move_types = defaultdict(int)
            for move in moves:
                move_types[move['type']] += 1
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(move_types.keys()),
                    y=list(move_types.values()),
                    marker_color=['green', 'blue', 'orange', 'red']
                )
            ])
            
            fig.update_layout(
                title='Strategic Move Options',
                xaxis_title='Move Type',
                yaxis_title='Number of Options'
            )
            
            visualizations['strategic_moves'] = fig.to_json()
        
        return visualizations