"""A/B Testing Framework for Business Intelligence"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import logging
import uuid
from collections import defaultdict
import numpy as np
from scipy import stats
import hashlib

logger = logging.getLogger(__name__)

class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class ExperimentType(Enum):
    PRICING = "pricing"
    CONTENT = "content"
    FEATURE = "feature"
    UI = "ui"

@dataclass
class ExperimentVariant:
    """Individual variant in an experiment"""
    variant_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    allocation_percentage: float
    is_control: bool = False

@dataclass
class ExperimentMetrics:
    """Metrics tracked for experiment"""
    primary_metric: str
    secondary_metrics: List[str]
    success_criteria: Dict[str, float]
    minimum_sample_size: int
    confidence_level: float = 0.95

@dataclass
class ExperimentResult:
    """Results from an experiment"""
    variant_id: str
    sample_size: int
    conversion_rate: float
    average_value: float
    confidence_interval: Tuple[float, float]
    p_value: float
    is_significant: bool
    lift_percentage: float

@dataclass
class Experiment:
    """A/B test experiment configuration"""
    experiment_id: str
    name: str
    description: str
    experiment_type: ExperimentType
    variants: List[ExperimentVariant]
    metrics: ExperimentMetrics
    segment_criteria: Dict[str, Any]
    start_date: datetime
    end_date: Optional[datetime]
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    results: Optional[List[ExperimentResult]] = None

class ABTestingEngine:
    """Scientific A/B testing for business optimization"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.active_experiments = {}
        self.experiment_history = []
        self.assignment_cache = {}  # Cache user-experiment assignments
        
    async def create_pricing_experiment(
        self,
        control_price: float,
        test_prices: List[float],
        segment: str,
        experiment_name: str,
        duration_days: int = 14
    ) -> str:
        """Create a pricing A/B test experiment"""
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        
        # Create variants
        variants = [
            ExperimentVariant(
                variant_id=f"control_{experiment_id}",
                name="Control",
                description=f"Current price: ${control_price}",
                parameters={"price": control_price},
                allocation_percentage=50.0,
                is_control=True
            )
        ]
        
        # Add test variants
        test_allocation = 50.0 / len(test_prices)
        for i, test_price in enumerate(test_prices):
            variants.append(
                ExperimentVariant(
                    variant_id=f"test_{i}_{experiment_id}",
                    name=f"Test {i+1}",
                    description=f"Test price: ${test_price}",
                    parameters={"price": test_price},
                    allocation_percentage=test_allocation,
                    is_control=False
                )
            )
        
        # Define metrics
        metrics = ExperimentMetrics(
            primary_metric="revenue_per_user",
            secondary_metrics=["conversion_rate", "churn_rate", "average_order_value"],
            success_criteria={
                "revenue_per_user": 0.05,  # 5% lift required
                "conversion_rate": -0.10   # Max 10% drop allowed
            },
            minimum_sample_size=self._calculate_sample_size(
                baseline_rate=0.10,  # Assume 10% baseline conversion
                minimum_effect=0.05,  # 5% relative change
                power=0.80
            )
        )
        
        # Create experiment
        experiment = Experiment(
            experiment_id=experiment_id,
            name=experiment_name,
            description=f"Pricing experiment for {segment} segment",
            experiment_type=ExperimentType.PRICING,
            variants=variants,
            metrics=metrics,
            segment_criteria={"segment": segment},
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=duration_days)
        )
        
        self.active_experiments[experiment_id] = experiment
        
        logger.info(f"Created pricing experiment {experiment_id} for segment {segment}")
        
        return experiment_id
    
    async def create_content_experiment(
        self,
        content_variations: List[Dict[str, Any]],
        target_audience: str,
        experiment_name: str,
        duration_days: int = 7
    ) -> str:
        """Create a content A/B test experiment"""
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        
        # Create variants
        variants = []
        allocation = 100.0 / len(content_variations)
        
        for i, variation in enumerate(content_variations):
            variants.append(
                ExperimentVariant(
                    variant_id=f"variant_{i}_{experiment_id}",
                    name=variation.get("name", f"Variant {i+1}"),
                    description=variation.get("description", ""),
                    parameters=variation,
                    allocation_percentage=allocation,
                    is_control=(i == 0)
                )
            )
        
        # Define metrics for content
        metrics = ExperimentMetrics(
            primary_metric="engagement_rate",
            secondary_metrics=["click_through_rate", "share_rate", "conversion_rate"],
            success_criteria={
                "engagement_rate": 0.10,  # 10% lift required
                "click_through_rate": 0.05  # 5% lift required
            },
            minimum_sample_size=self._calculate_sample_size(
                baseline_rate=0.05,  # 5% baseline engagement
                minimum_effect=0.10,  # 10% relative change
                power=0.80
            )
        )
        
        # Create experiment
        experiment = Experiment(
            experiment_id=experiment_id,
            name=experiment_name,
            description=f"Content experiment for {target_audience} audience",
            experiment_type=ExperimentType.CONTENT,
            variants=variants,
            metrics=metrics,
            segment_criteria={"audience": target_audience},
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=duration_days)
        )
        
        self.active_experiments[experiment_id] = experiment
        
        return experiment_id
    
    def _calculate_sample_size(
        self,
        baseline_rate: float,
        minimum_effect: float,
        power: float = 0.80,
        significance: float = 0.05
    ) -> int:
        """Calculate required sample size for statistical significance"""
        # Using formula for two-proportion z-test
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_effect)
        
        # Average proportion
        p_avg = (p1 + p2) / 2
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - significance / 2)
        z_beta = stats.norm.ppf(power)
        
        # Sample size calculation
        n = (2 * p_avg * (1 - p_avg) * (z_alpha + z_beta) ** 2) / ((p2 - p1) ** 2)
        
        return int(np.ceil(n))
    
    async def assign_user_to_variant(self, experiment_id: str, user_id: str) -> str:
        """Assign a user to an experiment variant"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        
        # Check if experiment is running
        if experiment.status != ExperimentStatus.RUNNING:
            # Return control variant if experiment not running
            control_variant = next(v for v in experiment.variants if v.is_control)
            return control_variant.variant_id
        
        # Check cache for existing assignment
        cache_key = f"{experiment_id}:{user_id}"
        if cache_key in self.assignment_cache:
            return self.assignment_cache[cache_key]
        
        # Hash user ID for consistent assignment
        hash_value = int(hashlib.md5(f"{experiment_id}:{user_id}".encode()).hexdigest(), 16)
        assignment_value = (hash_value % 100) / 100.0
        
        # Assign to variant based on allocation
        cumulative_allocation = 0.0
        for variant in experiment.variants:
            cumulative_allocation += variant.allocation_percentage / 100.0
            if assignment_value < cumulative_allocation:
                self.assignment_cache[cache_key] = variant.variant_id
                return variant.variant_id
        
        # Fallback to control
        control_variant = next(v for v in experiment.variants if v.is_control)
        return control_variant.variant_id
    
    async def track_experiment_event(
        self,
        experiment_id: str,
        variant_id: str,
        user_id: str,
        event_type: str,
        value: Optional[float] = None
    ):
        """Track an event for experiment analysis"""
        if experiment_id not in self.active_experiments:
            return
        
        # In production, this would write to a data warehouse
        event_data = {
            "experiment_id": experiment_id,
            "variant_id": variant_id,
            "user_id": user_id,
            "event_type": event_type,
            "value": value,
            "timestamp": datetime.now()
        }
        
        # Log event for analysis
        logger.info(f"Experiment event: {event_data}")
    
    async def start_experiment(self, experiment_id: str):
        """Start an experiment"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_date = datetime.now()
        
        logger.info(f"Started experiment {experiment_id}")
    
    async def pause_experiment(self, experiment_id: str):
        """Pause a running experiment"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        experiment.status = ExperimentStatus.PAUSED
        
        logger.info(f"Paused experiment {experiment_id}")
    
    async def analyze_experiment_results(self, experiment_id: str) -> List[ExperimentResult]:
        """Analyze results of an experiment"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        
        # In production, this would query actual experiment data
        # For now, generate sample results
        results = []
        
        # Get control variant
        control_variant = next(v for v in experiment.variants if v.is_control)
        control_conversion = 0.10  # 10% baseline
        control_value = 50.0  # $50 baseline
        
        for variant in experiment.variants:
            # Generate sample data
            sample_size = 1000
            
            if variant.is_control:
                conversion_rate = control_conversion
                average_value = control_value
            else:
                # Simulate some lift for test variants
                lift = np.random.uniform(-0.05, 0.15)  # -5% to +15% lift
                conversion_rate = control_conversion * (1 + lift)
                average_value = control_value * (1 + lift * 0.5)
            
            # Calculate confidence interval
            std_error = np.sqrt(conversion_rate * (1 - conversion_rate) / sample_size)
            confidence_interval = (
                conversion_rate - 1.96 * std_error,
                conversion_rate + 1.96 * std_error
            )
            
            # Calculate p-value (simplified)
            if variant.is_control:
                p_value = 1.0
                is_significant = False
                lift_percentage = 0.0
            else:
                # Two-proportion z-test
                pooled_rate = (control_conversion + conversion_rate) / 2
                se = np.sqrt(2 * pooled_rate * (1 - pooled_rate) / sample_size)
                z_score = (conversion_rate - control_conversion) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                is_significant = p_value < 0.05
                lift_percentage = ((conversion_rate - control_conversion) / control_conversion) * 100
            
            results.append(
                ExperimentResult(
                    variant_id=variant.variant_id,
                    sample_size=sample_size,
                    conversion_rate=conversion_rate,
                    average_value=average_value,
                    confidence_interval=confidence_interval,
                    p_value=p_value,
                    is_significant=is_significant,
                    lift_percentage=lift_percentage
                )
            )
        
        experiment.results = results
        return results
    
    async def get_experiment_recommendation(self, experiment_id: str) -> Dict[str, Any]:
        """Get recommendation based on experiment results"""
        results = await self.analyze_experiment_results(experiment_id)
        experiment = self.active_experiments[experiment_id]
        
        # Find winning variant
        control_result = next(r for r in results if r.variant_id.startswith("control"))
        significant_winners = [
            r for r in results 
            if r.is_significant and r.lift_percentage > 0 and not r.variant_id.startswith("control")
        ]
        
        if not significant_winners:
            recommendation = {
                "action": "keep_control",
                "reason": "No variant showed statistically significant improvement",
                "confidence": "high"
            }
        else:
            # Find best performer
            best_variant = max(significant_winners, key=lambda x: x.lift_percentage)
            best_variant_config = next(v for v in experiment.variants if v.variant_id == best_variant.variant_id)
            
            recommendation = {
                "action": "implement_variant",
                "variant_id": best_variant.variant_id,
                "expected_lift": f"{best_variant.lift_percentage:.1f}%",
                "confidence": "high" if best_variant.p_value < 0.01 else "medium",
                "implementation_notes": f"Implement {best_variant_config.name}: {best_variant_config.parameters}"
            }
        
        # Check for negative impacts
        negative_impacts = [
            r for r in results 
            if r.is_significant and r.lift_percentage < -5 and not r.variant_id.startswith("control")
        ]
        
        if negative_impacts:
            warnings = [
                f"Variant {r.variant_id} showed {r.lift_percentage:.1f}% decrease"
                for r in negative_impacts
            ]
            recommendation["warnings"] = warnings
        
        return {
            "experiment_id": experiment_id,
            "experiment_name": experiment.name,
            "recommendation": recommendation,
            "detailed_results": [
                {
                    "variant_id": r.variant_id,
                    "conversion_rate": f"{r.conversion_rate:.2%}",
                    "lift": f"{r.lift_percentage:.1f}%",
                    "p_value": r.p_value,
                    "sample_size": r.sample_size
                }
                for r in results
            ],
            "analysis_timestamp": datetime.now()
        }
    
    async def calculate_experiment_duration(
        self,
        baseline_rate: float,
        minimum_effect: float,
        daily_traffic: int
    ) -> Dict[str, Any]:
        """Calculate how long an experiment needs to run"""
        required_sample_size = self._calculate_sample_size(
            baseline_rate=baseline_rate,
            minimum_effect=minimum_effect
        )
        
        # Account for multiple variants (assume 50/50 split for simplicity)
        sample_per_variant = required_sample_size
        total_sample_needed = sample_per_variant * 2
        
        # Calculate duration
        days_needed = np.ceil(total_sample_needed / daily_traffic)
        
        return {
            "required_sample_size_per_variant": sample_per_variant,
            "total_sample_size": total_sample_needed,
            "estimated_days": int(days_needed),
            "daily_traffic": daily_traffic,
            "confidence_level": "95%",
            "statistical_power": "80%"
        }
    
    async def get_active_experiments(self) -> List[Dict[str, Any]]:
        """Get all active experiments"""
        active = []
        
        for exp_id, experiment in self.active_experiments.items():
            if experiment.status == ExperimentStatus.RUNNING:
                active.append({
                    "experiment_id": exp_id,
                    "name": experiment.name,
                    "type": experiment.experiment_type.value,
                    "variants": len(experiment.variants),
                    "start_date": experiment.start_date,
                    "estimated_end_date": experiment.end_date,
                    "primary_metric": experiment.metrics.primary_metric
                })
        
        return active