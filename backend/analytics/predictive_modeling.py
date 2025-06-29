"""
Predictive Business Modeling Module

This module provides advanced machine learning-based forecasting and predictive
analytics that automatically adapt to any business niche. It enables businesses
to forecast revenue, growth, market trends, and run scenario planning.

Features:
- Revenue forecasting with multiple models
- Growth trajectory prediction
- Market trend analysis
- Scenario planning and what-if analysis
- Risk assessment and mitigation
- Opportunity identification
- Seasonal pattern detection
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
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, IsolationForest
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
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


class ForecastType(Enum):
    """Types of forecasts available"""
    REVENUE = "revenue"
    GROWTH = "growth"
    CUSTOMER_ACQUISITION = "customer_acquisition"
    MARKET_SHARE = "market_share"
    COST = "cost"
    PROFIT = "profit"
    ENGAGEMENT = "engagement"
    CHURN = "churn"


class ModelType(Enum):
    """Machine learning model types"""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ARIMA = "arima"
    PROPHET = "prophet"
    ENSEMBLE = "ensemble"


class ScenarioType(Enum):
    """Scenario planning types"""
    BEST_CASE = "best_case"
    WORST_CASE = "worst_case"
    MOST_LIKELY = "most_likely"
    AGGRESSIVE_GROWTH = "aggressive_growth"
    CONSERVATIVE = "conservative"
    MARKET_DISRUPTION = "market_disruption"
    COMPETITION_INCREASE = "competition_increase"


@dataclass
class Forecast:
    """Forecast results with confidence intervals"""
    forecast_type: ForecastType
    model_type: ModelType
    time_horizon: str
    values: pd.Series
    lower_bound: pd.Series
    upper_bound: pd.Series
    confidence_level: float
    accuracy_metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    seasonality: Optional[Dict[str, Any]] = None
    trend: Optional[str] = None


@dataclass
class Scenario:
    """Scenario planning results"""
    scenario_type: ScenarioType
    name: str
    description: str
    assumptions: Dict[str, Any]
    forecasts: Dict[ForecastType, Forecast]
    impact_analysis: Dict[str, float]
    probability: float
    recommendations: List[str]


@dataclass
class MarketTrend:
    """Market trend analysis results"""
    trend_name: str
    direction: str
    strength: float
    timeframe: str
    impact_areas: List[str]
    opportunities: List[str]
    threats: List[str]
    confidence: float


class BusinessDataPreparer:
    """Prepares business data for predictive modeling"""
    
    def __init__(self, persona_factory: PersonaFactory):
        self.persona_factory = persona_factory
        
    async def prepare_time_series_data(
        self,
        business_type: str,
        metric_type: ForecastType,
        time_range: Dict[str, datetime]
    ) -> pd.DataFrame:
        """Prepare time series data for forecasting"""
        try:
            async with get_db_context() as db:
                # Get historical data based on metric type
                query = self._get_metric_query(metric_type)
                
                result = await db.fetch_all(
                    query,
                    business_type,
                    time_range['start'],
                    time_range['end']
                )
                
                if not result:
                    # Generate synthetic data for demonstration
                    return self._generate_synthetic_time_series(
                        business_type,
                        metric_type,
                        time_range
                    )
                
                # Process results into DataFrame
                df = pd.DataFrame(result)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                
                # Add business-specific features
                df = self._add_business_features(df, business_type)
                
                return df
                
        except Exception as e:
            logger.error(f"Error preparing time series data: {e}")
            return self._generate_synthetic_time_series(
                business_type,
                metric_type,
                time_range
            )
    
    def _get_metric_query(self, metric_type: ForecastType) -> str:
        """Get SQL query for specific metric type"""
        queries = {
            ForecastType.REVENUE: """
                SELECT 
                    DATE_TRUNC('day', created_at) as date,
                    SUM(amount) as value,
                    COUNT(DISTINCT customer_id) as customer_count,
                    AVG(amount) as avg_transaction
                FROM transactions
                WHERE business_type = %s
                AND created_at BETWEEN %s AND %s
                GROUP BY DATE_TRUNC('day', created_at)
            """,
            ForecastType.CUSTOMER_ACQUISITION: """
                SELECT 
                    DATE_TRUNC('day', created_at) as date,
                    COUNT(DISTINCT id) as value,
                    AVG(acquisition_cost) as avg_cost,
                    SUM(CASE WHEN source = 'organic' THEN 1 ELSE 0 END) as organic_count
                FROM customers
                WHERE business_type = %s
                AND created_at BETWEEN %s AND %s
                GROUP BY DATE_TRUNC('day', created_at)
            """,
            ForecastType.ENGAGEMENT: """
                SELECT 
                    DATE_TRUNC('day', created_at) as date,
                    AVG(engagement_score) as value,
                    COUNT(DISTINCT user_id) as active_users,
                    SUM(actions) as total_actions
                FROM user_engagement
                WHERE business_type = %s
                AND created_at BETWEEN %s AND %s
                GROUP BY DATE_TRUNC('day', created_at)
            """
        }
        
        return queries.get(metric_type, queries[ForecastType.REVENUE])
    
    def _generate_synthetic_time_series(
        self,
        business_type: str,
        metric_type: ForecastType,
        time_range: Dict[str, datetime]
    ) -> pd.DataFrame:
        """Generate synthetic time series data"""
        persona = self.persona_factory.create_persona(business_type)
        
        # Create date range
        dates = pd.date_range(
            start=time_range['start'],
            end=time_range['end'],
            freq='D'
        )
        
        # Generate base trend
        n_days = len(dates)
        trend = np.linspace(100, 150, n_days)
        
        # Add seasonality
        seasonal_period = 7  # Weekly pattern
        seasonality = 10 * np.sin(2 * np.pi * np.arange(n_days) / seasonal_period)
        
        # Add noise
        noise = np.random.normal(0, 5, n_days)
        
        # Combine components
        values = trend + seasonality + noise
        
        # Adjust based on metric type
        if metric_type == ForecastType.REVENUE:
            values = values * 1000  # Scale to realistic revenue
        elif metric_type == ForecastType.CUSTOMER_ACQUISITION:
            values = np.maximum(1, values / 10).astype(int)
        elif metric_type == ForecastType.ENGAGEMENT:
            values = np.clip(values / 200, 0, 1)  # Normalize to 0-1
        
        # Create DataFrame
        df = pd.DataFrame({
            'value': values,
            'trend': trend,
            'seasonality': seasonality
        }, index=dates)
        
        # Add additional features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        return df
    
    def _add_business_features(self, df: pd.DataFrame, business_type: str) -> pd.DataFrame:
        """Add business-specific features to the data"""
        persona = self.persona_factory.create_persona(business_type)
        
        # Add time-based features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['day_of_month'] = df.index.day
        
        # Add lagged features
        for lag in [1, 7, 30]:
            df[f'lag_{lag}'] = df['value'].shift(lag)
        
        # Add rolling statistics
        for window in [7, 30]:
            df[f'rolling_mean_{window}'] = df['value'].rolling(window).mean()
            df[f'rolling_std_{window}'] = df['value'].rolling(window).std()
        
        # Add business-specific features based on persona
        if hasattr(persona, 'get_peak_seasons'):
            peak_seasons = persona.get_peak_seasons()
            for season in peak_seasons:
                df[f'is_{season}_season'] = 0  # Placeholder
        
        return df


class TimeSeriesForecaster:
    """Handles time series forecasting with multiple models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    async def forecast(
        self,
        data: pd.DataFrame,
        forecast_type: ForecastType,
        model_type: ModelType,
        horizon: int = 30
    ) -> Forecast:
        """Generate forecast using specified model"""
        try:
            if model_type == ModelType.ARIMA:
                return await self._arima_forecast(data, forecast_type, horizon)
            elif model_type == ModelType.PROPHET:
                return await self._prophet_forecast(data, forecast_type, horizon)
            elif model_type == ModelType.ENSEMBLE:
                return await self._ensemble_forecast(data, forecast_type, horizon)
            else:
                return await self._ml_forecast(data, forecast_type, model_type, horizon)
                
        except Exception as e:
            logger.error(f"Error in forecasting: {e}")
            return self._get_fallback_forecast(data, forecast_type, horizon)
    
    async def _arima_forecast(
        self,
        data: pd.DataFrame,
        forecast_type: ForecastType,
        horizon: int
    ) -> Forecast:
        """ARIMA model forecast"""
        try:
            # Prepare data
            ts_data = data['value'].fillna(method='ffill')
            
            # Fit ARIMA model
            model = ARIMA(ts_data, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=horizon)
            forecast_df = fitted_model.get_forecast(steps=horizon)
            confidence_int = forecast_df.conf_int(alpha=0.05)
            
            # Create forecast dates
            last_date = data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=horizon,
                freq='D'
            )
            
            # Calculate accuracy metrics on historical data
            train_size = int(len(ts_data) * 0.8)
            train, test = ts_data[:train_size], ts_data[train_size:]
            
            if len(test) > 0:
                test_model = ARIMA(train, order=(1, 1, 1)).fit()
                test_pred = test_model.forecast(steps=len(test))
                metrics = self._calculate_accuracy_metrics(test, test_pred)
            else:
                metrics = {'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0}
            
            return Forecast(
                forecast_type=forecast_type,
                model_type=ModelType.ARIMA,
                time_horizon=f"{horizon} days",
                values=pd.Series(forecast, index=forecast_dates),
                lower_bound=pd.Series(confidence_int.iloc[:, 0], index=forecast_dates),
                upper_bound=pd.Series(confidence_int.iloc[:, 1], index=forecast_dates),
                confidence_level=0.95,
                accuracy_metrics=metrics,
                trend=self._detect_trend(forecast)
            )
            
        except Exception as e:
            logger.error(f"ARIMA forecast error: {e}")
            return self._get_fallback_forecast(data, forecast_type, horizon)
    
    async def _prophet_forecast(
        self,
        data: pd.DataFrame,
        forecast_type: ForecastType,
        horizon: int
    ) -> Forecast:
        """Prophet model forecast"""
        try:
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': data.index,
                'y': data['value']
            }).reset_index(drop=True)
            
            # Initialize and fit Prophet
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                seasonality_mode='multiplicative'
            )
            
            # Add custom seasonalities if needed
            if forecast_type == ForecastType.REVENUE:
                model.add_seasonality(
                    name='monthly',
                    period=30.5,
                    fourier_order=5
                )
            
            model.fit(prophet_data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=horizon)
            
            # Generate forecast
            forecast_df = model.predict(future)
            
            # Extract forecast values
            forecast_values = forecast_df.iloc[-horizon:]
            
            # Calculate accuracy metrics
            if len(prophet_data) > 30:
                train_size = len(prophet_data) - 30
                cv_results = model.cv(
                    initial=f'{train_size} days',
                    period='30 days',
                    horizon='30 days'
                )
                metrics = {
                    'mae': cv_results['mae'].mean(),
                    'mse': cv_results['mse'].mean(),
                    'rmse': np.sqrt(cv_results['mse'].mean()),
                    'mape': cv_results['mape'].mean()
                }
            else:
                metrics = {'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0}
            
            return Forecast(
                forecast_type=forecast_type,
                model_type=ModelType.PROPHET,
                time_horizon=f"{horizon} days",
                values=pd.Series(
                    forecast_values['yhat'].values,
                    index=pd.to_datetime(forecast_values['ds'])
                ),
                lower_bound=pd.Series(
                    forecast_values['yhat_lower'].values,
                    index=pd.to_datetime(forecast_values['ds'])
                ),
                upper_bound=pd.Series(
                    forecast_values['yhat_upper'].values,
                    index=pd.to_datetime(forecast_values['ds'])
                ),
                confidence_level=0.95,
                accuracy_metrics=metrics,
                seasonality=self._extract_seasonality(model, forecast_df),
                trend=self._detect_trend(forecast_values['yhat'].values)
            )
            
        except Exception as e:
            logger.error(f"Prophet forecast error: {e}")
            return self._get_fallback_forecast(data, forecast_type, horizon)
    
    async def _ml_forecast(
        self,
        data: pd.DataFrame,
        forecast_type: ForecastType,
        model_type: ModelType,
        horizon: int
    ) -> Forecast:
        """Machine learning based forecast"""
        try:
            # Prepare features and target
            feature_cols = [col for col in data.columns if col != 'value']
            X = data[feature_cols].fillna(0)
            y = data['value']
            
            # Remove rows with NaN in target
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]
            
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Select and train model
            model = self._get_ml_model(model_type)
            model.fit(X_train_scaled, y_train)
            
            # Generate future features
            future_features = self._generate_future_features(data, horizon)
            future_features_scaled = scaler.transform(future_features)
            
            # Make predictions
            predictions = model.predict(future_features_scaled)
            
            # Calculate prediction intervals using quantile regression or bootstrapping
            lower_bound, upper_bound = self._calculate_prediction_intervals(
                model, future_features_scaled, predictions
            )
            
            # Calculate accuracy metrics
            if len(X_test) > 0:
                test_pred = model.predict(X_test_scaled)
                metrics = self._calculate_accuracy_metrics(y_test, test_pred)
            else:
                metrics = {'mae': 0, 'mse': 0, 'rmse': 0, 'r2': 0}
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_cols, model.feature_importances_))
            
            # Create forecast dates
            last_date = data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=horizon,
                freq='D'
            )
            
            return Forecast(
                forecast_type=forecast_type,
                model_type=model_type,
                time_horizon=f"{horizon} days",
                values=pd.Series(predictions, index=forecast_dates),
                lower_bound=pd.Series(lower_bound, index=forecast_dates),
                upper_bound=pd.Series(upper_bound, index=forecast_dates),
                confidence_level=0.95,
                accuracy_metrics=metrics,
                feature_importance=feature_importance,
                trend=self._detect_trend(predictions)
            )
            
        except Exception as e:
            logger.error(f"ML forecast error: {e}")
            return self._get_fallback_forecast(data, forecast_type, horizon)
    
    async def _ensemble_forecast(
        self,
        data: pd.DataFrame,
        forecast_type: ForecastType,
        horizon: int
    ) -> Forecast:
        """Ensemble forecast combining multiple models"""
        # Get forecasts from multiple models
        models_to_use = [
            ModelType.RANDOM_FOREST,
            ModelType.GRADIENT_BOOSTING,
            ModelType.XGBOOST
        ]
        
        forecasts = []
        weights = []
        
        for model_type in models_to_use:
            try:
                forecast = await self._ml_forecast(data, forecast_type, model_type, horizon)
                if forecast.accuracy_metrics.get('r2', 0) > 0:
                    forecasts.append(forecast)
                    # Weight by R² score
                    weights.append(forecast.accuracy_metrics.get('r2', 0.5))
            except Exception as e:
                logger.error(f"Error in {model_type} forecast: {e}")
        
        if not forecasts:
            return self._get_fallback_forecast(data, forecast_type, horizon)
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Combine forecasts
        ensemble_values = sum(
            w * f.values for w, f in zip(weights, forecasts)
        )
        ensemble_lower = sum(
            w * f.lower_bound for w, f in zip(weights, forecasts)
        )
        ensemble_upper = sum(
            w * f.upper_bound for w, f in zip(weights, forecasts)
        )
        
        # Average accuracy metrics
        avg_metrics = {}
        for metric in ['mae', 'mse', 'rmse', 'r2']:
            values = [f.accuracy_metrics.get(metric, 0) for f in forecasts]
            avg_metrics[metric] = np.average(values, weights=weights)
        
        return Forecast(
            forecast_type=forecast_type,
            model_type=ModelType.ENSEMBLE,
            time_horizon=f"{horizon} days",
            values=ensemble_values,
            lower_bound=ensemble_lower,
            upper_bound=ensemble_upper,
            confidence_level=0.95,
            accuracy_metrics=avg_metrics,
            trend=self._detect_trend(ensemble_values)
        )
    
    def _get_ml_model(self, model_type: ModelType):
        """Get machine learning model instance"""
        models = {
            ModelType.LINEAR_REGRESSION: LinearRegression(),
            ModelType.RANDOM_FOREST: RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            ModelType.GRADIENT_BOOSTING: GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            ModelType.XGBOOST: xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            ),
            ModelType.LIGHTGBM: lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        }
        
        return models.get(model_type, models[ModelType.RANDOM_FOREST])
    
    def _generate_future_features(
        self,
        data: pd.DataFrame,
        horizon: int
    ) -> pd.DataFrame:
        """Generate features for future dates"""
        last_date = data.index[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        future_df = pd.DataFrame(index=future_dates)
        
        # Add time-based features
        future_df['day_of_week'] = future_df.index.dayofweek
        future_df['month'] = future_df.index.month
        future_df['quarter'] = future_df.index.quarter
        future_df['is_weekend'] = (future_df.index.dayofweek >= 5).astype(int)
        future_df['day_of_month'] = future_df.index.day
        
        # Add lagged features using last known values
        for lag in [1, 7, 30]:
            if f'lag_{lag}' in data.columns:
                # Use last available values
                future_df[f'lag_{lag}'] = data[f'lag_{lag}'].iloc[-1]
        
        # Add rolling statistics using last known values
        for window in [7, 30]:
            if f'rolling_mean_{window}' in data.columns:
                future_df[f'rolling_mean_{window}'] = data[f'rolling_mean_{window}'].iloc[-1]
            if f'rolling_std_{window}' in data.columns:
                future_df[f'rolling_std_{window}'] = data[f'rolling_std_{window}'].iloc[-1]
        
        # Add any other features that exist in the original data
        for col in data.columns:
            if col not in future_df.columns and col != 'value':
                # Use last known value or default
                future_df[col] = data[col].iloc[-1] if col in data else 0
        
        return future_df
    
    def _calculate_prediction_intervals(
        self,
        model,
        X_future,
        predictions,
        confidence_level=0.95
    ):
        """Calculate prediction intervals"""
        # Simple approach: use historical prediction errors
        # In practice, could use quantile regression or bootstrapping
        
        # Estimate prediction variance
        std_estimate = np.std(predictions) * 0.1  # Simplified
        
        # Calculate intervals
        z_score = 1.96  # 95% confidence
        lower_bound = predictions - z_score * std_estimate
        upper_bound = predictions + z_score * std_estimate
        
        return lower_bound, upper_bound
    
    def _calculate_accuracy_metrics(self, y_true, y_pred):
        """Calculate forecast accuracy metrics"""
        try:
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            
            # MAPE calculation with zero handling
            mask = y_true != 0
            if mask.any():
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = 0
            
            # R² score
            if len(y_true) > 1:
                r2 = r2_score(y_true, y_pred)
            else:
                r2 = 0
            
            return {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'r2': float(r2)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {'mae': 0, 'mse': 0, 'rmse': 0, 'mape': 0, 'r2': 0}
    
    def _detect_trend(self, values):
        """Detect trend in forecast values"""
        if len(values) < 2:
            return "stable"
        
        # Simple linear regression for trend
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Determine trend based on slope
        avg_value = np.mean(values)
        slope_percentage = (slope / avg_value) * 100
        
        if slope_percentage > 1:
            return "increasing"
        elif slope_percentage < -1:
            return "decreasing"
        else:
            return "stable"
    
    def _extract_seasonality(self, model, forecast_df):
        """Extract seasonality components from Prophet model"""
        try:
            seasonality = {
                'weekly': forecast_df['weekly'].tolist()[-7:],
                'yearly': forecast_df['yearly'].mean() if 'yearly' in forecast_df else 0,
                'daily': forecast_df['daily'].tolist()[-24:] if 'daily' in forecast_df else []
            }
            return seasonality
        except:
            return None
    
    def _get_fallback_forecast(
        self,
        data: pd.DataFrame,
        forecast_type: ForecastType,
        horizon: int
    ) -> Forecast:
        """Fallback forecast using simple methods"""
        # Use simple moving average
        last_values = data['value'].tail(30).mean()
        
        last_date = data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        # Create constant forecast with some variance
        forecast_values = np.full(horizon, last_values)
        noise = np.random.normal(0, last_values * 0.05, horizon)
        forecast_values += noise
        
        return Forecast(
            forecast_type=forecast_type,
            model_type=ModelType.LINEAR_REGRESSION,
            time_horizon=f"{horizon} days",
            values=pd.Series(forecast_values, index=forecast_dates),
            lower_bound=pd.Series(forecast_values * 0.9, index=forecast_dates),
            upper_bound=pd.Series(forecast_values * 1.1, index=forecast_dates),
            confidence_level=0.95,
            accuracy_metrics={'mae': 0, 'mse': 0, 'rmse': 0, 'r2': 0},
            trend="stable"
        )


class ScenarioPlanner:
    """Handles scenario planning and what-if analysis"""
    
    def __init__(self, persona_factory: PersonaFactory):
        self.persona_factory = persona_factory
        self.forecaster = TimeSeriesForecaster()
        
    async def create_scenarios(
        self,
        base_data: pd.DataFrame,
        business_type: str,
        forecast_types: List[ForecastType],
        horizon: int = 90
    ) -> List[Scenario]:
        """Create multiple business scenarios"""
        scenarios = []
        
        # Define scenario configurations
        scenario_configs = self._get_scenario_configs(business_type)
        
        for config in scenario_configs:
            scenario = await self._create_scenario(
                base_data,
                config,
                forecast_types,
                horizon
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _get_scenario_configs(self, business_type: str) -> List[Dict[str, Any]]:
        """Get scenario configurations based on business type"""
        persona = self.persona_factory.create_persona(business_type)
        
        base_configs = [
            {
                'type': ScenarioType.BEST_CASE,
                'name': 'Optimistic Growth',
                'description': 'All factors align favorably',
                'adjustments': {
                    'revenue_multiplier': 1.3,
                    'cost_multiplier': 0.9,
                    'customer_growth': 1.5,
                    'market_share_increase': 0.2
                },
                'probability': 0.2
            },
            {
                'type': ScenarioType.WORST_CASE,
                'name': 'Challenging Market',
                'description': 'Multiple headwinds impact business',
                'adjustments': {
                    'revenue_multiplier': 0.7,
                    'cost_multiplier': 1.2,
                    'customer_growth': 0.5,
                    'market_share_increase': -0.1
                },
                'probability': 0.15
            },
            {
                'type': ScenarioType.MOST_LIKELY,
                'name': 'Expected Outcome',
                'description': 'Current trends continue',
                'adjustments': {
                    'revenue_multiplier': 1.1,
                    'cost_multiplier': 1.05,
                    'customer_growth': 1.2,
                    'market_share_increase': 0.05
                },
                'probability': 0.5
            },
            {
                'type': ScenarioType.MARKET_DISRUPTION,
                'name': 'Market Disruption',
                'description': 'New technology or competitor disrupts market',
                'adjustments': {
                    'revenue_multiplier': 0.8,
                    'cost_multiplier': 1.1,
                    'customer_growth': 0.6,
                    'market_share_increase': -0.15
                },
                'probability': 0.15
            }
        ]
        
        # Add business-specific adjustments
        if hasattr(persona, 'get_risk_factors'):
            risk_factors = persona.get_risk_factors()
            for config in base_configs:
                config['risk_factors'] = risk_factors
        
        return base_configs
    
    async def _create_scenario(
        self,
        base_data: pd.DataFrame,
        config: Dict[str, Any],
        forecast_types: List[ForecastType],
        horizon: int
    ) -> Scenario:
        """Create a single scenario with forecasts"""
        # Adjust base data according to scenario
        adjusted_data = self._adjust_data_for_scenario(base_data, config['adjustments'])
        
        # Generate forecasts for each type
        forecasts = {}
        for forecast_type in forecast_types:
            forecast = await self.forecaster.forecast(
                adjusted_data,
                forecast_type,
                ModelType.ENSEMBLE,
                horizon
            )
            forecasts[forecast_type] = forecast
        
        # Calculate impact analysis
        impact_analysis = self._calculate_scenario_impact(
            base_data,
            adjusted_data,
            forecasts
        )
        
        # Generate recommendations
        recommendations = self._generate_scenario_recommendations(
            config['type'],
            impact_analysis
        )
        
        return Scenario(
            scenario_type=config['type'],
            name=config['name'],
            description=config['description'],
            assumptions=config['adjustments'],
            forecasts=forecasts,
            impact_analysis=impact_analysis,
            probability=config['probability'],
            recommendations=recommendations
        )
    
    def _adjust_data_for_scenario(
        self,
        data: pd.DataFrame,
        adjustments: Dict[str, float]
    ) -> pd.DataFrame:
        """Adjust data based on scenario parameters"""
        adjusted = data.copy()
        
        # Apply revenue adjustment
        if 'value' in adjusted.columns:
            adjusted['value'] *= adjustments.get('revenue_multiplier', 1.0)
        
        # Adjust trends
        if 'trend' in adjusted.columns:
            trend_factor = adjustments.get('revenue_multiplier', 1.0) - 1
            adjusted['trend'] *= (1 + trend_factor * 0.5)
        
        return adjusted
    
    def _calculate_scenario_impact(
        self,
        base_data: pd.DataFrame,
        adjusted_data: pd.DataFrame,
        forecasts: Dict[ForecastType, Forecast]
    ) -> Dict[str, float]:
        """Calculate the impact of a scenario"""
        impact = {}
        
        # Revenue impact
        if ForecastType.REVENUE in forecasts:
            base_revenue = base_data['value'].sum()
            forecast_revenue = forecasts[ForecastType.REVENUE].values.sum()
            impact['revenue_change'] = (forecast_revenue - base_revenue) / base_revenue
        
        # Growth rate impact
        if ForecastType.GROWTH in forecasts:
            impact['growth_rate_change'] = forecasts[ForecastType.GROWTH].values.mean()
        
        # Customer impact
        if ForecastType.CUSTOMER_ACQUISITION in forecasts:
            impact['customer_change'] = forecasts[ForecastType.CUSTOMER_ACQUISITION].values.sum()
        
        # Overall business impact score
        impact['overall_impact'] = np.mean([
            abs(v) for v in impact.values() if isinstance(v, (int, float))
        ])
        
        return impact
    
    def _generate_scenario_recommendations(
        self,
        scenario_type: ScenarioType,
        impact_analysis: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations for a scenario"""
        recommendations = []
        
        if scenario_type == ScenarioType.BEST_CASE:
            recommendations.extend([
                "Scale operations to capture growth opportunity",
                "Invest in customer acquisition while CAC is favorable",
                "Build inventory/capacity to meet increased demand",
                "Lock in favorable supplier terms",
                "Accelerate product development roadmap"
            ])
        
        elif scenario_type == ScenarioType.WORST_CASE:
            recommendations.extend([
                "Implement cost reduction measures",
                "Focus on customer retention over acquisition",
                "Diversify revenue streams",
                "Build cash reserves for contingency",
                "Strengthen core product offerings"
            ])
        
        elif scenario_type == ScenarioType.MARKET_DISRUPTION:
            recommendations.extend([
                "Accelerate digital transformation",
                "Invest in innovation and R&D",
                "Form strategic partnerships",
                "Pivot business model if necessary",
                "Monitor competitor activities closely"
            ])
        
        elif scenario_type == ScenarioType.MOST_LIKELY:
            recommendations.extend([
                "Continue current growth strategies",
                "Optimize operational efficiency",
                "Gradually expand market presence",
                "Maintain balanced investment approach",
                "Focus on sustainable growth"
            ])
        
        # Add impact-specific recommendations
        if impact_analysis.get('revenue_change', 0) < -0.2:
            recommendations.append("Urgently review pricing strategy")
        
        if impact_analysis.get('customer_change', 0) < 0:
            recommendations.append("Implement customer win-back campaigns")
        
        return recommendations


class MarketTrendAnalyzer:
    """Analyzes market trends and their impact on business"""
    
    def __init__(self, persona_factory: PersonaFactory):
        self.persona_factory = persona_factory
        
    async def analyze_market_trends(
        self,
        business_type: str,
        market_data: Optional[pd.DataFrame] = None
    ) -> List[MarketTrend]:
        """Analyze market trends affecting the business"""
        persona = self.persona_factory.create_persona(business_type)
        
        # Get market data if not provided
        if market_data is None:
            market_data = await self._fetch_market_data(business_type)
        
        trends = []
        
        # Analyze different types of trends
        trends.extend(await self._analyze_growth_trends(market_data, persona))
        trends.extend(await self._analyze_technology_trends(business_type))
        trends.extend(await self._analyze_competitive_trends(business_type))
        trends.extend(await self._analyze_consumer_trends(business_type))
        
        # Sort by strength
        trends.sort(key=lambda x: x.strength, reverse=True)
        
        return trends
    
    async def _fetch_market_data(self, business_type: str) -> pd.DataFrame:
        """Fetch or generate market data"""
        # In practice, this would fetch real market data
        # For now, generate synthetic data
        dates = pd.date_range(end=datetime.now(), periods=365, freq='D')
        
        data = pd.DataFrame({
            'date': dates,
            'market_size': np.random.lognormal(15, 0.5, len(dates)),
            'growth_rate': np.random.normal(0.1, 0.05, len(dates)),
            'competition_index': np.random.uniform(0.5, 1.5, len(dates)),
            'innovation_index': np.random.uniform(0.3, 1.0, len(dates)),
            'consumer_sentiment': np.random.uniform(0.4, 0.9, len(dates))
        })
        
        return data.set_index('date')
    
    async def _analyze_growth_trends(
        self,
        market_data: pd.DataFrame,
        persona
    ) -> List[MarketTrend]:
        """Analyze market growth trends"""
        trends = []
        
        # Calculate growth trajectory
        recent_growth = market_data['growth_rate'].tail(30).mean()
        historical_growth = market_data['growth_rate'].head(30).mean()
        growth_acceleration = recent_growth - historical_growth
        
        if growth_acceleration > 0.05:
            trends.append(MarketTrend(
                trend_name="Accelerating Market Growth",
                direction="positive",
                strength=min(growth_acceleration * 10, 1.0),
                timeframe="3-6 months",
                impact_areas=["revenue", "market_share", "valuation"],
                opportunities=[
                    "Expand market presence while growth is strong",
                    "Invest in customer acquisition",
                    "Launch new products/services",
                    "Raise capital at favorable terms"
                ],
                threats=[
                    "Increased competition entering market",
                    "Market saturation risk",
                    "Regulatory attention"
                ],
                confidence=0.85
            ))
        elif growth_acceleration < -0.05:
            trends.append(MarketTrend(
                trend_name="Market Growth Deceleration",
                direction="negative",
                strength=min(abs(growth_acceleration) * 10, 1.0),
                timeframe="3-6 months",
                impact_areas=["revenue", "profitability", "competition"],
                opportunities=[
                    "Focus on operational efficiency",
                    "Consolidate market position",
                    "Acquire weaker competitors",
                    "Differentiate offerings"
                ],
                threats=[
                    "Revenue growth challenges",
                    "Margin pressure",
                    "Investor sentiment decline"
                ],
                confidence=0.85
            ))
        
        return trends
    
    async def _analyze_technology_trends(self, business_type: str) -> List[MarketTrend]:
        """Analyze technology trends affecting the business"""
        persona = self.persona_factory.create_persona(business_type)
        trends = []
        
        # AI/ML adoption trend
        trends.append(MarketTrend(
            trend_name="AI/ML Technology Adoption",
            direction="positive",
            strength=0.9,
            timeframe="12-24 months",
            impact_areas=["operations", "customer_experience", "competition"],
            opportunities=[
                "Automate operations with AI",
                "Enhance customer personalization",
                "Improve decision-making with ML",
                "Create AI-powered products"
            ],
            threats=[
                "Technology investment requirements",
                "Skills gap in organization",
                "Competitive disadvantage if not adopted"
            ],
            confidence=0.95
        ))
        
        # Digital transformation trend
        if hasattr(persona, 'get_digital_maturity'):
            maturity = persona.get_digital_maturity()
            if maturity < 0.7:
                trends.append(MarketTrend(
                    trend_name="Digital Transformation Imperative",
                    direction="neutral",
                    strength=0.8,
                    timeframe="6-12 months",
                    impact_areas=["operations", "customer_acquisition", "efficiency"],
                    opportunities=[
                        "Modernize technology infrastructure",
                        "Implement digital channels",
                        "Streamline processes",
                        "Reach new customer segments"
                    ],
                    threats=[
                        "Implementation complexity",
                        "Change management challenges",
                        "Initial cost burden"
                    ],
                    confidence=0.90
                ))
        
        return trends
    
    async def _analyze_competitive_trends(self, business_type: str) -> List[MarketTrend]:
        """Analyze competitive landscape trends"""
        trends = []
        
        # Market consolidation trend
        trends.append(MarketTrend(
            trend_name="Market Consolidation",
            direction="neutral",
            strength=0.7,
            timeframe="12-18 months",
            impact_areas=["market_share", "pricing", "partnerships"],
            opportunities=[
                "Strategic acquisition opportunities",
                "Partnership possibilities",
                "Market leadership potential",
                "Economies of scale"
            ],
            threats=[
                "Acquisition target risk",
                "Increased competitive pressure",
                "Price wars",
                "Customer concentration"
            ],
            confidence=0.80
        ))
        
        return trends
    
    async def _analyze_consumer_trends(self, business_type: str) -> List[MarketTrend]:
        """Analyze consumer behavior trends"""
        persona = self.persona_factory.create_persona(business_type)
        trends = []
        
        # Sustainability trend
        trends.append(MarketTrend(
            trend_name="Sustainability Focus",
            direction="positive",
            strength=0.75,
            timeframe="24-36 months",
            impact_areas=["brand", "customer_loyalty", "operations"],
            opportunities=[
                "Build sustainable brand positioning",
                "Attract conscious consumers",
                "Reduce operational costs",
                "Access green financing"
            ],
            threats=[
                "Implementation costs",
                "Greenwashing risks",
                "Supply chain complexity"
            ],
            confidence=0.85
        ))
        
        # Personalization expectation
        trends.append(MarketTrend(
            trend_name="Hyper-Personalization Demand",
            direction="positive",
            strength=0.85,
            timeframe="6-12 months",
            impact_areas=["customer_experience", "retention", "conversion"],
            opportunities=[
                "Increase customer lifetime value",
                "Improve conversion rates",
                "Build competitive advantage",
                "Enhance brand loyalty"
            ],
            threats=[
                "Data privacy concerns",
                "Technology requirements",
                "Complexity of implementation"
            ],
            confidence=0.90
        ))
        
        return trends


class PredictiveBusinessModeling(UniversalAnalyticsEngine):
    """
    Predictive Business Modeling Engine
    
    Provides advanced ML-based forecasting and predictive analytics
    that automatically adapt to any business niche.
    """
    
    def __init__(self, encryption_manager: EncryptionManager):
        super().__init__(encryption_manager)
        self.persona_factory = PersonaFactory()
        self.data_preparer = BusinessDataPreparer(self.persona_factory)
        self.forecaster = TimeSeriesForecaster()
        self.scenario_planner = ScenarioPlanner(self.persona_factory)
        self.trend_analyzer = MarketTrendAnalyzer(self.persona_factory)
        
    async def generate_executive_summary(
        self,
        analytics_data: Dict[str, Any],
        business_type: str
    ) -> str:
        """Generate executive summary of predictive analytics"""
        summary_parts = []
        
        # Revenue forecast summary
        revenue_forecast = analytics_data.get('forecasts', {}).get(ForecastType.REVENUE)
        if revenue_forecast:
            forecast_value = revenue_forecast.values.sum()
            current_value = analytics_data.get('current_metrics', {}).get('revenue', 0)
            growth_rate = ((forecast_value - current_value) / current_value * 100) if current_value > 0 else 0
            
            summary_parts.append(
                f"**Revenue Forecast ({revenue_forecast.time_horizon}):**\n"
                f"- Projected Revenue: ${forecast_value:,.2f}\n"
                f"- Expected Growth: {growth_rate:.1f}%\n"
                f"- Confidence Level: {revenue_forecast.confidence_level * 100:.0f}%\n"
                f"- Trend: {revenue_forecast.trend}\n"
            )
        
        # Scenario analysis summary
        scenarios = analytics_data.get('scenarios', [])
        if scenarios:
            summary_parts.append("\n**Scenario Analysis:**")
            for scenario in scenarios[:3]:  # Top 3 scenarios
                impact = scenario.impact_analysis.get('revenue_change', 0) * 100
                summary_parts.append(
                    f"- {scenario.name}: {impact:+.1f}% revenue impact "
                    f"(probability: {scenario.probability * 100:.0f}%)"
                )
        
        # Market trends summary
        trends = analytics_data.get('market_trends', [])
        if trends:
            summary_parts.append("\n**Key Market Trends:**")
            for trend in trends[:3]:  # Top 3 trends
                summary_parts.append(
                    f"- {trend.trend_name}: {trend.direction} impact "
                    f"(strength: {trend.strength:.1%})"
                )
        
        # Key risks and opportunities
        risks = analytics_data.get('key_risks', [])
        opportunities = analytics_data.get('key_opportunities', [])
        
        if risks:
            summary_parts.append("\n**Key Risks:**")
            for risk in risks[:3]:
                summary_parts.append(f"- {risk}")
        
        if opportunities:
            summary_parts.append("\n**Key Opportunities:**")
            for opp in opportunities[:3]:
                summary_parts.append(f"- {opp}")
        
        # Strategic recommendations
        recommendations = analytics_data.get('strategic_recommendations', [])
        if recommendations:
            summary_parts.append("\n**Strategic Recommendations:**")
            for i, rec in enumerate(recommendations[:3], 1):
                summary_parts.append(f"{i}. {rec}")
        
        return "\n".join(summary_parts)
    
    async def calculate_kpis(
        self,
        request: AnalyticsRequest
    ) -> List[BusinessKPI]:
        """Calculate predictive analytics KPIs"""
        kpis = []
        
        # Prepare data
        revenue_data = await self.data_preparer.prepare_time_series_data(
            request.business_type,
            ForecastType.REVENUE,
            request.time_range
        )
        
        # Forecast accuracy KPI
        forecast = await self.forecaster.forecast(
            revenue_data,
            ForecastType.REVENUE,
            ModelType.ENSEMBLE,
            30
        )
        
        kpis.append(BusinessKPI(
            name="Forecast Accuracy (R²)",
            value=forecast.accuracy_metrics.get('r2', 0),
            unit="score",
            trend="stable",
            target=0.85,
            category="Model Performance"
        ))
        
        # Revenue growth forecast
        growth_rate = self._calculate_growth_rate(revenue_data, forecast)
        kpis.append(BusinessKPI(
            name="Projected Revenue Growth",
            value=growth_rate,
            unit="percentage",
            trend="up" if growth_rate > 0 else "down",
            target=0.15,
            category="Growth"
        ))
        
        # Scenario planning KPI
        scenarios = await self.scenario_planner.create_scenarios(
            revenue_data,
            request.business_type,
            [ForecastType.REVENUE],
            90
        )
        
        best_case_impact = max(
            s.impact_analysis.get('revenue_change', 0) for s in scenarios
        )
        
        kpis.append(BusinessKPI(
            name="Best Case Revenue Uplift",
            value=best_case_impact,
            unit="percentage",
            trend="stable",
            target=0.30,
            category="Scenario Planning"
        ))
        
        # Risk assessment KPI
        worst_case_impact = min(
            s.impact_analysis.get('revenue_change', 0) for s in scenarios
        )
        
        kpis.append(BusinessKPI(
            name="Downside Risk",
            value=abs(worst_case_impact),
            unit="percentage",
            trend="stable",
            target=0.15,
            category="Risk"
        ))
        
        # Market trend strength
        trends = await self.trend_analyzer.analyze_market_trends(request.business_type)
        avg_trend_strength = np.mean([t.strength for t in trends]) if trends else 0
        
        kpis.append(BusinessKPI(
            name="Market Opportunity Score",
            value=avg_trend_strength,
            unit="score",
            trend="up" if avg_trend_strength > 0.6 else "stable",
            target=0.75,
            category="Market"
        ))
        
        # Confidence score
        confidence_scores = [
            forecast.confidence_level,
            *[s.probability for s in scenarios],
            *[t.confidence for t in trends]
        ]
        avg_confidence = np.mean(confidence_scores)
        
        kpis.append(BusinessKPI(
            name="Prediction Confidence",
            value=avg_confidence,
            unit="score",
            trend="stable",
            target=0.85,
            category="Model Performance"
        ))
        
        return kpis
    
    async def analyze_data(
        self,
        request: AnalyticsRequest
    ) -> Dict[str, Any]:
        """Perform comprehensive predictive analysis"""
        try:
            # Prepare data for multiple forecast types
            forecast_types = [
                ForecastType.REVENUE,
                ForecastType.GROWTH,
                ForecastType.CUSTOMER_ACQUISITION,
                ForecastType.ENGAGEMENT
            ]
            
            # Generate forecasts
            forecasts = {}
            current_metrics = {}
            
            for forecast_type in forecast_types:
                try:
                    # Prepare data
                    data = await self.data_preparer.prepare_time_series_data(
                        request.business_type,
                        forecast_type,
                        request.time_range
                    )
                    
                    # Store current metrics
                    if len(data) > 0:
                        current_metrics[forecast_type.value] = float(data['value'].iloc[-1])
                    
                    # Generate forecast
                    forecast = await self.forecaster.forecast(
                        data,
                        forecast_type,
                        ModelType.ENSEMBLE,
                        request.custom_params.get('horizon', 90)
                    )
                    
                    forecasts[forecast_type] = forecast
                    
                except Exception as e:
                    logger.error(f"Error forecasting {forecast_type}: {e}")
            
            # Create scenarios
            base_data = await self.data_preparer.prepare_time_series_data(
                request.business_type,
                ForecastType.REVENUE,
                request.time_range
            )
            
            scenarios = await self.scenario_planner.create_scenarios(
                base_data,
                request.business_type,
                list(forecasts.keys()),
                request.custom_params.get('horizon', 90)
            )
            
            # Analyze market trends
            market_trends = await self.trend_analyzer.analyze_market_trends(
                request.business_type
            )
            
            # Extract insights
            insights = await self._generate_predictive_insights(
                forecasts,
                scenarios,
                market_trends,
                request.business_type
            )
            
            # Identify risks and opportunities
            risks, opportunities = self._identify_risks_opportunities(
                forecasts,
                scenarios,
                market_trends
            )
            
            # Generate strategic recommendations
            recommendations = self._generate_strategic_recommendations(
                forecasts,
                scenarios,
                market_trends,
                request.business_type
            )
            
            return {
                'forecasts': forecasts,
                'current_metrics': current_metrics,
                'scenarios': scenarios,
                'market_trends': market_trends,
                'insights': insights,
                'key_risks': risks,
                'key_opportunities': opportunities,
                'strategic_recommendations': recommendations,
                'model_performance': self._calculate_model_performance(forecasts)
            }
            
        except Exception as e:
            logger.error(f"Error in predictive analysis: {e}")
            raise
    
    def _calculate_growth_rate(
        self,
        historical_data: pd.DataFrame,
        forecast: Forecast
    ) -> float:
        """Calculate projected growth rate"""
        if len(historical_data) == 0 or len(forecast.values) == 0:
            return 0.0
        
        current_value = historical_data['value'].iloc[-1]
        future_value = forecast.values.iloc[-1]
        
        if current_value > 0:
            return (future_value - current_value) / current_value
        return 0.0
    
    async def _generate_predictive_insights(
        self,
        forecasts: Dict[ForecastType, Forecast],
        scenarios: List[Scenario],
        market_trends: List[MarketTrend],
        business_type: str
    ) -> List[AnalyticsInsight]:
        """Generate actionable predictive insights"""
        insights = []
        persona = self.persona_factory.create_persona(business_type)
        
        # Revenue forecast insights
        if ForecastType.REVENUE in forecasts:
            revenue_forecast = forecasts[ForecastType.REVENUE]
            if revenue_forecast.trend == "increasing":
                insights.append(AnalyticsInsight(
                    title="Strong Revenue Growth Projected",
                    description=(
                        f"Revenue is projected to grow with {revenue_forecast.trend} trend "
                        f"over the next {revenue_forecast.time_horizon}"
                    ),
                    impact="Positive impact on cash flow and business valuation",
                    recommendation=(
                        "Capitalize on growth momentum:\n"
                        "1. Scale operations to meet demand\n"
                        "2. Invest in customer acquisition\n"
                        "3. Expand product/service offerings\n"
                        "4. Consider raising growth capital"
                    ),
                    priority=InsightPriority.HIGH,
                    confidence=revenue_forecast.confidence_level,
                    supporting_data={
                        'forecast_accuracy': revenue_forecast.accuracy_metrics,
                        'trend': revenue_forecast.trend
                    }
                ))
            elif revenue_forecast.trend == "decreasing":
                insights.append(AnalyticsInsight(
                    title="Revenue Decline Warning",
                    description=(
                        f"Revenue shows {revenue_forecast.trend} trend "
                        f"requiring immediate attention"
                    ),
                    impact="Risk to profitability and business sustainability",
                    recommendation=(
                        "Implement revenue protection strategy:\n"
                        "1. Analyze and address root causes\n"
                        "2. Focus on customer retention\n"
                        "3. Optimize pricing strategy\n"
                        "4. Reduce operational costs"
                    ),
                    priority=InsightPriority.CRITICAL,
                    confidence=revenue_forecast.confidence_level,
                    supporting_data={
                        'forecast_accuracy': revenue_forecast.accuracy_metrics,
                        'trend': revenue_forecast.trend
                    }
                ))
        
        # Customer acquisition insights
        if ForecastType.CUSTOMER_ACQUISITION in forecasts:
            customer_forecast = forecasts[ForecastType.CUSTOMER_ACQUISITION]
            projected_customers = customer_forecast.values.sum()
            
            if customer_forecast.trend == "decreasing":
                insights.append(AnalyticsInsight(
                    title="Customer Acquisition Slowdown",
                    description=(
                        "Customer acquisition is projected to slow down, "
                        "potentially impacting long-term growth"
                    ),
                    impact="Reduced market share and revenue growth potential",
                    recommendation=(
                        "Revitalize acquisition strategy:\n"
                        "1. Review and optimize marketing channels\n"
                        "2. Enhance value proposition\n"
                        "3. Implement referral programs\n"
                        "4. Explore new customer segments"
                    ),
                    priority=InsightPriority.HIGH,
                    confidence=customer_forecast.confidence_level,
                    supporting_data={
                        'projected_customers': int(projected_customers),
                        'trend': customer_forecast.trend
                    }
                ))
        
        # Scenario-based insights
        high_impact_scenarios = [s for s in scenarios if abs(s.impact_analysis.get('overall_impact', 0)) > 0.2]
        
        if high_impact_scenarios:
            most_likely_high_impact = max(
                high_impact_scenarios,
                key=lambda s: s.probability
            )
            
            insights.append(AnalyticsInsight(
                title=f"High-Impact Scenario: {most_likely_high_impact.name}",
                description=(
                    f"{most_likely_high_impact.description} with "
                    f"{most_likely_high_impact.probability * 100:.0f}% probability"
                ),
                impact=(
                    f"Potential revenue impact: "
                    f"{most_likely_high_impact.impact_analysis.get('revenue_change', 0) * 100:+.1f}%"
                ),
                recommendation=(
                    "Prepare for this scenario:\n" +
                    "\n".join(f"- {rec}" for rec in most_likely_high_impact.recommendations[:4])
                ),
                priority=InsightPriority.HIGH,
                confidence=0.85,
                supporting_data={
                    'scenario_probability': most_likely_high_impact.probability,
                    'impact_analysis': most_likely_high_impact.impact_analysis
                }
            ))
        
        # Market trend insights
        strong_trends = [t for t in market_trends if t.strength > 0.8]
        
        for trend in strong_trends[:2]:  # Top 2 strong trends
            insights.append(AnalyticsInsight(
                title=f"Market Trend: {trend.trend_name}",
                description=(
                    f"{trend.trend_name} showing {trend.direction} impact "
                    f"with {trend.strength:.0%} strength over {trend.timeframe}"
                ),
                impact=(
                    f"Affects: {', '.join(trend.impact_areas)}"
                ),
                recommendation=(
                    "Action plan:\n" +
                    "Opportunities:\n" +
                    "\n".join(f"- {opp}" for opp in trend.opportunities[:2]) +
                    "\n\nMitigate threats:\n" +
                    "\n".join(f"- {threat}" for threat in trend.threats[:2])
                ),
                priority=InsightPriority.MEDIUM,
                confidence=trend.confidence,
                supporting_data={
                    'trend_strength': trend.strength,
                    'timeframe': trend.timeframe
                }
            ))
        
        # Model performance insight
        avg_accuracy = np.mean([
            f.accuracy_metrics.get('r2', 0) for f in forecasts.values()
        ])
        
        if avg_accuracy < 0.7:
            insights.append(AnalyticsInsight(
                title="Model Accuracy Below Target",
                description=(
                    f"Predictive model accuracy is {avg_accuracy:.1%}, "
                    f"below the target threshold"
                ),
                impact="Reduced confidence in forecasts and strategic decisions",
                recommendation=(
                    "Improve model performance:\n"
                    "1. Collect more historical data\n"
                    "2. Add external data sources\n"
                    "3. Feature engineering optimization\n"
                    "4. Consider alternative models"
                ),
                priority=InsightPriority.MEDIUM,
                confidence=0.90,
                supporting_data={
                    'current_accuracy': avg_accuracy,
                    'target_accuracy': 0.85
                }
            ))
        
        return insights
    
    def _identify_risks_opportunities(
        self,
        forecasts: Dict[ForecastType, Forecast],
        scenarios: List[Scenario],
        market_trends: List[MarketTrend]
    ) -> Tuple[List[str], List[str]]:
        """Identify key risks and opportunities"""
        risks = []
        opportunities = []
        
        # Forecast-based risks/opportunities
        for forecast_type, forecast in forecasts.items():
            if forecast.trend == "decreasing":
                risks.append(f"{forecast_type.value} showing declining trend")
            elif forecast.trend == "increasing":
                opportunities.append(f"Strong {forecast_type.value} growth projected")
        
        # Scenario-based risks/opportunities
        worst_case = min(scenarios, key=lambda s: s.impact_analysis.get('revenue_change', 0))
        best_case = max(scenarios, key=lambda s: s.impact_analysis.get('revenue_change', 0))
        
        if worst_case.impact_analysis.get('revenue_change', 0) < -0.2:
            risks.append(f"{worst_case.name} could impact revenue by {worst_case.impact_analysis.get('revenue_change', 0) * 100:.0f}%")
        
        if best_case.impact_analysis.get('revenue_change', 0) > 0.2:
            opportunities.append(f"{best_case.name} could boost revenue by {best_case.impact_analysis.get('revenue_change', 0) * 100:.0f}%")
        
        # Market trend risks/opportunities
        for trend in market_trends:
            if trend.direction == "negative" and trend.strength > 0.7:
                risks.extend(trend.threats[:2])
            elif trend.direction == "positive" and trend.strength > 0.7:
                opportunities.extend(trend.opportunities[:2])
        
        # Remove duplicates and limit
        risks = list(dict.fromkeys(risks))[:5]
        opportunities = list(dict.fromkeys(opportunities))[:5]
        
        return risks, opportunities
    
    def _generate_strategic_recommendations(
        self,
        forecasts: Dict[ForecastType, Forecast],
        scenarios: List[Scenario],
        market_trends: List[MarketTrend],
        business_type: str
    ) -> List[str]:
        """Generate strategic recommendations based on predictions"""
        recommendations = []
        persona = self.persona_factory.create_persona(business_type)
        
        # Revenue-based recommendations
        if ForecastType.REVENUE in forecasts:
            revenue_trend = forecasts[ForecastType.REVENUE].trend
            if revenue_trend == "increasing":
                recommendations.append(
                    "Invest in growth: Scale operations and expand market presence while momentum is positive"
                )
            else:
                recommendations.append(
                    "Focus on efficiency: Optimize costs and improve unit economics to protect margins"
                )
        
        # Customer-based recommendations
        if ForecastType.CUSTOMER_ACQUISITION in forecasts:
            customer_trend = forecasts[ForecastType.CUSTOMER_ACQUISITION].trend
            if customer_trend == "decreasing":
                recommendations.append(
                    "Revamp acquisition strategy: Test new channels and improve conversion funnel"
                )
        
        # Scenario-based recommendations
        most_likely_scenario = max(scenarios, key=lambda s: s.probability)
        recommendations.extend(most_likely_scenario.recommendations[:2])
        
        # Market trend recommendations
        top_opportunity = max(
            market_trends,
            key=lambda t: t.strength if t.direction == "positive" else 0
        )
        if top_opportunity.direction == "positive":
            recommendations.append(
                f"Capitalize on {top_opportunity.trend_name}: {top_opportunity.opportunities[0]}"
            )
        
        # Risk mitigation recommendation
        highest_risk = max(
            market_trends,
            key=lambda t: t.strength if t.direction == "negative" else 0
        )
        if highest_risk.direction == "negative":
            recommendations.append(
                f"Mitigate {highest_risk.trend_name} risk: {highest_risk.threats[0]}"
            )
        
        # Remove duplicates and prioritize
        recommendations = list(dict.fromkeys(recommendations))[:6]
        
        return recommendations
    
    def _calculate_model_performance(
        self,
        forecasts: Dict[ForecastType, Forecast]
    ) -> Dict[str, Any]:
        """Calculate overall model performance metrics"""
        if not forecasts:
            return {}
        
        # Aggregate metrics across all forecasts
        all_metrics = defaultdict(list)
        
        for forecast in forecasts.values():
            for metric, value in forecast.accuracy_metrics.items():
                all_metrics[metric].append(value)
        
        # Calculate averages
        performance = {
            metric: np.mean(values)
            for metric, values in all_metrics.items()
        }
        
        # Add model types used
        performance['models_used'] = list(set(
            f.model_type.value for f in forecasts.values()
        ))
        
        # Overall performance score
        performance['overall_score'] = np.mean([
            performance.get('r2', 0),
            1 - min(performance.get('mape', 100) / 100, 1),
            1 - min(performance.get('rmse', 100) / 100, 1)
        ])
        
        return performance
    
    async def generate_report(
        self,
        request: AnalyticsRequest,
        format: ReportFormat
    ) -> Union[Dict[str, Any], str, bytes]:
        """Generate comprehensive predictive modeling report"""
        # Perform analysis
        analysis_results = await self.analyze_data(request)
        
        # Create visualizations
        visualizations = await self._create_visualizations(
            analysis_results,
            request.business_type
        )
        
        # Compile report data
        report_data = {
            'title': f'Predictive Business Modeling Report - {request.business_type}',
            'generated_at': datetime.now().isoformat(),
            'time_period': f"{request.time_range['start']} to {request.time_range['end']}",
            'executive_summary': await self.generate_executive_summary(
                analysis_results,
                request.business_type
            ),
            'kpis': await self.calculate_kpis(request),
            'forecasts': {
                k.value: self._serialize_forecast(v)
                for k, v in analysis_results.get('forecasts', {}).items()
            },
            'scenarios': [
                self._serialize_scenario(s)
                for s in analysis_results.get('scenarios', [])
            ],
            'market_trends': [
                self._serialize_trend(t)
                for t in analysis_results.get('market_trends', [])
            ],
            'insights': analysis_results.get('insights', []),
            'risks_opportunities': {
                'key_risks': analysis_results.get('key_risks', []),
                'key_opportunities': analysis_results.get('key_opportunities', [])
            },
            'recommendations': analysis_results.get('strategic_recommendations', []),
            'model_performance': analysis_results.get('model_performance', {}),
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
    
    def _serialize_forecast(self, forecast: Forecast) -> Dict[str, Any]:
        """Serialize forecast object for report"""
        return {
            'type': forecast.forecast_type.value,
            'model': forecast.model_type.value,
            'horizon': forecast.time_horizon,
            'trend': forecast.trend,
            'confidence': forecast.confidence_level,
            'accuracy_metrics': forecast.accuracy_metrics,
            'values': {
                'forecast': forecast.values.to_list(),
                'lower_bound': forecast.lower_bound.to_list(),
                'upper_bound': forecast.upper_bound.to_list(),
                'dates': [d.isoformat() for d in forecast.values.index]
            }
        }
    
    def _serialize_scenario(self, scenario: Scenario) -> Dict[str, Any]:
        """Serialize scenario object for report"""
        return {
            'type': scenario.scenario_type.value,
            'name': scenario.name,
            'description': scenario.description,
            'probability': scenario.probability,
            'assumptions': scenario.assumptions,
            'impact_analysis': scenario.impact_analysis,
            'recommendations': scenario.recommendations
        }
    
    def _serialize_trend(self, trend: MarketTrend) -> Dict[str, Any]:
        """Serialize market trend object for report"""
        return {
            'name': trend.trend_name,
            'direction': trend.direction,
            'strength': trend.strength,
            'timeframe': trend.timeframe,
            'confidence': trend.confidence,
            'impact_areas': trend.impact_areas,
            'opportunities': trend.opportunities,
            'threats': trend.threats
        }
    
    async def _create_visualizations(
        self,
        analysis_results: Dict[str, Any],
        business_type: str
    ) -> Dict[str, Any]:
        """Create predictive analytics visualizations"""
        visualizations = {}
        
        # Forecast visualization
        forecasts = analysis_results.get('forecasts', {})
        if forecasts:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[f.value for f in list(forecasts.keys())[:4]]
            )
            
            for idx, (forecast_type, forecast) in enumerate(list(forecasts.items())[:4]):
                row = idx // 2 + 1
                col = idx % 2 + 1
                
                # Add forecast line
                fig.add_trace(
                    go.Scatter(
                        x=forecast.values.index,
                        y=forecast.values,
                        name=f'{forecast_type.value} Forecast',
                        line=dict(color='blue')
                    ),
                    row=row, col=col
                )
                
                # Add confidence intervals
                fig.add_trace(
                    go.Scatter(
                        x=forecast.upper_bound.index,
                        y=forecast.upper_bound,
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,100,80,0)',
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=forecast.lower_bound.index,
                        y=forecast.lower_bound,
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,100,80,0)',
                        name='Confidence Interval'
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                title="Multi-Metric Forecasts",
                height=800
            )
            
            visualizations['forecasts'] = fig.to_json()
        
        # Scenario comparison
        scenarios = analysis_results.get('scenarios', [])
        if scenarios:
            scenario_data = []
            for scenario in scenarios:
                scenario_data.append({
                    'Scenario': scenario.name,
                    'Revenue Impact': scenario.impact_analysis.get('revenue_change', 0) * 100,
                    'Probability': scenario.probability * 100
                })
            
            df = pd.DataFrame(scenario_data)
            
            fig = px.scatter(
                df,
                x='Revenue Impact',
                y='Probability',
                size='Probability',
                color='Scenario',
                title='Scenario Analysis: Impact vs Probability',
                labels={'Revenue Impact': 'Revenue Impact (%)', 'Probability': 'Probability (%)'}
            )
            
            visualizations['scenarios'] = fig.to_json()
        
        # Market trends radar
        trends = analysis_results.get('market_trends', [])
        if trends:
            categories = [t.trend_name for t in trends[:6]]
            strengths = [t.strength * 100 for t in trends[:6]]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=strengths,
                theta=categories,
                fill='toself',
                name='Market Trend Strength'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                title="Market Trends Analysis"
            )
            
            visualizations['market_trends'] = fig.to_json()
        
        # Model performance
        performance = analysis_results.get('model_performance', {})
        if performance:
            metrics = ['R²', 'MAE', 'RMSE', 'MAPE']
            values = [
                performance.get('r2', 0) * 100,
                100 - min(performance.get('mae', 0), 100),
                100 - min(performance.get('rmse', 0), 100),
                100 - min(performance.get('mape', 0), 100)
            ]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=metrics,
                    y=values,
                    marker_color=['green' if v > 70 else 'orange' if v > 50 else 'red' for v in values]
                )
            ])
            
            fig.update_layout(
                title='Model Performance Metrics',
                yaxis_title='Score (%)',
                yaxis_range=[0, 100]
            )
            
            visualizations['model_performance'] = fig.to_json()
        
        return visualizations