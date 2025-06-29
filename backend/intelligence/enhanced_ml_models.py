"""Enhanced Machine Learning Models for Business Intelligence"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from dataclasses import dataclass
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class AnomalyScore:
    """Anomaly detection result"""
    is_anomaly: bool
    confidence: float
    anomaly_type: str
    severity: str
    explanation: str
    recommended_actions: List[str]

class EnhancedAnomalyDetector:
    """Advanced anomaly detection using multiple ML techniques"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            n_estimators=200,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.model_path = model_path or Path("models/anomaly_detector")
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize deep learning model for complex pattern detection
        self.deep_model = self._build_deep_model()
        
    def _build_deep_model(self) -> nn.Module:
        """Build deep learning model for anomaly detection"""
        class AnomalyNet(nn.Module):
            def __init__(self, input_dim: int = 20):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, input_dim)
                )
                
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        return AnomalyNet()
    
    async def train_models(self, historical_data: pd.DataFrame):
        """Train anomaly detection models on historical data"""
        try:
            # Prepare features
            features = self._extract_features(historical_data)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Train Isolation Forest
            self.isolation_forest.fit(scaled_features)
            
            # Train deep model (autoencoder for anomaly detection)
            await self._train_deep_model(scaled_features)
            
            # Save models
            joblib.dump(self.isolation_forest, self.model_path / "isolation_forest.pkl")
            joblib.dump(self.scaler, self.model_path / "scaler.pkl")
            torch.save(self.deep_model.state_dict(), self.model_path / "deep_model.pth")
            
            logger.info("Anomaly detection models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training anomaly models: {e}")
            raise
    
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract relevant features for anomaly detection"""
        features = []
        
        # Revenue features
        if 'revenue' in data.columns:
            features.extend([
                data['revenue'].mean(),
                data['revenue'].std(),
                data['revenue'].pct_change().mean(),
                data['revenue'].rolling(7).mean().iloc[-1] if len(data) > 7 else data['revenue'].mean()
            ])
        
        # Engagement features
        if 'engagement_rate' in data.columns:
            features.extend([
                data['engagement_rate'].mean(),
                data['engagement_rate'].std(),
                data['engagement_rate'].min(),
                data['engagement_rate'].max()
            ])
        
        # Performance features
        if 'response_time' in data.columns:
            features.extend([
                data['response_time'].mean(),
                data['response_time'].quantile(0.95),
                data['response_time'].quantile(0.99)
            ])
        
        # Time-based features
        if 'timestamp' in data.columns:
            data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
            data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
            features.extend([
                data['hour'].mode()[0] if not data['hour'].mode().empty else 12,
                data['day_of_week'].mode()[0] if not data['day_of_week'].mode().empty else 3
            ])
        
        return np.array(features).reshape(1, -1)
    
    async def _train_deep_model(self, features: np.ndarray, epochs: int = 100):
        """Train deep autoencoder for anomaly detection"""
        tensor_features = torch.FloatTensor(features)
        
        optimizer = torch.optim.Adam(self.deep_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.deep_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed = self.deep_model(tensor_features)
            loss = criterion(reconstructed, tensor_features)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    async def detect_anomalies(self, current_data: Dict[str, Any]) -> List[AnomalyScore]:
        """Detect anomalies using ensemble of methods"""
        anomalies = []
        
        # Convert to DataFrame for processing
        df = pd.DataFrame([current_data])
        features = self._extract_features(df)
        
        if hasattr(self, 'scaler') and self.scaler.mean_ is not None:
            scaled_features = self.scaler.transform(features)
            
            # Isolation Forest detection
            iso_score = self.isolation_forest.decision_function(scaled_features)[0]
            iso_anomaly = self.isolation_forest.predict(scaled_features)[0] == -1
            
            # Deep model reconstruction error
            tensor_features = torch.FloatTensor(scaled_features)
            self.deep_model.eval()
            with torch.no_grad():
                reconstructed = self.deep_model(tensor_features)
                reconstruction_error = torch.mean((tensor_features - reconstructed) ** 2).item()
            
            # Combine scores for final decision
            if iso_anomaly or reconstruction_error > 0.1:
                anomaly_type = self._classify_anomaly_type(current_data, iso_score, reconstruction_error)
                severity = self._calculate_severity(iso_score, reconstruction_error)
                
                anomalies.append(AnomalyScore(
                    is_anomaly=True,
                    confidence=min(abs(iso_score) + reconstruction_error, 1.0),
                    anomaly_type=anomaly_type,
                    severity=severity,
                    explanation=self._generate_explanation(anomaly_type, current_data),
                    recommended_actions=self._get_recommended_actions(anomaly_type, severity)
                ))
        
        return anomalies
    
    def _classify_anomaly_type(self, data: Dict[str, Any], iso_score: float, recon_error: float) -> str:
        """Classify the type of anomaly detected"""
        if 'revenue' in data and data.get('revenue', 0) < data.get('expected_revenue', 100) * 0.5:
            return "revenue_drop"
        elif 'response_time' in data and data.get('response_time', 0) > 1000:
            return "performance_degradation"
        elif 'engagement_rate' in data and data.get('engagement_rate', 0) < 1.0:
            return "engagement_decline"
        elif recon_error > 0.2:
            return "unusual_pattern"
        else:
            return "general_anomaly"
    
    def _calculate_severity(self, iso_score: float, recon_error: float) -> str:
        """Calculate anomaly severity"""
        combined_score = abs(iso_score) + recon_error
        
        if combined_score > 0.8:
            return "critical"
        elif combined_score > 0.5:
            return "high"
        elif combined_score > 0.3:
            return "medium"
        else:
            return "low"
    
    def _generate_explanation(self, anomaly_type: str, data: Dict[str, Any]) -> str:
        """Generate human-readable explanation for the anomaly"""
        explanations = {
            "revenue_drop": f"Revenue dropped to ${data.get('revenue', 0):.2f}, significantly below expected levels",
            "performance_degradation": f"System response time increased to {data.get('response_time', 0)}ms",
            "engagement_decline": f"Engagement rate fell to {data.get('engagement_rate', 0):.2f}%",
            "unusual_pattern": "Detected unusual pattern in metrics that doesn't match historical behavior",
            "general_anomaly": "Multiple metrics showing abnormal values"
        }
        return explanations.get(anomaly_type, "Anomaly detected in system metrics")
    
    def _get_recommended_actions(self, anomaly_type: str, severity: str) -> List[str]:
        """Get recommended actions based on anomaly type and severity"""
        actions = {
            "revenue_drop": [
                "Check payment processing systems",
                "Review recent content performance",
                "Analyze competitor activities",
                "Verify tracking implementation"
            ],
            "performance_degradation": [
                "Scale infrastructure resources",
                "Check database query performance",
                "Review recent deployments",
                "Enable performance profiling"
            ],
            "engagement_decline": [
                "Review content quality and relevance",
                "Check posting schedule optimization",
                "Analyze audience demographics changes",
                "Test new content formats"
            ]
        }
        
        base_actions = actions.get(anomaly_type, ["Investigate metric anomalies", "Review system logs"])
        
        if severity == "critical":
            base_actions.insert(0, "IMMEDIATE ACTION REQUIRED")
        
        return base_actions


class PredictiveRevenueModel:
    """Advanced revenue prediction using ensemble methods"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=200, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=150, random_state=42),
            'neural_net': self._build_revenue_nn()
        }
        self.feature_importance = {}
        
    def _build_revenue_nn(self) -> nn.Module:
        """Build neural network for revenue prediction"""
        class RevenueNet(nn.Module):
            def __init__(self, input_dim: int = 30):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
                
            def forward(self, x):
                return self.layers(x)
        
        return RevenueNet()
    
    async def train(self, historical_data: pd.DataFrame):
        """Train all models in the ensemble"""
        # Feature engineering
        features = self._engineer_features(historical_data)
        target = historical_data['revenue'].values
        
        # Train each model
        for name, model in self.models.items():
            if name == 'neural_net':
                await self._train_neural_net(features, target)
            else:
                model.fit(features, target)
                
                # Calculate feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = model.feature_importances_
        
        logger.info("Predictive revenue models trained successfully")
    
    def _engineer_features(self, data: pd.DataFrame) -> np.ndarray:
        """Engineer features for revenue prediction"""
        features = []
        
        # Time-based features
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
        data['month'] = pd.to_datetime(data['timestamp']).dt.month
        data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
        
        # Lag features
        for lag in [1, 7, 30]:
            data[f'revenue_lag_{lag}'] = data['revenue'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            data[f'revenue_rolling_mean_{window}'] = data['revenue'].rolling(window).mean()
            data[f'revenue_rolling_std_{window}'] = data['revenue'].rolling(window).std()
        
        # Platform-specific features
        if 'platform' in data.columns:
            platform_dummies = pd.get_dummies(data['platform'], prefix='platform')
            data = pd.concat([data, platform_dummies], axis=1)
        
        # Content type features
        if 'content_type' in data.columns:
            content_dummies = pd.get_dummies(data['content_type'], prefix='content')
            data = pd.concat([data, content_dummies], axis=1)
        
        # Drop NaN values
        data = data.dropna()
        
        # Select feature columns
        feature_cols = [col for col in data.columns if col not in ['revenue', 'timestamp', 'platform', 'content_type']]
        
        return data[feature_cols].values
    
    async def _train_neural_net(self, features: np.ndarray, target: np.ndarray, epochs: int = 200):
        """Train neural network model"""
        X_tensor = torch.FloatTensor(features)
        y_tensor = torch.FloatTensor(target).reshape(-1, 1)
        
        optimizer = torch.optim.Adam(self.models['neural_net'].parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.models['neural_net'].train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.models['neural_net'](X_tensor)
            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                logger.info(f"Neural Net Training - Epoch {epoch}, Loss: {loss.item():.4f}")
    
    async def predict(self, features: Dict[str, Any], horizon_days: int = 30) -> Dict[str, Any]:
        """Make revenue predictions using ensemble"""
        # Prepare features
        feature_array = self._prepare_prediction_features(features)
        
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            if name == 'neural_net':
                self.models['neural_net'].eval()
                with torch.no_grad():
                    tensor_features = torch.FloatTensor(feature_array)
                    pred = self.models['neural_net'](tensor_features).item()
                predictions[name] = pred
            else:
                predictions[name] = model.predict(feature_array)[0]
        
        # Ensemble prediction (weighted average)
        weights = {'random_forest': 0.4, 'gradient_boost': 0.4, 'neural_net': 0.2}
        ensemble_prediction = sum(predictions[name] * weights[name] for name in predictions)
        
        # Calculate confidence interval
        prediction_std = np.std(list(predictions.values()))
        confidence_interval = (
            ensemble_prediction - 1.96 * prediction_std,
            ensemble_prediction + 1.96 * prediction_std
        )
        
        return {
            'point_forecast': ensemble_prediction,
            'confidence_interval': confidence_interval,
            'model_predictions': predictions,
            'forecast_horizon_days': horizon_days,
            'feature_importance': self._get_top_features()
        }
    
    def _prepare_prediction_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare features for prediction"""
        # This would match the feature engineering done during training
        # For now, return a dummy array
        return np.random.rand(1, 30)
    
    def _get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top n important features"""
        if not self.feature_importance:
            return []
        
        # Average importance across models
        avg_importance = {}
        for model_importance in self.feature_importance.values():
            for i, importance in enumerate(model_importance):
                feature_name = f"feature_{i}"
                avg_importance[feature_name] = avg_importance.get(feature_name, 0) + importance
        
        # Normalize
        n_models = len(self.feature_importance)
        for feature in avg_importance:
            avg_importance[feature] /= n_models
        
        # Sort and return top n
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]


class CustomerSegmentationEngine:
    """AI-powered customer segmentation using clustering"""
    
    def __init__(self):
        self.clustering_model = DBSCAN(eps=0.3, min_samples=5)
        self.pca = PCA(n_components=3)
        self.segment_profiles = {}
        
    async def segment_customers(self, client_id: str, customer_data: pd.DataFrame) -> Dict[str, List[str]]:
        """Segment customers using AI clustering"""
        # Feature engineering for segmentation
        features = self._prepare_segmentation_features(customer_data)
        
        # Reduce dimensions for visualization
        reduced_features = self.pca.fit_transform(features)
        
        # Perform clustering
        clusters = self.clustering_model.fit_predict(features)
        
        # Analyze each cluster
        segments = {
            "high_value": [],
            "growth_potential": [],
            "at_risk": [],
            "champions": [],
            "new_customers": [],
            "dormant": []
        }
        
        for cluster_id in np.unique(clusters):
            if cluster_id == -1:  # Noise points
                continue
                
            cluster_mask = clusters == cluster_id
            cluster_customers = customer_data[cluster_mask]
            
            # Classify cluster based on behavior
            segment_type = self._classify_segment(cluster_customers)
            customer_ids = cluster_customers['customer_id'].tolist()
            
            if segment_type in segments:
                segments[segment_type].extend(customer_ids)
            
            # Store segment profile
            self.segment_profiles[segment_type] = self._create_segment_profile(cluster_customers)
        
        return segments
    
    def _prepare_segmentation_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for customer segmentation"""
        features = []
        
        # RFM features
        if all(col in data.columns for col in ['recency', 'frequency', 'monetary']):
            features.extend([
                data['recency'].values,
                data['frequency'].values,
                data['monetary'].values
            ])
        
        # Engagement features
        if 'engagement_score' in data.columns:
            features.append(data['engagement_score'].values)
        
        # Platform preferences
        platform_cols = [col for col in data.columns if col.startswith('platform_')]
        for col in platform_cols:
            features.append(data[col].values)
        
        return np.array(features).T
    
    def _classify_segment(self, cluster_data: pd.DataFrame) -> str:
        """Classify customer segment based on behavior"""
        avg_monetary = cluster_data['monetary'].mean() if 'monetary' in cluster_data else 0
        avg_frequency = cluster_data['frequency'].mean() if 'frequency' in cluster_data else 0
        avg_recency = cluster_data['recency'].mean() if 'recency' in cluster_data else 0
        
        # High value: High monetary, high frequency, low recency
        if avg_monetary > cluster_data['monetary'].quantile(0.75) and avg_frequency > 5 and avg_recency < 30:
            return "champions"
        
        # Growth potential: Medium monetary, increasing frequency
        elif avg_monetary > cluster_data['monetary'].quantile(0.5) and avg_frequency > 3:
            return "growth_potential"
        
        # At risk: Was high value but increasing recency
        elif avg_monetary > cluster_data['monetary'].quantile(0.5) and avg_recency > 60:
            return "at_risk"
        
        # New customers: Low frequency, low recency
        elif avg_frequency <= 2 and avg_recency < 30:
            return "new_customers"
        
        # Dormant: High recency, low frequency
        elif avg_recency > 90:
            return "dormant"
        
        else:
            return "high_value"
    
    def _create_segment_profile(self, segment_data: pd.DataFrame) -> Dict[str, Any]:
        """Create detailed profile for customer segment"""
        return {
            'size': len(segment_data),
            'avg_lifetime_value': segment_data['monetary'].sum() if 'monetary' in segment_data else 0,
            'avg_order_value': segment_data['monetary'].mean() if 'monetary' in segment_data else 0,
            'avg_frequency': segment_data['frequency'].mean() if 'frequency' in segment_data else 0,
            'preferred_platforms': segment_data['preferred_platform'].mode()[0] if 'preferred_platform' in segment_data and not segment_data['preferred_platform'].mode().empty else 'unknown',
            'common_behaviors': self._extract_common_behaviors(segment_data)
        }
    
    def _extract_common_behaviors(self, data: pd.DataFrame) -> List[str]:
        """Extract common behaviors from segment data"""
        behaviors = []
        
        # Analyze purchase patterns
        if 'purchase_time' in data.columns:
            common_hour = pd.to_datetime(data['purchase_time']).dt.hour.mode()
            if not common_hour.empty:
                behaviors.append(f"Most active at {common_hour[0]}:00")
        
        # Analyze content preferences
        if 'content_type_preference' in data.columns:
            top_content = data['content_type_preference'].value_counts().head(1)
            if not top_content.empty:
                behaviors.append(f"Prefers {top_content.index[0]} content")
        
        return behaviors