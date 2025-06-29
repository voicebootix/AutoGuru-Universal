"""
Executive Dashboard Generator Module

Provides real-time executive insights with unified view across all analytics.
Works universally for any business niche through AI-driven customization.

Features:
- Real-time KPI monitoring
- Strategic alerts and notifications
- Integrated insights from all analytics modules
- Customizable executive views
- Multi-format dashboard delivery
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
import json

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash_daq import Gauge, Indicator

from .base_analytics import (
    UniversalAnalyticsEngine, AnalyticsInsight, InsightPriority,
    AnalyticsScope, BusinessKPI, AnalyticsRequest, ReportFormat
)
from .cross_platform_analytics import CrossPlatformAnalyticsEngine
from .bi_reports import BusinessIntelligenceReports
from .customer_success_analytics import CustomerSuccessAnalytics
from .predictive_modeling import PredictiveBusinessModeling
from .competitive_intelligence import CompetitiveIntelligenceSystem
from backend.core.persona_factory import PersonaFactory
from backend.utils.encryption import EncryptionManager
from backend.database.connection import get_db_context

logger = logging.getLogger(__name__)


class DashboardTheme(Enum):
    """Dashboard theme options"""
    LIGHT = "light"
    DARK = "dark"
    CORPORATE = "corporate"
    MODERN = "modern"
    MINIMAL = "minimal"


class WidgetType(Enum):
    """Types of dashboard widgets"""
    KPI_CARD = "kpi_card"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    PIE_CHART = "pie_chart"
    GAUGE = "gauge"
    HEATMAP = "heatmap"
    TABLE = "table"
    INSIGHT_CARD = "insight_card"
    ALERT = "alert"
    TREND_INDICATOR = "trend_indicator"


class UpdateFrequency(Enum):
    """Dashboard update frequencies"""
    REAL_TIME = "real_time"
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MANUAL = "manual"


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    widget_type: WidgetType
    title: str
    data_source: str
    position: Dict[str, int]  # x, y, width, height
    config: Dict[str, Any]
    update_frequency: UpdateFrequency
    filters: Optional[Dict[str, Any]] = None
    interactions: Optional[List[str]] = None


@dataclass
class DashboardLayout:
    """Dashboard layout configuration"""
    layout_id: str
    name: str
    description: str
    widgets: List[DashboardWidget]
    theme: DashboardTheme
    grid_size: Tuple[int, int]
    responsive: bool
    custom_css: Optional[str] = None


@dataclass
class ExecutiveSummary:
    """Executive summary data structure"""
    period: str
    highlights: List[str]
    key_metrics: Dict[str, Any]
    critical_alerts: List[Dict[str, Any]]
    opportunities: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime


class DashboardDataAggregator:
    """Aggregates data from multiple analytics engines"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.persona_factory = PersonaFactory()
        
        # Initialize analytics engines
        self.cross_platform = CrossPlatformAnalyticsEngine(encryption_manager)
        self.bi_reports = BusinessIntelligenceReports(encryption_manager)
        self.customer_success = CustomerSuccessAnalytics(encryption_manager)
        self.predictive_modeling = PredictiveBusinessModeling(encryption_manager)
        self.competitive_intel = CompetitiveIntelligenceSystem(encryption_manager)
        
    async def aggregate_executive_data(
        self,
        request: AnalyticsRequest
    ) -> Dict[str, Any]:
        """Aggregate data from all analytics engines"""
        try:
            # Run all analytics in parallel
            results = await asyncio.gather(
                self._get_cross_platform_data(request),
                self._get_bi_data(request),
                self._get_customer_data(request),
                self._get_predictive_data(request),
                self._get_competitive_data(request),
                return_exceptions=True
            )
            
            # Process results
            aggregated_data = {
                'cross_platform': results[0] if not isinstance(results[0], Exception) else {},
                'business_intelligence': results[1] if not isinstance(results[1], Exception) else {},
                'customer_success': results[2] if not isinstance(results[2], Exception) else {},
                'predictive_analytics': results[3] if not isinstance(results[3], Exception) else {},
                'competitive_intelligence': results[4] if not isinstance(results[4], Exception) else {},
                'timestamp': datetime.now(),
                'business_type': request.business_type
            }
            
            # Extract key metrics
            aggregated_data['key_metrics'] = self._extract_key_metrics(aggregated_data)
            
            # Generate alerts
            aggregated_data['alerts'] = self._generate_alerts(aggregated_data)
            
            return aggregated_data
            
        except Exception as e:
            logger.error(f"Error aggregating executive data: {e}")
            raise
    
    async def _get_cross_platform_data(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Get cross-platform analytics data"""
        try:
            kpis = await self.cross_platform.calculate_kpis(request)
            analysis = await self.cross_platform.analyze_data(request)
            
            return {
                'kpis': kpis,
                'platform_performance': analysis.get('platform_performance', {}),
                'content_performance': analysis.get('content_performance', {}),
                'audience_insights': analysis.get('audience_insights', {})
            }
        except Exception as e:
            logger.error(f"Error getting cross-platform data: {e}")
            return {}
    
    async def _get_bi_data(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Get business intelligence data"""
        try:
            kpis = await self.bi_reports.calculate_kpis(request)
            analysis = await self.bi_reports.analyze_data(request)
            
            return {
                'kpis': kpis,
                'financial_metrics': analysis.get('financial_analysis', {}).get('summary', {}),
                'operational_metrics': analysis.get('operational_analysis', {}).get('summary', {}),
                'market_metrics': analysis.get('market_analysis', {}).get('summary', {})
            }
        except Exception as e:
            logger.error(f"Error getting BI data: {e}")
            return {}
    
    async def _get_customer_data(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Get customer success data"""
        try:
            kpis = await self.customer_success.calculate_kpis(request)
            analysis = await self.customer_success.analyze_data(request)
            
            return {
                'kpis': kpis,
                'health_statistics': analysis.get('health_statistics', {}),
                'churn_statistics': analysis.get('churn_statistics', {}),
                'segment_distribution': analysis.get('segment_distribution', {})
            }
        except Exception as e:
            logger.error(f"Error getting customer data: {e}")
            return {}
    
    async def _get_predictive_data(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Get predictive analytics data"""
        try:
            kpis = await self.predictive_modeling.calculate_kpis(request)
            analysis = await self.predictive_modeling.analyze_data(request)
            
            return {
                'kpis': kpis,
                'forecasts': analysis.get('forecasts', {}),
                'scenarios': analysis.get('scenarios', []),
                'market_trends': analysis.get('market_trends', [])
            }
        except Exception as e:
            logger.error(f"Error getting predictive data: {e}")
            return {}
    
    async def _get_competitive_data(self, request: AnalyticsRequest) -> Dict[str, Any]:
        """Get competitive intelligence data"""
        try:
            kpis = await self.competitive_intel.calculate_kpis(request)
            analysis = await self.competitive_intel.analyze_data(request)
            
            return {
                'kpis': kpis,
                'competitive_position': analysis.get('competitive_position', {}),
                'competitor_profiles': analysis.get('competitor_profiles', []),
                'market_dynamics': analysis.get('market_dynamics', {})
            }
        except Exception as e:
            logger.error(f"Error getting competitive data: {e}")
            return {}
    
    def _extract_key_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from aggregated data"""
        metrics = {}
        
        # Financial metrics
        bi_data = data.get('business_intelligence', {})
        if bi_data:
            financial = bi_data.get('financial_metrics', {})
            metrics['revenue'] = financial.get('total_revenue', 0)
            metrics['profit_margin'] = financial.get('profit_margin', 0)
            metrics['growth_rate'] = financial.get('revenue_growth_rate', 0)
        
        # Customer metrics
        customer_data = data.get('customer_success', {})
        if customer_data:
            health = customer_data.get('health_statistics', {})
            churn = customer_data.get('churn_statistics', {})
            metrics['customer_health'] = health.get('avg_health_score', 0)
            metrics['churn_rate'] = churn.get('predicted_churn_rate', 0)
            metrics['total_customers'] = health.get('total_customers', 0)
        
        # Platform metrics
        platform_data = data.get('cross_platform', {})
        if platform_data:
            performance = platform_data.get('platform_performance', {})
            metrics['total_reach'] = performance.get('total_reach', 0)
            metrics['engagement_rate'] = performance.get('avg_engagement_rate', 0)
        
        # Competitive metrics
        competitive_data = data.get('competitive_intelligence', {})
        if competitive_data:
            position = competitive_data.get('competitive_position', {})
            metrics['market_share'] = position.get('market_share', 0)
            metrics['competitive_strength'] = position.get('relative_strength', 0)
        
        return metrics
    
    def _generate_alerts(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on data thresholds"""
        alerts = []
        
        # Revenue alerts
        metrics = data.get('key_metrics', {})
        if metrics.get('profit_margin', 1) < 0.05:
            alerts.append({
                'type': 'critical',
                'category': 'financial',
                'message': 'Profit margin below 5%',
                'value': metrics['profit_margin'],
                'action': 'Review cost structure immediately'
            })
        
        # Customer alerts
        if metrics.get('churn_rate', 0) > 0.15:
            alerts.append({
                'type': 'warning',
                'category': 'customer',
                'message': 'High churn risk detected',
                'value': metrics['churn_rate'],
                'action': 'Activate retention campaigns'
            })
        
        # Competitive alerts
        if metrics.get('competitive_strength', 1) < 0.5:
            alerts.append({
                'type': 'warning',
                'category': 'competitive',
                'message': 'Weak competitive position',
                'value': metrics['competitive_strength'],
                'action': 'Review competitive strategy'
            })
        
        return alerts


class DashboardBuilder:
    """Builds interactive executive dashboards"""
    
    def __init__(self):
        self.app = None
        self.theme_colors = {
            DashboardTheme.LIGHT: {
                'background': '#FFFFFF',
                'text': '#2C3E50',
                'primary': '#3498DB',
                'secondary': '#2ECC71',
                'danger': '#E74C3C',
                'warning': '#F39C12'
            },
            DashboardTheme.DARK: {
                'background': '#1A1A2E',
                'text': '#ECF0F1',
                'primary': '#3498DB',
                'secondary': '#27AE60',
                'danger': '#E74C3C',
                'warning': '#F1C40F'
            }
        }
    
    def create_dashboard_app(
        self,
        layout: DashboardLayout,
        initial_data: Dict[str, Any]
    ) -> dash.Dash:
        """Create Dash application for the dashboard"""
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
        
        # Set layout
        self.app.layout = self._create_layout(layout, initial_data)
        
        # Register callbacks
        self._register_callbacks(layout)
        
        return self.app
    
    def _create_layout(
        self,
        layout: DashboardLayout,
        data: Dict[str, Any]
    ) -> html.Div:
        """Create dashboard layout"""
        theme_colors = self.theme_colors[layout.theme]
        
        # Create header
        header = self._create_header(data, theme_colors)
        
        # Create widgets grid
        widgets_grid = self._create_widgets_grid(layout.widgets, data, theme_colors)
        
        # Create footer
        footer = self._create_footer(theme_colors)
        
        # Combine all elements
        return html.Div([
            header,
            widgets_grid,
            footer,
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # Update every minute
                n_intervals=0
            )
        ], style={
            'backgroundColor': theme_colors['background'],
            'color': theme_colors['text'],
            'minHeight': '100vh'
        })
    
    def _create_header(
        self,
        data: Dict[str, Any],
        theme_colors: Dict[str, str]
    ) -> dbc.Container:
        """Create dashboard header"""
        business_type = data.get('business_type', 'Business')
        timestamp = data.get('timestamp', datetime.now())
        
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1(
                        f"{business_type} Executive Dashboard",
                        style={'color': theme_colors['primary']}
                    ),
                    html.P(
                        f"Last updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                        style={'color': theme_colors['text'], 'opacity': 0.7}
                    )
                ], width=8),
                dbc.Col([
                    self._create_alert_summary(data.get('alerts', []), theme_colors)
                ], width=4)
            ])
        ], fluid=True, style={'padding': '20px 0'})
    
    def _create_alert_summary(
        self,
        alerts: List[Dict[str, Any]],
        theme_colors: Dict[str, str]
    ) -> html.Div:
        """Create alert summary widget"""
        critical_count = len([a for a in alerts if a['type'] == 'critical'])
        warning_count = len([a for a in alerts if a['type'] == 'warning'])
        
        return html.Div([
            html.H5("Alerts", style={'marginBottom': '10px'}),
            html.Div([
                html.Span(
                    f"{critical_count} Critical",
                    style={
                        'color': theme_colors['danger'],
                        'marginRight': '20px',
                        'fontWeight': 'bold'
                    }
                ),
                html.Span(
                    f"{warning_count} Warnings",
                    style={
                        'color': theme_colors['warning'],
                        'fontWeight': 'bold'
                    }
                )
            ])
        ])
    
    def _create_widgets_grid(
        self,
        widgets: List[DashboardWidget],
        data: Dict[str, Any],
        theme_colors: Dict[str, str]
    ) -> dbc.Container:
        """Create grid of dashboard widgets"""
        rows = []
        
        # Group widgets by row
        widgets_by_row = defaultdict(list)
        for widget in widgets:
            row = widget.position['y']
            widgets_by_row[row].append(widget)
        
        # Create rows
        for row_num in sorted(widgets_by_row.keys()):
            row_widgets = widgets_by_row[row_num]
            row_content = []
            
            for widget in sorted(row_widgets, key=lambda w: w.position['x']):
                widget_element = self._create_widget(widget, data, theme_colors)
                row_content.append(
                    dbc.Col(
                        widget_element,
                        width=widget.position['width']
                    )
                )
            
            rows.append(dbc.Row(row_content, style={'marginBottom': '20px'}))
        
        return dbc.Container(rows, fluid=True)
    
    def _create_widget(
        self,
        widget: DashboardWidget,
        data: Dict[str, Any],
        theme_colors: Dict[str, str]
    ) -> html.Div:
        """Create individual widget based on type"""
        widget_creators = {
            WidgetType.KPI_CARD: self._create_kpi_card,
            WidgetType.LINE_CHART: self._create_line_chart,
            WidgetType.BAR_CHART: self._create_bar_chart,
            WidgetType.PIE_CHART: self._create_pie_chart,
            WidgetType.GAUGE: self._create_gauge,
            WidgetType.TABLE: self._create_table,
            WidgetType.INSIGHT_CARD: self._create_insight_card
        }
        
        creator = widget_creators.get(widget.widget_type)
        if creator:
            return creator(widget, data, theme_colors)
        
        return html.Div("Widget type not supported")
    
    def _create_kpi_card(
        self,
        widget: DashboardWidget,
        data: Dict[str, Any],
        theme_colors: Dict[str, str]
    ) -> dbc.Card:
        """Create KPI card widget"""
        # Extract KPI value from data
        kpi_value = self._extract_widget_data(widget, data)
        
        # Determine trend
        trend = widget.config.get('trend', 'stable')
        trend_icon = '↑' if trend == 'up' else '↓' if trend == 'down' else '→'
        trend_color = (
            theme_colors['secondary'] if trend == 'up'
            else theme_colors['danger'] if trend == 'down'
            else theme_colors['text']
        )
        
        return dbc.Card([
            dbc.CardBody([
                html.H6(widget.title, style={'opacity': 0.7}),
                html.H2(
                    self._format_value(kpi_value, widget.config.get('format', 'number')),
                    style={'margin': '10px 0'}
                ),
                html.Div([
                    html.Span(
                        trend_icon,
                        style={'color': trend_color, 'fontSize': '24px', 'marginRight': '10px'}
                    ),
                    html.Span(
                        widget.config.get('change', '0%'),
                        style={'color': trend_color}
                    )
                ])
            ])
        ], style={
            'backgroundColor': theme_colors['background'],
            'border': f"1px solid {theme_colors['text']}22",
            'height': '100%'
        })
    
    def _create_line_chart(
        self,
        widget: DashboardWidget,
        data: Dict[str, Any],
        theme_colors: Dict[str, str]
    ) -> dbc.Card:
        """Create line chart widget"""
        chart_data = self._extract_widget_data(widget, data)
        
        if isinstance(chart_data, dict) and 'x' in chart_data and 'y' in chart_data:
            fig = go.Figure()
            
            # Add traces
            if isinstance(chart_data['y'], list):
                fig.add_trace(go.Scatter(
                    x=chart_data['x'],
                    y=chart_data['y'],
                    mode='lines+markers',
                    name=widget.config.get('series_name', 'Value'),
                    line=dict(color=theme_colors['primary'])
                ))
            elif isinstance(chart_data['y'], dict):
                # Multiple series
                for series_name, series_data in chart_data['y'].items():
                    fig.add_trace(go.Scatter(
                        x=chart_data['x'],
                        y=series_data,
                        mode='lines+markers',
                        name=series_name
                    ))
            
            # Update layout
            fig.update_layout(
                title=widget.title,
                template='plotly_dark' if widget.config.get('theme') == 'dark' else 'plotly_white',
                height=widget.config.get('height', 300),
                showlegend=widget.config.get('show_legend', True)
            )
            
            return dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig, id=f"graph-{widget.widget_id}")
                ])
            ], style={'height': '100%'})
        
        return dbc.Card([
            dbc.CardBody([
                html.P("No data available for chart")
            ])
        ])
    
    def _create_bar_chart(
        self,
        widget: DashboardWidget,
        data: Dict[str, Any],
        theme_colors: Dict[str, str]
    ) -> dbc.Card:
        """Create bar chart widget"""
        chart_data = self._extract_widget_data(widget, data)
        
        if isinstance(chart_data, dict) and 'x' in chart_data and 'y' in chart_data:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=chart_data['x'],
                y=chart_data['y'],
                marker_color=theme_colors['primary']
            ))
            
            fig.update_layout(
                title=widget.title,
                template='plotly_dark' if widget.config.get('theme') == 'dark' else 'plotly_white',
                height=widget.config.get('height', 300)
            )
            
            return dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig, id=f"graph-{widget.widget_id}")
                ])
            ], style={'height': '100%'})
        
        return dbc.Card([
            dbc.CardBody([
                html.P("No data available for chart")
            ])
        ])
    
    def _create_pie_chart(
        self,
        widget: DashboardWidget,
        data: Dict[str, Any],
        theme_colors: Dict[str, str]
    ) -> dbc.Card:
        """Create pie chart widget"""
        chart_data = self._extract_widget_data(widget, data)
        
        if isinstance(chart_data, dict) and 'labels' in chart_data and 'values' in chart_data:
            fig = go.Figure(data=[go.Pie(
                labels=chart_data['labels'],
                values=chart_data['values'],
                hole=widget.config.get('hole', 0)
            )])
            
            fig.update_layout(
                title=widget.title,
                template='plotly_dark' if widget.config.get('theme') == 'dark' else 'plotly_white',
                height=widget.config.get('height', 300)
            )
            
            return dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig, id=f"graph-{widget.widget_id}")
                ])
            ], style={'height': '100%'})
        
        return dbc.Card([
            dbc.CardBody([
                html.P("No data available for chart")
            ])
        ])
    
    def _create_gauge(
        self,
        widget: DashboardWidget,
        data: Dict[str, Any],
        theme_colors: Dict[str, str]
    ) -> dbc.Card:
        """Create gauge widget"""
        value = self._extract_widget_data(widget, data)
        
        if value is not None:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': widget.title},
                delta={'reference': widget.config.get('target', value)},
                gauge={
                    'axis': {'range': [None, widget.config.get('max', 100)]},
                    'bar': {'color': theme_colors['primary']},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': widget.config.get('threshold', 90)
                    }
                }
            ))
            
            fig.update_layout(
                template='plotly_dark' if widget.config.get('theme') == 'dark' else 'plotly_white',
                height=widget.config.get('height', 250)
            )
            
            return dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig, id=f"gauge-{widget.widget_id}")
                ])
            ], style={'height': '100%'})
        
        return dbc.Card([
            dbc.CardBody([
                html.P("No data available for gauge")
            ])
        ])
    
    def _create_table(
        self,
        widget: DashboardWidget,
        data: Dict[str, Any],
        theme_colors: Dict[str, str]
    ) -> dbc.Card:
        """Create table widget"""
        table_data = self._extract_widget_data(widget, data)
        
        if isinstance(table_data, list) and table_data:
            return dbc.Card([
                dbc.CardHeader(widget.title),
                dbc.CardBody([
                    dash_table.DataTable(
                        data=table_data,
                        columns=[{"name": i, "id": i} for i in table_data[0].keys()],
                        style_cell={
                            'backgroundColor': theme_colors['background'],
                            'color': theme_colors['text']
                        },
                        style_header={
                            'backgroundColor': theme_colors['primary'],
                            'color': 'white',
                            'fontWeight': 'bold'
                        },
                        page_size=widget.config.get('page_size', 10)
                    )
                ])
            ], style={'height': '100%'})
        
        return dbc.Card([
            dbc.CardBody([
                html.P("No data available for table")
            ])
        ])
    
    def _create_insight_card(
        self,
        widget: DashboardWidget,
        data: Dict[str, Any],
        theme_colors: Dict[str, str]
    ) -> dbc.Card:
        """Create insight card widget"""
        insights = self._extract_widget_data(widget, data)
        
        if insights and isinstance(insights, list):
            insight = insights[0] if insights else {}
            
            priority_colors = {
                'critical': theme_colors['danger'],
                'high': theme_colors['warning'],
                'medium': theme_colors['primary'],
                'low': theme_colors['secondary']
            }
            
            return dbc.Card([
                dbc.CardHeader([
                    html.Span(widget.title),
                    dbc.Badge(
                        insight.get('priority', 'medium'),
                        color=priority_colors.get(insight.get('priority', 'medium'), 'primary'),
                        className="float-right"
                    )
                ]),
                dbc.CardBody([
                    html.H5(insight.get('title', 'Insight')),
                    html.P(insight.get('description', '')),
                    html.Hr(),
                    html.H6("Impact:"),
                    html.P(insight.get('impact', '')),
                    html.H6("Recommendation:"),
                    html.P(insight.get('recommendation', ''))
                ])
            ], style={'height': '100%'})
        
        return dbc.Card([
            dbc.CardBody([
                html.P("No insights available")
            ])
        ])
    
    def _extract_widget_data(
        self,
        widget: DashboardWidget,
        data: Dict[str, Any]
    ) -> Any:
        """Extract data for widget from aggregated data"""
        # Navigate through data structure based on data_source
        path_parts = widget.data_source.split('.')
        current_data = data
        
        for part in path_parts:
            if isinstance(current_data, dict) and part in current_data:
                current_data = current_data[part]
            else:
                return None
        
        return current_data
    
    def _format_value(self, value: Any, format_type: str) -> str:
        """Format value based on type"""
        if value is None:
            return "N/A"
        
        if format_type == 'currency':
            return f"${value:,.2f}"
        elif format_type == 'percentage':
            return f"{value:.1%}"
        elif format_type == 'number':
            if isinstance(value, (int, float)):
                return f"{value:,.0f}"
        
        return str(value)
    
    def _create_footer(self, theme_colors: Dict[str, str]) -> dbc.Container:
        """Create dashboard footer"""
        return dbc.Container([
            html.Hr(),
            html.P(
                "AutoGuru Universal Executive Dashboard © 2024",
                style={
                    'textAlign': 'center',
                    'opacity': 0.5,
                    'marginTop': '50px'
                }
            )
        ], fluid=True)
    
    def _register_callbacks(self, layout: DashboardLayout):
        """Register dashboard callbacks for interactivity"""
        if not self.app:
            return
        
        # Auto-refresh callback
        @self.app.callback(
            Output('interval-component', 'interval'),
            Input('interval-component', 'n_intervals')
        )
        def update_interval(n):
            # Could be modified based on user preferences
            return 60 * 1000  # 1 minute


class ExecutiveDashboardGenerator(UniversalAnalyticsEngine):
    """
    Executive Dashboard Generator
    
    Provides real-time executive dashboards with comprehensive business
    insights that automatically adapt to any business niche.
    """
    
    def __init__(self, encryption_manager: EncryptionManager):
        super().__init__(encryption_manager)
        self.persona_factory = PersonaFactory()
        self.data_aggregator = DashboardDataAggregator(encryption_manager)
        self.dashboard_builder = DashboardBuilder()
        
    async def generate_executive_summary(
        self,
        analytics_data: Dict[str, Any],
        business_type: str
    ) -> str:
        """Generate executive summary for dashboard"""
        summary_parts = []
        
        # Overall performance
        metrics = analytics_data.get('key_metrics', {})
        summary_parts.append(
            f"**Executive Summary - {business_type}**\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        )
        
        # Financial highlights
        summary_parts.append(
            f"\n**Financial Performance:**\n"
            f"- Revenue: ${metrics.get('revenue', 0):,.2f}\n"
            f"- Profit Margin: {metrics.get('profit_margin', 0):.1%}\n"
            f"- Growth Rate: {metrics.get('growth_rate', 0):.1%}\n"
        )
        
        # Customer metrics
        summary_parts.append(
            f"\n**Customer Metrics:**\n"
            f"- Total Customers: {metrics.get('total_customers', 0):,}\n"
            f"- Customer Health: {metrics.get('customer_health', 0):.1%}\n"
            f"- Churn Rate: {metrics.get('churn_rate', 0):.1%}\n"
        )
        
        # Market position
        summary_parts.append(
            f"\n**Market Position:**\n"
            f"- Market Share: {metrics.get('market_share', 0):.1%}\n"
            f"- Competitive Strength: {metrics.get('competitive_strength', 0):.1%}\n"
        )
        
        # Critical alerts
        alerts = analytics_data.get('alerts', [])
        critical_alerts = [a for a in alerts if a['type'] == 'critical']
        if critical_alerts:
            summary_parts.append("\n**Critical Alerts:**")
            for alert in critical_alerts[:3]:
                summary_parts.append(f"- {alert['message']}: {alert['action']}")
        
        # Key opportunities
        opportunities = self._identify_top_opportunities(analytics_data)
        if opportunities:
            summary_parts.append("\n**Top Opportunities:**")
            for i, opp in enumerate(opportunities[:3], 1):
                summary_parts.append(f"{i}. {opp}")
        
        return "\n".join(summary_parts)
    
    async def calculate_kpis(
        self,
        request: AnalyticsRequest
    ) -> List[BusinessKPI]:
        """Calculate executive dashboard KPIs"""
        # Aggregate data from all sources
        data = await self.data_aggregator.aggregate_executive_data(request)
        
        kpis = []
        metrics = data.get('key_metrics', {})
        
        # Overall business health
        overall_health = self._calculate_overall_health(metrics)
        kpis.append(BusinessKPI(
            name="Overall Business Health",
            value=overall_health,
            unit="score",
            trend=self._determine_health_trend(overall_health),
            target=0.85,
            category="Executive"
        ))
        
        # Revenue per customer
        if metrics.get('total_customers', 0) > 0:
            revenue_per_customer = metrics.get('revenue', 0) / metrics.get('total_customers', 1)
            kpis.append(BusinessKPI(
                name="Revenue per Customer",
                value=revenue_per_customer,
                unit="currency",
                trend="up" if revenue_per_customer > 1000 else "down",
                target=1500,
                category="Financial"
            ))
        
        # Market efficiency
        market_efficiency = (
            metrics.get('market_share', 0) / 
            max(metrics.get('competitive_strength', 0.1), 0.1)
        )
        kpis.append(BusinessKPI(
            name="Market Efficiency Ratio",
            value=market_efficiency,
            unit="ratio",
            trend="stable",
            target=1.2,
            category="Market"
        ))
        
        # Customer lifetime value to CAC ratio
        ltv_cac_ratio = 3.5  # Placeholder - would calculate from real data
        kpis.append(BusinessKPI(
            name="LTV:CAC Ratio",
            value=ltv_cac_ratio,
            unit="ratio",
            trend="up" if ltv_cac_ratio > 3 else "down",
            target=3.0,
            category="Customer"
        ))
        
        # Innovation index
        innovation_score = 0.7  # Placeholder - would calculate from real data
        kpis.append(BusinessKPI(
            name="Innovation Index",
            value=innovation_score,
            unit="score",
            trend="up",
            target=0.8,
            category="Operations"
        ))
        
        return kpis
    
    def _calculate_overall_health(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall business health score"""
        weights = {
            'profit_margin': 0.25,
            'growth_rate': 0.20,
            'customer_health': 0.20,
            'market_share': 0.15,
            'competitive_strength': 0.10,
            'engagement_rate': 0.10
        }
        
        health_score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in metrics:
                value = metrics[metric]
                # Normalize values to 0-1 range
                if metric == 'profit_margin':
                    normalized = min(max(value / 0.2, 0), 1)  # 20% = perfect
                elif metric == 'growth_rate':
                    normalized = min(max(value / 0.3, 0), 1)  # 30% = perfect
                elif metric == 'market_share':
                    normalized = min(value, 1)
                else:
                    normalized = min(max(value, 0), 1)
                
                health_score += normalized * weight
                total_weight += weight
        
        return health_score / total_weight if total_weight > 0 else 0.5
    
    def _determine_health_trend(self, health_score: float) -> str:
        """Determine health trend based on score"""
        if health_score > 0.8:
            return "up"
        elif health_score < 0.6:
            return "down"
        return "stable"
    
    def _identify_top_opportunities(
        self,
        data: Dict[str, Any]
    ) -> List[str]:
        """Identify top opportunities from all analytics"""
        opportunities = []
        
        # Extract opportunities from different modules
        predictive = data.get('predictive_analytics', {})
        if predictive:
            market_trends = predictive.get('market_trends', [])
            for trend in market_trends[:2]:
                if isinstance(trend, dict) and trend.get('direction') == 'positive':
                    opportunities.extend(trend.get('opportunities', [])[:1])
        
        competitive = data.get('competitive_intelligence', {})
        if competitive:
            position = competitive.get('competitive_position', {})
            opportunities.extend(position.get('opportunities', [])[:2])
        
        # Limit and deduplicate
        return list(dict.fromkeys(opportunities))[:5]
    
    async def analyze_data(
        self,
        request: AnalyticsRequest
    ) -> Dict[str, Any]:
        """Perform comprehensive executive analysis"""
        try:
            # Aggregate all data
            aggregated_data = await self.data_aggregator.aggregate_executive_data(request)
            
            # Generate executive summary object
            executive_summary = ExecutiveSummary(
                period=f"{request.time_range['start']} to {request.time_range['end']}",
                highlights=self._generate_highlights(aggregated_data),
                key_metrics=aggregated_data.get('key_metrics', {}),
                critical_alerts=[
                    a for a in aggregated_data.get('alerts', [])
                    if a['type'] == 'critical'
                ],
                opportunities=self._extract_opportunities(aggregated_data),
                recommendations=self._generate_recommendations(aggregated_data),
                generated_at=datetime.now()
            )
            
            # Create default dashboard layout
            dashboard_layout = self._create_default_layout(request.business_type)
            
            return {
                'aggregated_data': aggregated_data,
                'executive_summary': executive_summary,
                'dashboard_layout': dashboard_layout,
                'update_frequency': UpdateFrequency.MINUTE,
                'theme': DashboardTheme.MODERN
            }
            
        except Exception as e:
            logger.error(f"Error in executive analysis: {e}")
            raise
    
    def _generate_highlights(self, data: Dict[str, Any]) -> List[str]:
        """Generate key highlights from data"""
        highlights = []
        metrics = data.get('key_metrics', {})
        
        # Revenue highlight
        if metrics.get('growth_rate', 0) > 0.1:
            highlights.append(
                f"Revenue growing at {metrics['growth_rate']:.1%} - exceeding targets"
            )
        
        # Customer highlight
        if metrics.get('customer_health', 0) > 0.8:
            highlights.append(
                f"Customer health score at {metrics['customer_health']:.1%} - all-time high"
            )
        
        # Market highlight
        if metrics.get('market_share', 0) > 0.2:
            highlights.append(
                f"Market leader with {metrics['market_share']:.1%} market share"
            )
        
        # Platform performance
        if metrics.get('engagement_rate', 0) > 0.05:
            highlights.append(
                f"Platform engagement at {metrics['engagement_rate']:.1%} - above industry average"
            )
        
        return highlights
    
    def _extract_opportunities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and prioritize opportunities"""
        all_opportunities = []
        
        # Get opportunities from each module
        modules = [
            'cross_platform',
            'business_intelligence',
            'customer_success',
            'predictive_analytics',
            'competitive_intelligence'
        ]
        
        for module in modules:
            module_data = data.get(module, {})
            if isinstance(module_data, dict):
                # Look for insights or opportunities
                insights = module_data.get('insights', [])
                for insight in insights:
                    if isinstance(insight, dict):
                        all_opportunities.append({
                            'source': module,
                            'title': insight.get('title', 'Opportunity'),
                            'impact': insight.get('impact', 'Medium'),
                            'priority': insight.get('priority', 'medium'),
                            'action': insight.get('recommendation', '')
                        })
        
        # Sort by priority and return top opportunities
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        all_opportunities.sort(
            key=lambda x: priority_order.get(x['priority'], 3)
        )
        
        return all_opportunities[:10]
    
    def _generate_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate executive recommendations"""
        recommendations = []
        metrics = data.get('key_metrics', {})
        alerts = data.get('alerts', [])
        
        # Financial recommendations
        if metrics.get('profit_margin', 1) < 0.1:
            recommendations.append(
                "Implement cost optimization program to improve profit margins"
            )
        
        # Growth recommendations
        if metrics.get('growth_rate', 0) < 0.05:
            recommendations.append(
                "Accelerate growth through market expansion or product innovation"
            )
        
        # Customer recommendations
        if metrics.get('churn_rate', 0) > 0.1:
            recommendations.append(
                "Launch customer retention initiative to reduce churn"
            )
        
        # Competitive recommendations
        if metrics.get('competitive_strength', 1) < 0.5:
            recommendations.append(
                "Strengthen competitive position through differentiation"
            )
        
        # Alert-based recommendations
        for alert in alerts[:2]:
            if alert.get('action'):
                recommendations.append(alert['action'])
        
        return list(dict.fromkeys(recommendations))[:5]
    
    def _create_default_layout(self, business_type: str) -> DashboardLayout:
        """Create default dashboard layout"""
        return DashboardLayout(
            layout_id="default_executive",
            name=f"{business_type} Executive Dashboard",
            description="Comprehensive executive view of business performance",
            widgets=self._get_default_widgets(),
            theme=DashboardTheme.MODERN,
            grid_size=(12, 10),
            responsive=True
        )
    
    def _get_default_widgets(self) -> List[DashboardWidget]:
        """Get default dashboard widgets"""
        return [
            # Top row - Key KPIs
            DashboardWidget(
                widget_id="revenue_kpi",
                widget_type=WidgetType.KPI_CARD,
                title="Total Revenue",
                data_source="key_metrics.revenue",
                position={'x': 0, 'y': 0, 'width': 3, 'height': 1},
                config={'format': 'currency', 'trend': 'up', 'change': '+12%'},
                update_frequency=UpdateFrequency.HOURLY
            ),
            DashboardWidget(
                widget_id="customers_kpi",
                widget_type=WidgetType.KPI_CARD,
                title="Total Customers",
                data_source="key_metrics.total_customers",
                position={'x': 3, 'y': 0, 'width': 3, 'height': 1},
                config={'format': 'number', 'trend': 'up', 'change': '+8%'},
                update_frequency=UpdateFrequency.HOURLY
            ),
            DashboardWidget(
                widget_id="health_kpi",
                widget_type=WidgetType.KPI_CARD,
                title="Business Health",
                data_source="key_metrics.customer_health",
                position={'x': 6, 'y': 0, 'width': 3, 'height': 1},
                config={'format': 'percentage', 'trend': 'stable', 'change': '+2%'},
                update_frequency=UpdateFrequency.HOURLY
            ),
            DashboardWidget(
                widget_id="market_share_kpi",
                widget_type=WidgetType.KPI_CARD,
                title="Market Share",
                data_source="key_metrics.market_share",
                position={'x': 9, 'y': 0, 'width': 3, 'height': 1},
                config={'format': 'percentage', 'trend': 'up', 'change': '+5%'},
                update_frequency=UpdateFrequency.DAILY
            ),
            
            # Second row - Charts
            DashboardWidget(
                widget_id="revenue_trend",
                widget_type=WidgetType.LINE_CHART,
                title="Revenue Trend",
                data_source="business_intelligence.financial_metrics.revenue_trend",
                position={'x': 0, 'y': 1, 'width': 6, 'height': 2},
                config={'height': 350, 'show_legend': True},
                update_frequency=UpdateFrequency.HOURLY
            ),
            DashboardWidget(
                widget_id="customer_segments",
                widget_type=WidgetType.PIE_CHART,
                title="Customer Segments",
                data_source="customer_success.segment_distribution",
                position={'x': 6, 'y': 1, 'width': 6, 'height': 2},
                config={'height': 350, 'hole': 0.4},
                update_frequency=UpdateFrequency.DAILY
            ),
            
            # Third row - Insights and alerts
            DashboardWidget(
                widget_id="top_insight",
                widget_type=WidgetType.INSIGHT_CARD,
                title="Key Insight",
                data_source="predictive_analytics.insights",
                position={'x': 0, 'y': 3, 'width': 6, 'height': 2},
                config={},
                update_frequency=UpdateFrequency.HOURLY
            ),
            DashboardWidget(
                widget_id="alerts_table",
                widget_type=WidgetType.TABLE,
                title="Active Alerts",
                data_source="alerts",
                position={'x': 6, 'y': 3, 'width': 6, 'height': 2},
                config={'page_size': 5},
                update_frequency=UpdateFrequency.MINUTE
            )
        ]
    
    async def generate_report(
        self,
        request: AnalyticsRequest,
        format: ReportFormat
    ) -> Union[Dict[str, Any], str, bytes]:
        """Generate executive dashboard or report"""
        # Perform analysis
        analysis_results = await self.analyze_data(request)
        
        if format == ReportFormat.DASHBOARD:
            # Create interactive dashboard
            dashboard = await self._generate_dashboard(analysis_results)
            return dashboard
        else:
            # Generate static report
            report_data = {
                'title': f'Executive Dashboard Report - {request.business_type}',
                'generated_at': datetime.now().isoformat(),
                'time_period': f"{request.time_range['start']} to {request.time_range['end']}",
                'executive_summary': await self.generate_executive_summary(
                    analysis_results['aggregated_data'],
                    request.business_type
                ),
                'kpis': await self.calculate_kpis(request),
                'key_metrics': analysis_results['aggregated_data']['key_metrics'],
                'alerts': analysis_results['aggregated_data']['alerts'],
                'opportunities': analysis_results['executive_summary'].opportunities,
                'recommendations': analysis_results['executive_summary'].recommendations,
                'visualizations': await self._create_static_visualizations(
                    analysis_results['aggregated_data']
                )
            }
            
            if format == ReportFormat.JSON:
                return report_data
            elif format == ReportFormat.PDF:
                return await self._generate_pdf_report(report_data)
            elif format == ReportFormat.EXCEL:
                return await self._generate_excel_report(report_data)
            else:
                return report_data
    
    async def _generate_dashboard(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate interactive dashboard"""
        # Create Dash app
        layout = analysis_results['dashboard_layout']
        data = analysis_results['aggregated_data']
        
        app = self.dashboard_builder.create_dashboard_app(layout, data)
        
        # Return dashboard configuration
        return {
            'type': 'dashboard',
            'app': app,
            'layout': layout,
            'data': data,
            'update_frequency': analysis_results['update_frequency'],
            'theme': analysis_results['theme'],
            'access_url': '/executive-dashboard'
        }
    
    async def _create_static_visualizations(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create static visualizations for reports"""
        visualizations = {}
        
        # Business health gauge
        metrics = data.get('key_metrics', {})
        health_score = self._calculate_overall_health(metrics)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Business Health"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))
        
        visualizations['health_gauge'] = fig.to_json()
        
        # KPI comparison
        kpi_names = list(metrics.keys())[:6]
        kpi_values = [metrics[k] for k in kpi_names]
        
        fig = go.Figure(data=[
            go.Bar(
                x=kpi_names,
                y=kpi_values,
                marker_color=['green' if v > 0 else 'red' for v in kpi_values]
            )
        ])
        
        fig.update_layout(
            title="Key Performance Indicators",
            xaxis_title="Metric",
            yaxis_title="Value"
        )
        
        visualizations['kpi_comparison'] = fig.to_json()
        
        # Alert distribution
        alerts = data.get('alerts', [])
        if alerts:
            alert_types = defaultdict(int)
            for alert in alerts:
                alert_types[alert['type']] += 1
            
            fig = px.pie(
                values=list(alert_types.values()),
                names=list(alert_types.keys()),
                title="Alert Distribution"
            )
            
            visualizations['alert_distribution'] = fig.to_json()
        
        return visualizations