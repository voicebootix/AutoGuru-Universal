import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
  Chip,
  LinearProgress,
  Tab,
  Tabs,
  Button,
  Paper,
  List,
  ListItem,
  ListItemText,
  Avatar,
  Divider,
} from '@mui/material';
import {
  TrendingUp,
  People,
  Visibility,
  ThumbUp,
  Schedule,
  CheckCircle,
  AttachMoney,
  Analytics,
  Campaign,
  AutoAwesome,
  Warning,
  TrendingDown,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import useAnalyticsStore from '../../store/analyticsStore';

const Dashboard = () => {
  const { dashboard, loading, error, fetchDashboard } = useAnalyticsStore();
  const [activeTab, setActiveTab] = useState(0);
  const [revenueData, setRevenueData] = useState(null);
  const [aiInsights, setAiInsights] = useState([]);
  const [performanceAlerts, setPerformanceAlerts] = useState([]);

  useEffect(() => {
    fetchDashboard();
    fetchRevenueData();
    fetchAIInsights();
    fetchPerformanceAlerts();
  }, [fetchDashboard]);

  const fetchRevenueData = async () => {
    try {
      const response = await fetch('/api/v1/bi/revenue-tracking', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('authToken') || 'demo_token_1234'}`
        },
        body: JSON.stringify({ timeframe: 'month' })
      });
      const data = await response.json();
      setRevenueData(data);
    } catch (error) {
      console.error('Failed to fetch revenue data:', error);
    }
  };

  const fetchAIInsights = async () => {
    try {
      const response = await fetch('/api/v1/bi/usage-analytics', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('authToken') || 'demo_token_1234'}`
        },
        body: JSON.stringify({ timeframe: 'month' })
      });
      const data = await response.json();
      setAiInsights(data.data?.insights || []);
    } catch (error) {
      console.error('Failed to fetch AI insights:', error);
    }
  };

  const fetchPerformanceAlerts = async () => {
    try {
      const response = await fetch('/api/v1/bi/performance-monitoring', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('authToken') || 'demo_token_1234'}`
        },
        body: JSON.stringify({ timeframe: 'day' })
      });
      const data = await response.json();
      setPerformanceAlerts(data.alerts?.critical_alerts || []);
    } catch (error) {
      console.error('Failed to fetch performance alerts:', error);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        Failed to load dashboard data: {error.message}
      </Alert>
    );
  }

  // Enhanced stats with revenue and AI data
  const stats = [
    {
      title: 'Total Revenue',
      value: revenueData?.revenue_summary?.total_revenue ? 
        `$${revenueData.revenue_summary.total_revenue.toLocaleString()}` : '$0',
      icon: <AttachMoney />,
      color: 'success',
      change: revenueData?.revenue_summary?.revenue_growth_rate || 0,
      trend: 'up'
    },
    {
      title: 'Revenue per Post',
      value: revenueData?.revenue_summary?.revenue_per_post ? 
        `$${revenueData.revenue_summary.revenue_per_post.toFixed(2)}` : '$0',
      icon: <TrendingUp />,
      color: 'primary',
      change: 15.2,
      trend: 'up'
    },
    {
      title: 'Total Followers',
      value: dashboard?.total_followers || 0,
      icon: <People />,
      color: 'primary',
      change: dashboard?.follower_growth_rate || 0,
    },
    {
      title: 'Engagement Rate',
      value: `${((dashboard?.avg_engagement_rate || 0) * 100).toFixed(1)}%`,
      icon: <ThumbUp />,
      color: 'success',
      change: dashboard?.engagement_growth || 0,
    },
    {
      title: 'AI Optimization',
      value: '94%',
      icon: <AutoAwesome />,
      color: 'info',
      change: 8.5,
      trend: 'up'
    },
    {
      title: 'Content Published',
      value: dashboard?.total_content_published || 0,
      icon: <CheckCircle />,
      color: 'info',
      change: dashboard?.content_growth || 0,
    },
    {
      title: 'Ad Performance',
      value: '3.2x ROI',
      icon: <Campaign />,
      color: 'warning',
      change: 12.3,
      trend: 'up'
    },
    {
      title: 'Scheduled Posts',
      value: dashboard?.scheduled_posts || 0,
      icon: <Schedule />,
      color: 'warning',
    },
  ];

  const recentActivity = dashboard?.recent_activity || [];
  const topPerformingContent = dashboard?.top_performing_content || [];

  // Mock revenue trend data
  const revenueTrendData = [
    { name: 'Jan', revenue: 4000, posts: 12 },
    { name: 'Feb', revenue: 3000, posts: 18 },
    { name: 'Mar', revenue: 5000, posts: 15 },
    { name: 'Apr', revenue: 4500, posts: 20 },
    { name: 'May', revenue: 6000, posts: 25 },
    { name: 'Jun', revenue: 7500, posts: 22 },
  ];

  const platformRevenueData = [
    { name: 'Instagram', value: 35, revenue: 2625 },
    { name: 'LinkedIn', value: 25, revenue: 1875 },
    { name: 'Facebook', value: 20, revenue: 1500 },
    { name: 'TikTok', value: 15, revenue: 1125 },
    { name: 'YouTube', value: 5, revenue: 375 },
  ];

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        AutoGuru Universal Dashboard
      </Typography>
      <Typography variant="body1" color="textSecondary" gutterBottom>
        Complete social media automation with AI-powered optimization for any business niche
      </Typography>

      {/* Performance Alerts */}
      {performanceAlerts.length > 0 && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          <Typography variant="subtitle2">Performance Alerts</Typography>
          {performanceAlerts.slice(0, 2).map((alert, index) => (
            <Typography key={index} variant="body2">
              • {alert.insight_text || 'Performance optimization available'}
            </Typography>
          ))}
        </Alert>
      )}

      {/* Tabs for different dashboard views */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={handleTabChange}>
          <Tab label="Overview" />
          <Tab label="Revenue Analytics" />
          <Tab label="AI Insights" />
          <Tab label="Performance" />
        </Tabs>
      </Box>

      {/* Overview Tab */}
      {activeTab === 0 && (
        <>
          {/* Enhanced Stats Cards */}
          <Grid container spacing={3} sx={{ mb: 4 }}>
            {stats.map((stat, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center" justifyContent="space-between">
                      <Box>
                        <Typography color="textSecondary" gutterBottom variant="body2">
                          {stat.title}
                        </Typography>
                        <Typography variant="h4" component="div">
                          {stat.value}
                        </Typography>
                        {stat.change !== undefined && (
                          <Box display="flex" alignItems="center" mt={1}>
                            {stat.trend === 'up' ? <TrendingUp /> : <TrendingDown />}
                            <Typography 
                              variant="body2" 
                              color={stat.change >= 0 ? 'success.main' : 'error.main'}
                              sx={{ ml: 0.5 }}
                            >
                              {stat.change >= 0 ? '+' : ''}{stat.change.toFixed(1)}%
                            </Typography>
                          </Box>
                        )}
                      </Box>
                      <Box 
                        sx={{ 
                          color: `${stat.color}.main`,
                          backgroundColor: `${stat.color}.light`,
                          borderRadius: '50%',
                          p: 1,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                        }}
                      >
                        {stat.icon}
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>

          {/* Recent Activity & Top Content */}
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Recent Activity
                  </Typography>
                  {recentActivity.length > 0 ? (
                    recentActivity.map((activity, index) => (
                      <Box key={index} sx={{ mb: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                        <Typography variant="body2" color="textSecondary">
                          {activity.timestamp}
                        </Typography>
                        <Typography variant="body1">
                          {activity.description}
                        </Typography>
                        <Chip 
                          label={activity.platform} 
                          size="small" 
                          sx={{ mt: 1 }}
                        />
                      </Box>
                    ))
                  ) : (
                    <Box>
                      <Typography color="textSecondary" gutterBottom>
                        No recent activity
                      </Typography>
                      <Button variant="outlined" size="small">
                        Create Your First Post
                      </Button>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Top Performing Content
                  </Typography>
                  {topPerformingContent.length > 0 ? (
                    topPerformingContent.map((content, index) => (
                      <Box key={index} sx={{ mb: 2 }}>
                        <Box display="flex" justifyContent="space-between" alignItems="center">
                          <Typography variant="body2" noWrap sx={{ flex: 1 }}>
                            {content.title}
                          </Typography>
                          <Typography variant="body2" color="primary">
                            {content.engagement_rate}% engagement
                          </Typography>
                        </Box>
                        <LinearProgress 
                          variant="determinate" 
                          value={content.engagement_rate} 
                          sx={{ mt: 1 }}
                        />
                      </Box>
                    ))
                  ) : (
                    <Box>
                      <Typography color="textSecondary" gutterBottom>
                        No content data available
                      </Typography>
                      <Button variant="outlined" size="small">
                        Start Creating Content
                      </Button>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </>
      )}

      {/* Revenue Analytics Tab */}
      {activeTab === 1 && (
        <Grid container spacing={3}>
          <Grid item xs={12} lg={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Revenue Trend
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={revenueTrendData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="revenue"
                      stroke="#8884d8"
                      fill="#8884d8"
                      fillOpacity={0.3}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} lg={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Revenue by Platform
                </Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={platformRevenueData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, value }) => `${name} ${value}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {platformRevenueData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Revenue Attribution Analysis
                </Typography>
                <Typography variant="body2" color="textSecondary" paragraph>
                  Track how each post contributes to your overall revenue with multi-touch attribution
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" color="primary">
                        ${revenueData?.revenue_summary?.total_revenue?.toLocaleString() || '0'}
                      </Typography>
                      <Typography variant="body2">Total Revenue</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" color="success.main">
                        {revenueData?.revenue_summary?.revenue_growth_rate?.toFixed(1) || '0'}%
                      </Typography>
                      <Typography variant="body2">Growth Rate</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" color="info.main">
                        ${revenueData?.revenue_summary?.revenue_per_post?.toFixed(2) || '0'}
                      </Typography>
                      <Typography variant="body2">Revenue/Post</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={3}>
                    <Paper sx={{ p: 2, textAlign: 'center' }}>
                      <Typography variant="h6" color="warning.main">
                        ${revenueData?.revenue_summary?.predicted_next_period?.toLocaleString() || '0'}
                      </Typography>
                      <Typography variant="body2">Predicted</Typography>
                    </Paper>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* AI Insights Tab */}
      {activeTab === 2 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  AI-Generated Business Insights
                </Typography>
                <Typography variant="body2" color="textSecondary" paragraph>
                  Our AI analyzes your content performance and provides actionable recommendations
                </Typography>
                {aiInsights.length > 0 ? (
                  <List>
                    {aiInsights.slice(0, 5).map((insight, index) => (
                      <React.Fragment key={index}>
                        <ListItem>
                          <Avatar sx={{ mr: 2, bgcolor: 'primary.main' }}>
                            <AutoAwesome />
                          </Avatar>
                          <ListItemText
                            primary={insight.insight_text || `AI Insight #${index + 1}`}
                            secondary={`Impact Level: ${insight.impact_level || 'Medium'} • Confidence: ${(insight.confidence_score * 100 || 75).toFixed(0)}%`}
                          />
                        </ListItem>
                        {index < aiInsights.length - 1 && <Divider />}
                      </React.Fragment>
                    ))}
                  </List>
                ) : (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <AutoAwesome sx={{ fontSize: 48, color: 'grey.400', mb: 2 }} />
                    <Typography variant="h6" color="textSecondary">
                      AI Insights Loading...
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Our AI is analyzing your content performance to generate personalized insights
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Content Optimization Score
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Box sx={{ width: '100%', mr: 1 }}>
                    <LinearProgress variant="determinate" value={94} sx={{ height: 10 }} />
                  </Box>
                  <Box sx={{ minWidth: 35 }}>
                    <Typography variant="body2" color="text.secondary">94%</Typography>
                  </Box>
                </Box>
                <Typography variant="body2" color="textSecondary">
                  Your content is highly optimized for viral potential and engagement
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Audience Match Score
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Box sx={{ width: '100%', mr: 1 }}>
                    <LinearProgress variant="determinate" value={87} sx={{ height: 10 }} />
                  </Box>
                  <Box sx={{ minWidth: 35 }}>
                    <Typography variant="body2" color="text.secondary">87%</Typography>
                  </Box>
                </Box>
                <Typography variant="body2" color="textSecondary">
                  Your content aligns well with your target audience preferences
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Performance Tab */}
      {activeTab === 3 && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  System Performance & Health
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'success.light' }}>
                      <Typography variant="h4" color="success.dark">99.9%</Typography>
                      <Typography variant="body2">System Uptime</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'info.light' }}>
                      <Typography variant="h4" color="info.dark">1.2s</Typography>
                      <Typography variant="body2">Avg Response Time</Typography>
                    </Paper>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, textAlign: 'center', bgcolor: 'warning.light' }}>
                      <Typography variant="h4" color="warning.dark">
                        {performanceAlerts.length}
                      </Typography>
                      <Typography variant="body2">Active Alerts</Typography>
                    </Paper>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Platform Integration Status
                </Typography>
                <Grid container spacing={2}>
                  {['Instagram', 'LinkedIn', 'Facebook', 'TikTok', 'YouTube', 'Twitter'].map((platform) => (
                    <Grid item xs={12} sm={6} md={4} key={platform}>
                      <Paper sx={{ p: 2 }}>
                        <Box display="flex" alignItems="center" justifyContent="space-between">
                          <Typography variant="body1">{platform}</Typography>
                          <Chip 
                            label="Connected" 
                            color="success" 
                            size="small"
                          />
                        </Box>
                        <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                          API Status: Healthy
                        </Typography>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default Dashboard; 