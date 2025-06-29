import React, { useEffect } from 'react';
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
} from '@mui/material';
import {
  TrendingUp,
  People,
  Visibility,
  ThumbUp,
  Schedule,
  CheckCircle,
} from '@mui/icons-material';
import useAnalyticsStore from '../../store/analyticsStore';

const Dashboard = () => {
  const { dashboard, loading, error, fetchDashboard } = useAnalyticsStore();

  useEffect(() => {
    fetchDashboard();
  }, [fetchDashboard]);

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

  const stats = [
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
      title: 'Content Published',
      value: dashboard?.total_content_published || 0,
      icon: <CheckCircle />,
      color: 'info',
      change: dashboard?.content_growth || 0,
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

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {stats.map((stat, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography color="textSecondary" gutterBottom>
                      {stat.title}
                    </Typography>
                    <Typography variant="h4" component="div">
                      {stat.value}
                    </Typography>
                    {stat.change !== undefined && (
                      <Box display="flex" alignItems="center" mt={1}>
                        <TrendingUp 
                          sx={{ 
                            color: stat.change >= 0 ? 'success.main' : 'error.main',
                            mr: 0.5 
                          }} 
                        />
                        <Typography 
                          variant="body2" 
                          color={stat.change >= 0 ? 'success.main' : 'error.main'}
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
                <Typography color="textSecondary">
                  No recent activity
                </Typography>
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
                <Typography color="textSecondary">
                  No content data available
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard; 