import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  LinearProgress,
  Tabs,
  Tab,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Avatar,
  Switch,
  FormControlLabel,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Tooltip,
  Badge,
  Drawer,
  AppBar,
  Toolbar,
  Divider,
  CircularProgress,
} from '@mui/material';
import {
  Dashboard,
  Security,
  Settings,
  People,
  Analytics,
  Warning,
  CheckCircle,
  Error,
  Info,
  Storage,
  NetworkCheck,
  Speed,
  Memory,
  CloudUpload,
  Download,
  Refresh,
  Edit,
  Delete,
  Add,
  Visibility,
  VisibilityOff,
  Lock,
  Shield,
  Monitor,
  Code,
  Bug,
  Backup,
  Update,
  Notifications,
  Email,
  Phone,
  Business,
  Assessment,
  Timeline,
  TrendingUp,
  AutoAwesome,
  Campaign,
  AttachMoney,
} from '@mui/icons-material';

const AdminDashboard = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [systemStats, setSystemStats] = useState(null);
  const [userAccounts, setUserAccounts] = useState([]);
  const [securityLog, setSecurityLog] = useState([]);
  const [performanceMetrics, setPerformanceMetrics] = useState(null);
  const [configSettings, setConfigSettings] = useState({});
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedUser, setSelectedUser] = useState(null);
  const [userDialogOpen, setUserDialogOpen] = useState(false);
  const [backupStatus, setBackupStatus] = useState(null);

  useEffect(() => {
    fetchAdminData();
  }, []);

  const fetchAdminData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        fetchSystemStats(),
        fetchUserAccounts(),
        fetchSecurityLog(),
        fetchPerformanceMetrics(),
        fetchConfigSettings(),
        fetchAlerts(),
        fetchBackupStatus(),
      ]);
    } catch (error) {
      console.error('Failed to fetch admin data:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchSystemStats = async () => {
    try {
      const response = await fetch('/api/v1/admin/system-stats', {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('authToken') || 'demo_token_1234'}` }
      });
      const data = await response.json();
      setSystemStats(data);
    } catch (error) {
      console.error('Failed to fetch system stats:', error);
    }
  };

  const fetchUserAccounts = async () => {
    try {
      const response = await fetch('/api/v1/admin/users', {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('authToken') || 'demo_token_1234'}` }
      });
      const data = await response.json();
      setUserAccounts(data.users || []);
    } catch (error) {
      console.error('Failed to fetch user accounts:', error);
    }
  };

  const fetchSecurityLog = async () => {
    try {
      const response = await fetch('/api/v1/admin/security-log', {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('authToken') || 'demo_token_1234'}` }
      });
      const data = await response.json();
      setSecurityLog(data.security_events || []);
    } catch (error) {
      console.error('Failed to fetch security log:', error);
    }
  };

  const fetchPerformanceMetrics = async () => {
    try {
      const response = await fetch('/api/v1/admin/performance-metrics', {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('authToken') || 'demo_token_1234'}` }
      });
      const data = await response.json();
      setPerformanceMetrics(data);
    } catch (error) {
      console.error('Failed to fetch performance metrics:', error);
    }
  };

  const fetchConfigSettings = async () => {
    try {
      const response = await fetch('/api/v1/admin/config', {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('authToken') || 'demo_token_1234'}` }
      });
      const data = await response.json();
      setConfigSettings(data);
    } catch (error) {
      console.error('Failed to fetch config settings:', error);
    }
  };

  const fetchAlerts = async () => {
    try {
      const response = await fetch('/api/v1/admin/alerts', {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('authToken') || 'demo_token_1234'}` }
      });
      const data = await response.json();
      setAlerts(data.alerts || []);
    } catch (error) {
      console.error('Failed to fetch alerts:', error);
    }
  };

  const fetchBackupStatus = async () => {
    try {
      const response = await fetch('/api/v1/admin/backup-status', {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('authToken') || 'demo_token_1234'}` }
      });
      const data = await response.json();
      setBackupStatus(data);
    } catch (error) {
      console.error('Failed to fetch backup status:', error);
    }
  };

  const handleUserAction = async (userId, action) => {
    try {
      await fetch(`/api/v1/admin/users/${userId}/${action}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${localStorage.getItem('authToken') || 'demo_token_1234'}` }
      });
      fetchUserAccounts(); // Refresh user list
    } catch (error) {
      console.error(`Failed to ${action} user:`, error);
    }
  };

  const handleBackupAction = async (action) => {
    try {
      await fetch(`/api/v1/admin/backup/${action}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${localStorage.getItem('authToken') || 'demo_token_1234'}` }
      });
      fetchBackupStatus(); // Refresh backup status
    } catch (error) {
      console.error(`Failed to ${action} backup:`, error);
    }
  };

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  const getAlertIcon = (severity) => {
    switch (severity) {
      case 'critical': return <Error color="error" />;
      case 'warning': return <Warning color="warning" />;
      case 'info': return <Info color="info" />;
      default: return <CheckCircle color="success" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'success';
      case 'suspended': return 'warning';
      case 'banned': return 'error';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Box mb={3}>
        <Typography variant="h4" gutterBottom>
          Admin Dashboard
        </Typography>
        <Typography variant="body1" color="textSecondary">
          Comprehensive system administration and monitoring
        </Typography>
      </Box>

      {/* Critical Alerts */}
      {alerts.filter(alert => alert.severity === 'critical').length > 0 && (
        <Alert severity="error" sx={{ mb: 3 }}>
          <Typography variant="subtitle2">Critical Alerts</Typography>
          {alerts.filter(alert => alert.severity === 'critical').slice(0, 3).map((alert, index) => (
            <Typography key={index} variant="body2">
              • {alert.message}
            </Typography>
          ))}
        </Alert>
      )}

      {/* Quick Stats */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Total Users
                  </Typography>
                  <Typography variant="h4">
                    {systemStats?.total_users || 0}
                  </Typography>
                  <Typography variant="body2" color="success.main">
                    +{systemStats?.user_growth || 0}% this month
                  </Typography>
                </Box>
                <People sx={{ fontSize: 40, color: 'primary.main' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    System Health
                  </Typography>
                  <Typography variant="h4" color="success.main">
                    {performanceMetrics?.system_health || 98}%
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    All systems operational
                  </Typography>
                </Box>
                <Monitor sx={{ fontSize: 40, color: 'success.main' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Revenue Today
                  </Typography>
                  <Typography variant="h4" color="primary">
                    ${systemStats?.daily_revenue?.toLocaleString() || '0'}
                  </Typography>
                  <Typography variant="body2" color="success.main">
                    +15% vs yesterday
                  </Typography>
                </Box>
                <AttachMoney sx={{ fontSize: 40, color: 'primary.main' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Active Campaigns
                  </Typography>
                  <Typography variant="h4">
                    {systemStats?.active_campaigns || 0}
                  </Typography>
                  <Typography variant="body2" color="info.main">
                    Across all users
                  </Typography>
                </Box>
                <Campaign sx={{ fontSize: 40, color: 'info.main' }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Tabs value={activeTab} onChange={handleTabChange} sx={{ mb: 3 }}>
        <Tab label="System Monitor" />
        <Tab label="User Management" />
        <Tab label="Security" />
        <Tab label="Performance" />
        <Tab label="Configuration" />
        <Tab label="Backups" />
      </Tabs>

      {/* System Monitor Tab */}
      {activeTab === 0 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  System Performance
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Box mb={2}>
                      <Typography variant="body2" gutterBottom>
                        CPU Usage
                      </Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={performanceMetrics?.cpu_usage || 45} 
                        sx={{ height: 8 }}
                      />
                      <Typography variant="body2" color="textSecondary">
                        {performanceMetrics?.cpu_usage || 45}% of capacity
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Box mb={2}>
                      <Typography variant="body2" gutterBottom>
                        Memory Usage
                      </Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={performanceMetrics?.memory_usage || 62} 
                        sx={{ height: 8 }}
                      />
                      <Typography variant="body2" color="textSecondary">
                        {performanceMetrics?.memory_usage || 62}% of capacity
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Box mb={2}>
                      <Typography variant="body2" gutterBottom>
                        Database Load
                      </Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={performanceMetrics?.database_load || 38} 
                        sx={{ height: 8 }}
                        color="info"
                      />
                      <Typography variant="body2" color="textSecondary">
                        {performanceMetrics?.database_load || 38}% load
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <Box mb={2}>
                      <Typography variant="body2" gutterBottom>
                        API Response Time
                      </Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={performanceMetrics?.api_response_time || 25} 
                        sx={{ height: 8 }}
                        color="success"
                      />
                      <Typography variant="body2" color="textSecondary">
                        {performanceMetrics?.api_response_time || 1.2}s avg
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Recent Activity
                </Typography>
                <List>
                  {[
                    { action: 'New user registration', time: '2 minutes ago', type: 'success' },
                    { action: 'High API usage detected', time: '15 minutes ago', type: 'warning' },
                    { action: 'Backup completed', time: '1 hour ago', type: 'success' },
                    { action: 'Security scan completed', time: '2 hours ago', type: 'info' },
                  ].map((activity, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        {getAlertIcon(activity.type)}
                      </ListItemIcon>
                      <ListItemText
                        primary={activity.action}
                        secondary={activity.time}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* User Management Tab */}
      {activeTab === 1 && (
        <Card>
          <CardContent>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">User Management</Typography>
              <Button variant="contained" startIcon={<Add />}>
                Add New User
              </Button>
            </Box>

            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>User</TableCell>
                    <TableCell>Email</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Plan</TableCell>
                    <TableCell>Last Active</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {userAccounts.length > 0 ? userAccounts.map((user) => (
                    <TableRow key={user.id}>
                      <TableCell>
                        <Box display="flex" alignItems="center">
                          <Avatar sx={{ mr: 2 }}>{user.name?.charAt(0) || 'U'}</Avatar>
                          <Box>
                            <Typography variant="body2">{user.name || 'User'}</Typography>
                            <Typography variant="caption" color="textSecondary">
                              ID: {user.id}
                            </Typography>
                          </Box>
                        </Box>
                      </TableCell>
                      <TableCell>{user.email || 'user@example.com'}</TableCell>
                      <TableCell>
                        <Chip 
                          label={user.status || 'active'} 
                          color={getStatusColor(user.status)}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{user.plan || 'Free'}</TableCell>
                      <TableCell>{user.last_active || 'Today'}</TableCell>
                      <TableCell>
                        <Box display="flex" gap={1}>
                          <Tooltip title="View Details">
                            <IconButton 
                              size="small"
                              onClick={() => {
                                setSelectedUser(user);
                                setUserDialogOpen(true);
                              }}
                            >
                              <Visibility />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Edit User">
                            <IconButton size="small">
                              <Edit />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Suspend User">
                            <IconButton 
                              size="small"
                              onClick={() => handleUserAction(user.id, 'suspend')}
                            >
                              <Lock />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </TableCell>
                    </TableRow>
                  )) : (
                    <TableRow>
                      <TableCell colSpan={6} align="center">
                        <Typography color="textSecondary">No users found</Typography>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {/* Security Tab */}
      {activeTab === 2 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Security Log
                </Typography>
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Event</TableCell>
                        <TableCell>User</TableCell>
                        <TableCell>IP Address</TableCell>
                        <TableCell>Timestamp</TableCell>
                        <TableCell>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {securityLog.length > 0 ? securityLog.slice(0, 10).map((log, index) => (
                        <TableRow key={index}>
                          <TableCell>{log.event || 'Login attempt'}</TableCell>
                          <TableCell>{log.user || 'Unknown'}</TableCell>
                          <TableCell>{log.ip_address || '192.168.1.1'}</TableCell>
                          <TableCell>{log.timestamp || 'Just now'}</TableCell>
                          <TableCell>
                            <Chip 
                              label={log.status || 'Success'} 
                              color={log.status === 'Success' ? 'success' : 'error'}
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      )) : (
                        <TableRow>
                          <TableCell colSpan={5} align="center">
                            <Typography color="textSecondary">No security events</Typography>
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Security Settings
                </Typography>
                <List>
                  <ListItem>
                    <ListItemIcon>
                      <Shield />
                    </ListItemIcon>
                    <ListItemText
                      primary="Two-Factor Authentication"
                      secondary="Enabled for all admins"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Lock />
                    </ListItemIcon>
                    <ListItemText
                      primary="Password Policy"
                      secondary="Strong passwords required"
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon>
                      <Security />
                    </ListItemIcon>
                    <ListItemText
                      primary="Firewall"
                      secondary="Active protection"
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Performance Tab */}
      {activeTab === 3 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  API Performance
                </Typography>
                <Box mb={2}>
                  <Typography variant="body2" gutterBottom>
                    Requests per minute
                  </Typography>
                  <Typography variant="h4" color="primary">
                    {performanceMetrics?.requests_per_minute || 1247}
                  </Typography>
                </Box>
                <Box mb={2}>
                  <Typography variant="body2" gutterBottom>
                    Average response time
                  </Typography>
                  <Typography variant="h4" color="success.main">
                    {performanceMetrics?.avg_response_time || 1.2}s
                  </Typography>
                </Box>
                <Box mb={2}>
                  <Typography variant="body2" gutterBottom>
                    Error rate
                  </Typography>
                  <Typography variant="h4" color="error.main">
                    {performanceMetrics?.error_rate || 0.01}%
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Database Performance
                </Typography>
                <Box mb={2}>
                  <Typography variant="body2" gutterBottom>
                    Query performance
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={performanceMetrics?.query_performance || 85} 
                    sx={{ height: 8 }}
                    color="success"
                  />
                  <Typography variant="body2" color="textSecondary">
                    {performanceMetrics?.query_performance || 85}% optimal
                  </Typography>
                </Box>
                <Box mb={2}>
                  <Typography variant="body2" gutterBottom>
                    Connection pool
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={performanceMetrics?.connection_pool || 42} 
                    sx={{ height: 8 }}
                    color="info"
                  />
                  <Typography variant="body2" color="textSecondary">
                    {performanceMetrics?.connection_pool || 42}% utilization
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Configuration Tab */}
      {activeTab === 4 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  System Configuration
                </Typography>
                <Box mb={2}>
                  <FormControlLabel
                    control={<Switch checked={configSettings.maintenance_mode || false} />}
                    label="Maintenance Mode"
                  />
                </Box>
                <Box mb={2}>
                  <FormControlLabel
                    control={<Switch checked={configSettings.debug_mode || false} />}
                    label="Debug Mode"
                  />
                </Box>
                <Box mb={2}>
                  <FormControlLabel
                    control={<Switch checked={configSettings.analytics_enabled || true} />}
                    label="Analytics Enabled"
                  />
                </Box>
                <Box mb={2}>
                  <TextField
                    fullWidth
                    label="Max Users per Plan"
                    type="number"
                    value={configSettings.max_users_per_plan || 1000}
                    size="small"
                  />
                </Box>
                <Button variant="contained" color="primary">
                  Save Configuration
                </Button>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Platform Settings
                </Typography>
                <Box mb={2}>
                  <FormControl fullWidth size="small">
                    <InputLabel>Default Content Tone</InputLabel>
                    <Select value={configSettings.default_content_tone || 'professional'}>
                      <MenuItem value="professional">Professional</MenuItem>
                      <MenuItem value="casual">Casual</MenuItem>
                      <MenuItem value="friendly">Friendly</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
                <Box mb={2}>
                  <TextField
                    fullWidth
                    label="AI Model Version"
                    value={configSettings.ai_model_version || 'gpt-4'}
                    size="small"
                  />
                </Box>
                <Box mb={2}>
                  <TextField
                    fullWidth
                    label="Max Posts per Day"
                    type="number"
                    value={configSettings.max_posts_per_day || 50}
                    size="small"
                  />
                </Box>
                <Button variant="contained" color="primary">
                  Update Settings
                </Button>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Backups Tab */}
      {activeTab === 5 && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Backup Status
                </Typography>
                <Box mb={2}>
                  <Typography variant="body2" color="textSecondary">
                    Last backup: {backupStatus?.last_backup || 'Today at 03:00 AM'}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Next backup: {backupStatus?.next_backup || 'Tomorrow at 03:00 AM'}
                  </Typography>
                </Box>
                <Box mb={2}>
                  <Typography variant="body2" gutterBottom>
                    Backup Health
                  </Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={backupStatus?.backup_health || 100} 
                    sx={{ height: 8 }}
                    color="success"
                  />
                  <Typography variant="body2" color="textSecondary">
                    {backupStatus?.backup_health || 100}% healthy
                  </Typography>
                </Box>
                <Box display="flex" gap={2}>
                  <Button 
                    variant="contained" 
                    startIcon={<Backup />}
                    onClick={() => handleBackupAction('create')}
                  >
                    Create Backup
                  </Button>
                  <Button 
                    variant="outlined" 
                    startIcon={<Download />}
                    onClick={() => handleBackupAction('download')}
                  >
                    Download
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Backup History
                </Typography>
                <List>
                  {(backupStatus?.backup_history || [
                    { date: 'Today 03:00 AM', size: '2.4 GB', status: 'Success' },
                    { date: 'Yesterday 03:00 AM', size: '2.3 GB', status: 'Success' },
                    { date: '2 days ago 03:00 AM', size: '2.2 GB', status: 'Success' },
                  ]).map((backup, index) => (
                    <ListItem key={index}>
                      <ListItemIcon>
                        <CheckCircle color="success" />
                      </ListItemIcon>
                      <ListItemText
                        primary={backup.date}
                        secondary={`${backup.size} - ${backup.status}`}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* User Details Dialog */}
      <Dialog open={userDialogOpen} onClose={() => setUserDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>User Details</DialogTitle>
        <DialogContent>
          {selectedUser && (
            <Box>
              <Typography variant="h6" gutterBottom>
                {selectedUser.name || 'User'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Email: {selectedUser.email || 'user@example.com'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Plan: {selectedUser.plan || 'Free'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Status: {selectedUser.status || 'Active'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Last Active: {selectedUser.last_active || 'Today'}
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUserDialogOpen(false)}>Close</Button>
          <Button variant="contained" color="primary">
            Edit User
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AdminDashboard;