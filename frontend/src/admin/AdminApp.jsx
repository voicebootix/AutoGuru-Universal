import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import {
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Box,
  Container,
  IconButton,
  Menu,
  MenuItem,
  Avatar,
  Divider,
  Paper,
  Alert,
  Snackbar
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Settings as SettingsIcon,
  People as PeopleIcon,
  Analytics as AnalyticsIcon,
  Security as SecurityIcon,
  CloudSync as CloudSyncIcon,
  AccountCircle,
  Logout,
  Notifications,
  Api as ApiIcon,
  MonitorHeart as MonitorIcon,
  Build as BuildIcon,
  Storage as StorageIcon
} from '@mui/icons-material';

// Admin Components
import AdminDashboard from './components/AdminDashboard';
import PlatformManager from './components/PlatformManager';
import APIKeyManager from './components/APIKeyManager';
import UserManager from './components/UserManager';
import SystemConfig from './components/SystemConfig';
import ConnectionTester from './components/ConnectionTester';
import RevenueAnalytics from './components/RevenueAnalytics';

// Admin Services
import { adminAPI } from './services/adminAPI';
import { useAdminAuth } from './hooks/useAdminAuth';

const drawerWidth = 260;

const adminNavItems = [
  { 
    text: 'Dashboard', 
    icon: <DashboardIcon />, 
    path: '/admin/dashboard',
    description: 'System overview and health metrics'
  },
  { 
    text: 'Platform Manager', 
    icon: <CloudSyncIcon />, 
    path: '/admin/platforms',
    description: 'Manage social media platform integrations'
  },
  { 
    text: 'API Keys', 
    icon: <ApiIcon />, 
    path: '/admin/api-keys',
    description: 'Manage AI service and platform API keys'
  },
  { 
    text: 'Connection Testing', 
    icon: <MonitorIcon />, 
    path: '/admin/connections',
    description: 'Test platform connections in real-time'
  },
  { 
    text: 'User Management', 
    icon: <PeopleIcon />, 
    path: '/admin/users',
    description: 'Manage user accounts and permissions'
  },
  { 
    text: 'System Config', 
    icon: <SettingsIcon />, 
    path: '/admin/config',
    description: 'Configure system settings and preferences'
  },
  { 
    text: 'Revenue Analytics', 
    icon: <AnalyticsIcon />, 
    path: '/admin/analytics',
    description: 'View revenue and usage analytics'
  },
  { 
    text: 'System Health', 
    icon: <SecurityIcon />, 
    path: '/admin/health',
    description: 'Monitor system health and performance'
  }
];

export default function AdminApp() {
  const { adminUser, isAuthenticated, logout } = useAdminAuth();
  const [anchorEl, setAnchorEl] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });
  const [systemHealth, setSystemHealth] = useState({ status: 'unknown', alerts: [] });

  useEffect(() => {
    if (isAuthenticated) {
      loadSystemHealth();
      loadNotifications();
      
      // Set up periodic health checks
      const healthInterval = setInterval(loadSystemHealth, 30000); // Every 30 seconds
      
      return () => clearInterval(healthInterval);
    }
  }, [isAuthenticated]);

  const loadSystemHealth = async () => {
    try {
      const health = await adminAPI.getSystemHealth();
      setSystemHealth(health);
      
      // Show critical alerts
      if (health.alerts && health.alerts.length > 0) {
        const criticalAlerts = health.alerts.filter(alert => alert.severity === 'critical');
        if (criticalAlerts.length > 0) {
          setSnackbar({
            open: true,
            message: `${criticalAlerts.length} critical system alerts require attention`,
            severity: 'error'
          });
        }
      }
    } catch (error) {
      console.error('Failed to load system health:', error);
    }
  };

  const loadNotifications = async () => {
    try {
      const notifs = await adminAPI.getNotifications();
      setNotifications(notifs);
    } catch (error) {
      console.error('Failed to load notifications:', error);
    }
  };

  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    logout();
    handleMenuClose();
  };

  const handleSnackbarClose = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  // Show login screen if not authenticated
  if (!isAuthenticated) {
    return <AdminLogin />;
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return '#4caf50';
      case 'warning': return '#ff9800';
      case 'critical': return '#f44336';
      default: return '#9e9e9e';
    }
  };

  return (
    <Router basename="/admin">
      <Box sx={{ display: 'flex', minHeight: '100vh' }}>
        <CssBaseline />
        
        {/* Admin App Bar */}
        <AppBar 
          position="fixed" 
          sx={{ 
            zIndex: 1201,
            background: 'linear-gradient(135deg, #1e3c72 0%, #2a5298 100%)'
          }}
        >
          <Toolbar>
            <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                AutoGuru Universal - Admin Portal
                <Box
                  sx={{
                    width: 12,
                    height: 12,
                    borderRadius: '50%',
                    backgroundColor: getStatusColor(systemHealth.status),
                    animation: systemHealth.status === 'critical' ? 'pulse 2s infinite' : 'none'
                  }}
                />
              </Box>
            </Typography>
            
            {/* Notifications */}
            <IconButton color="inherit" sx={{ mr: 1 }}>
              <Notifications />
              {notifications.length > 0 && (
                <Box
                  sx={{
                    position: 'absolute',
                    top: 8,
                    right: 8,
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    backgroundColor: '#f44336'
                  }}
                />
              )}
            </IconButton>
            
            {/* Admin Profile Menu */}
            <IconButton color="inherit" onClick={handleMenuOpen}>
              <Avatar sx={{ width: 32, height: 32, bgcolor: 'rgba(255,255,255,0.2)' }}>
                {adminUser?.username?.charAt(0).toUpperCase()}
              </Avatar>
            </IconButton>
            
            <Menu
              anchorEl={anchorEl}
              open={Boolean(anchorEl)}
              onClose={handleMenuClose}
              transformOrigin={{ horizontal: 'right', vertical: 'top' }}
              anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
            >
              <MenuItem disabled>
                <Box>
                  <Typography variant="subtitle2">{adminUser?.username}</Typography>
                  <Typography variant="caption" color="textSecondary">
                    {adminUser?.role}
                  </Typography>
                </Box>
              </MenuItem>
              <Divider />
              <MenuItem onClick={handleLogout}>
                <ListItemIcon>
                  <Logout fontSize="small" />
                </ListItemIcon>
                Logout
              </MenuItem>
            </Menu>
          </Toolbar>
        </AppBar>

        {/* Admin Sidebar */}
        <Drawer
          variant="permanent"
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            [`& .MuiDrawer-paper`]: { 
              width: drawerWidth, 
              boxSizing: 'border-box',
              backgroundColor: '#f8f9fa',
              borderRight: '1px solid #e0e0e0'
            },
          }}
        >
          <Toolbar />
          <Box sx={{ overflow: 'auto', p: 1 }}>
            <List>
              {adminNavItems.map((item) => (
                <ListItem 
                  button 
                  key={item.text} 
                  component="a"
                  href={item.path}
                  sx={{
                    borderRadius: 2,
                    mb: 0.5,
                    '&:hover': {
                      backgroundColor: 'rgba(25, 118, 210, 0.08)'
                    }
                  }}
                >
                  <ListItemIcon sx={{ color: '#1976d2' }}>
                    {item.icon}
                  </ListItemIcon>
                  <ListItemText 
                    primary={item.text}
                    secondary={item.description}
                    primaryTypographyProps={{ fontSize: '0.9rem', fontWeight: 500 }}
                    secondaryTypographyProps={{ fontSize: '0.75rem' }}
                  />
                </ListItem>
              ))}
            </List>
          </Box>

          {/* System Status Panel */}
          <Box sx={{ p: 2, mt: 'auto' }}>
            <Paper sx={{ p: 2, backgroundColor: '#e3f2fd' }}>
              <Typography variant="subtitle2" gutterBottom>
                System Status
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box
                  sx={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    backgroundColor: getStatusColor(systemHealth.status)
                  }}
                />
                <Typography variant="caption" sx={{ textTransform: 'capitalize' }}>
                  {systemHealth.status || 'Unknown'}
                </Typography>
              </Box>
              {systemHealth.alerts && systemHealth.alerts.length > 0 && (
                <Typography variant="caption" color="error" sx={{ display: 'block', mt: 1 }}>
                  {systemHealth.alerts.length} alerts
                </Typography>
              )}
            </Paper>
          </Box>
        </Drawer>

        {/* Main Content Area */}
        <Box component="main" sx={{ flexGrow: 1, bgcolor: '#f5f5f5', minHeight: '100vh' }}>
          <Toolbar />
          <Container maxWidth="xl" sx={{ py: 3 }}>
            <Routes>
              <Route path="/" element={<Navigate to="/admin/dashboard" replace />} />
              <Route path="/dashboard" element={<AdminDashboard />} />
              <Route path="/platforms" element={<PlatformManager />} />
              <Route path="/api-keys" element={<APIKeyManager />} />
              <Route path="/connections" element={<ConnectionTester />} />
              <Route path="/users" element={<UserManager />} />
              <Route path="/config" element={<SystemConfig />} />
              <Route path="/analytics" element={<RevenueAnalytics />} />
              <Route path="/health" element={<SystemHealthMonitor />} />
            </Routes>
          </Container>
        </Box>

        {/* Global Notifications */}
        <Snackbar
          open={snackbar.open}
          autoHideDuration={6000}
          onClose={handleSnackbarClose}
          anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
        >
          <Alert onClose={handleSnackbarClose} severity={snackbar.severity}>
            {snackbar.message}
          </Alert>
        </Snackbar>
      </Box>
    </Router>
  );
}

// Admin Login Component
function AdminLogin() {
  const [credentials, setCredentials] = useState({ username: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const { login } = useAdminAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      await login(credentials.username, credentials.password);
    } catch (err) {
      setError(err.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
      }}
    >
      <Paper
        elevation={24}
        sx={{
          p: 4,
          maxWidth: 400,
          width: '100%',
          borderRadius: 3
        }}
      >
        <Box sx={{ textAlign: 'center', mb: 3 }}>
          <Typography variant="h4" component="h1" gutterBottom>
            Admin Portal
          </Typography>
          <Typography variant="body2" color="textSecondary">
            AutoGuru Universal Administration
          </Typography>
        </Box>

        <form onSubmit={handleSubmit}>
          <Box sx={{ mb: 2 }}>
            <input
              type="text"
              placeholder="Username"
              value={credentials.username}
              onChange={(e) => setCredentials({ ...credentials, username: e.target.value })}
              style={{
                width: '100%',
                padding: '12px',
                fontSize: '16px',
                border: '1px solid #ddd',
                borderRadius: '8px',
                marginBottom: '12px'
              }}
              required
            />
          </Box>
          
          <Box sx={{ mb: 2 }}>
            <input
              type="password"
              placeholder="Password"
              value={credentials.password}
              onChange={(e) => setCredentials({ ...credentials, password: e.target.value })}
              style={{
                width: '100%',
                padding: '12px',
                fontSize: '16px',
                border: '1px solid #ddd',
                borderRadius: '8px'
              }}
              required
            />
          </Box>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <button
            type="submit"
            disabled={loading}
            style={{
              width: '100%',
              padding: '12px',
              fontSize: '16px',
              backgroundColor: '#1976d2',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              cursor: loading ? 'not-allowed' : 'pointer',
              opacity: loading ? 0.7 : 1
            }}
          >
            {loading ? 'Signing In...' : 'Sign In'}
          </button>
        </form>

        <Box sx={{ mt: 3, textAlign: 'center' }}>
          <Typography variant="caption" color="textSecondary">
            Default: admin / admin123
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
}

// Placeholder for System Health Monitor
function SystemHealthMonitor() {
  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        System Health Monitor
      </Typography>
      <Typography variant="body1">
        Comprehensive system health monitoring interface coming soon...
      </Typography>
    </Paper>
  );
}

// CSS for pulse animation
const style = document.createElement('style');
style.textContent = `
  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.3; }
    100% { opacity: 1; }
  }
`;
document.head.appendChild(style);