import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, Navigate, useLocation } from 'react-router-dom';
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
  Button,
  Divider,
  Badge,
  IconButton,
  Menu,
  MenuItem,
  Chip
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Analytics as AnalyticsIcon,
  ContentPaste as ContentPasteIcon,
  Settings as SettingsIcon,
  SupportAgent as SupportAgentIcon,
  CloudQueue as CloudQueueIcon,
  Link as LinkIcon,
  Logout as LogoutIcon,
  Campaign as CampaignIcon,
  AdminPanelSettings as AdminIcon,
  AttachMoney as MoneyIcon,
  Psychology as PsychologyIcon,
  TrendingUp as TrendingUpIcon,
  Notifications as NotificationsIcon,
  AccountCircle as AccountCircleIcon,
  AutoAwesome as AutoAwesomeIcon
} from '@mui/icons-material';

// Import the actual feature components
import Dashboard from './features/dashboard/Dashboard';
import Analytics from './features/analytics/Analytics';
import Content from './features/content/Content';
import Platforms from './features/platforms/Platforms';
import Tasks from './features/tasks/Tasks';
import Settings from './features/settings/Settings';
import Support from './features/support/Support';
import Login from './features/auth/Login';

// Import new components
import AdvertisingCreative from './features/advertising/AdvertisingCreative';
import AdminDashboard from './features/admin/AdminDashboard';
import LandingPage from './pages/LandingPage';

// Import auth utilities
import { getAuthToken, removeAuthToken } from './services/api';

const drawerWidth = 240;

const navItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/', category: 'main' },
  { text: 'Analytics', icon: <AnalyticsIcon />, path: '/analytics', category: 'main' },
  { text: 'Content', icon: <ContentPasteIcon />, path: '/content', category: 'main' },
  { text: 'Platforms', icon: <LinkIcon />, path: '/platforms', category: 'main' },
  { text: 'Tasks', icon: <CloudQueueIcon />, path: '/tasks', category: 'main' },
  { text: 'Ad Creative Engine', icon: <CampaignIcon />, path: '/advertising', category: 'revenue', badge: 'New' },
  { text: 'Revenue Tracking', icon: <MoneyIcon />, path: '/revenue', category: 'revenue' },
  { text: 'AI Insights', icon: <PsychologyIcon />, path: '/insights', category: 'ai' },
  { text: 'Performance', icon: <TrendingUpIcon />, path: '/performance', category: 'ai' },
  { text: 'Admin Tools', icon: <AdminIcon />, path: '/admin', category: 'admin', badge: 'Pro' },
  { text: 'Settings', icon: <SettingsIcon />, path: '/settings', category: 'settings' },
  { text: 'Support', icon: <SupportAgentIcon />, path: '/support', category: 'settings' },
];

// Protected Route Component
const ProtectedRoute = ({ children }) => {
  const token = getAuthToken();
  return token ? children : <Navigate to="/login" replace />;
};

// Public Route Component (for landing page)
const PublicRoute = ({ children }) => {
  return children;
};

// Navigation Categories
const categories = {
  main: { title: 'Main', color: 'primary' },
  revenue: { title: 'Revenue & Advertising', color: 'success' },
  ai: { title: 'AI & Analytics', color: 'info' },
  admin: { title: 'Administration', color: 'warning' },
  settings: { title: 'Settings', color: 'default' }
};

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [anchorEl, setAnchorEl] = useState(null);
  const [notificationCount, setNotificationCount] = useState(3);

  useEffect(() => {
    const token = getAuthToken();
    setIsAuthenticated(!!token);
    setIsLoading(false);
  }, []);

  const handleLogout = () => {
    removeAuthToken();
    setIsAuthenticated(false);
  };

  const handleProfileMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <Typography>Loading AutoGuru Universal...</Typography>
      </Box>
    );
  }

  return (
    <Router>
      <CssBaseline />
      <Routes>
        {/* Public routes */}
        <Route path="/landing" element={<PublicRoute><LandingPage /></PublicRoute>} />
        <Route path="/login" element={<Login onLogin={() => setIsAuthenticated(true)} />} />
        
        {/* Protected routes */}
        <Route path="/*" element={
          isAuthenticated ? (
            <AuthenticatedApp 
              handleLogout={handleLogout} 
              notificationCount={notificationCount}
              anchorEl={anchorEl}
              handleProfileMenuOpen={handleProfileMenuOpen}
              handleMenuClose={handleMenuClose}
            />
          ) : (
            <Navigate to="/login" replace />
          )
        } />
      </Routes>
    </Router>
  );
}

// Separate component for authenticated app layout
const AuthenticatedApp = ({ handleLogout, notificationCount, anchorEl, handleProfileMenuOpen, handleMenuClose }) => {
  const location = useLocation();
  
  const getPageTitle = (path) => {
    const item = navItems.find(item => item.path === path);
    return item ? item.text : 'AutoGuru Universal';
  };

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar position="fixed" sx={{ zIndex: 1201 }}>
        <Toolbar sx={{ justifyContent: 'space-between' }}>
          <Box display="flex" alignItems="center">
            <AutoAwesomeIcon sx={{ mr: 1 }} />
            <Typography variant="h6" noWrap component="div">
              AutoGuru Universal
            </Typography>
            <Chip 
              label="Universal Platform" 
              size="small" 
              sx={{ ml: 2, backgroundColor: 'rgba(255,255,255,0.2)', color: 'white' }}
            />
          </Box>
          
          <Box display="flex" alignItems="center" gap={1}>
            <Typography variant="body2" sx={{ display: { xs: 'none', md: 'block' } }}>
              {getPageTitle(location.pathname)}
            </Typography>
            
            <IconButton color="inherit">
              <Badge badgeContent={notificationCount} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>
            
            <IconButton
              color="inherit"
              onClick={handleProfileMenuOpen}
              sx={{ ml: 1 }}
            >
              <AccountCircleIcon />
            </IconButton>
            
            <Menu
              anchorEl={anchorEl}
              open={Boolean(anchorEl)}
              onClose={handleMenuClose}
            >
              <MenuItem onClick={handleMenuClose}>Profile</MenuItem>
              <MenuItem onClick={handleMenuClose}>Account Settings</MenuItem>
              <MenuItem onClick={handleMenuClose}>Billing</MenuItem>
              <MenuItem onClick={() => { handleMenuClose(); handleLogout(); }}>
                <LogoutIcon sx={{ mr: 1 }} />
                Logout
              </MenuItem>
            </Menu>
          </Box>
        </Toolbar>
      </AppBar>
      
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: 'border-box' },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: 'auto', p: 1 }}>
          {Object.entries(categories).map(([categoryKey, category]) => {
            const categoryItems = navItems.filter(item => item.category === categoryKey);
            if (categoryItems.length === 0) return null;
            
            return (
              <Box key={categoryKey} sx={{ mb: 2 }}>
                <Typography 
                  variant="overline" 
                  sx={{ 
                    px: 2, 
                    py: 1, 
                    display: 'block',
                    fontWeight: 'bold',
                    color: `${category.color}.main`
                  }}
                >
                  {category.title}
                </Typography>
                <List dense>
                  {categoryItems.map((item) => (
                    <ListItem 
                      button 
                      key={item.text} 
                      component={Link} 
                      to={item.path}
                      sx={{
                        borderRadius: 1,
                        mb: 0.5,
                        mx: 1,
                        ...(location.pathname === item.path && {
                          backgroundColor: 'primary.light',
                          color: 'primary.contrastText'
                        })
                      }}
                    >
                      <ListItemIcon 
                        sx={{ 
                          color: location.pathname === item.path ? 'primary.contrastText' : 'inherit' 
                        }}
                      >
                        {item.icon}
                      </ListItemIcon>
                      <ListItemText 
                        primary={item.text} 
                        primaryTypographyProps={{ fontSize: '0.875rem' }}
                      />
                      {item.badge && (
                        <Chip 
                          label={item.badge} 
                          size="small" 
                          color={item.badge === 'New' ? 'success' : 'warning'}
                          sx={{ ml: 1 }}
                        />
                      )}
                    </ListItem>
                  ))}
                </List>
                {categoryKey !== 'settings' && <Divider sx={{ my: 1 }} />}
              </Box>
            );
          })}
        </Box>
      </Drawer>
      
      <Box component="main" sx={{ flexGrow: 1, bgcolor: 'background.default', p: 3, ml: `${drawerWidth}px` }}>
        <Toolbar />
        <Container maxWidth="xl">
          <Routes>
            <Route path="/" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
            <Route path="/analytics" element={<ProtectedRoute><Analytics /></ProtectedRoute>} />
            <Route path="/content" element={<ProtectedRoute><Content /></ProtectedRoute>} />
            <Route path="/platforms" element={<ProtectedRoute><Platforms /></ProtectedRoute>} />
            <Route path="/tasks" element={<ProtectedRoute><Tasks /></ProtectedRoute>} />
            <Route path="/advertising" element={<ProtectedRoute><AdvertisingCreative /></ProtectedRoute>} />
            <Route path="/revenue" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
            <Route path="/insights" element={<ProtectedRoute><Analytics /></ProtectedRoute>} />
            <Route path="/performance" element={<ProtectedRoute><Analytics /></ProtectedRoute>} />
            <Route path="/admin" element={<ProtectedRoute><AdminDashboard /></ProtectedRoute>} />
            <Route path="/settings" element={<ProtectedRoute><Settings /></ProtectedRoute>} />
            <Route path="/support" element={<ProtectedRoute><Support /></ProtectedRoute>} />
            <Route path="/login" element={<Navigate to="/" replace />} />
          </Routes>
        </Container>
      </Box>
    </Box>
  );
}; 