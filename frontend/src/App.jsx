import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { CssBaseline, Box, Typography } from '@mui/material';

// Marketing Components
import MarketingLayout from './features/marketing/MarketingLayout';
import Landing from './features/marketing/Landing';
import Pricing from './features/marketing/Pricing';
import Signup from './features/marketing/Signup';

// App Components (existing)
import { AppBar, Toolbar, Drawer, List, ListItem, ListItemIcon, ListItemText, Container, Button } from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import ContentPasteIcon from '@mui/icons-material/ContentPaste';
import SettingsIcon from '@mui/icons-material/Settings';
import SupportAgentIcon from '@mui/icons-material/SupportAgent';
import CloudQueueIcon from '@mui/icons-material/CloudQueue';
import LinkIcon from '@mui/icons-material/Link';
import LogoutIcon from '@mui/icons-material/Logout';

// Import the actual feature components
import Dashboard from './features/dashboard/Dashboard';
import Analytics from './features/analytics/Analytics';
import Content from './features/content/Content';
import Platforms from './features/platforms/Platforms';
import Tasks from './features/tasks/Tasks';
import Settings from './features/settings/Settings';
import Support from './features/support/Support';
import Login from './features/auth/Login';

// Import auth utilities
import { getAuthToken, removeAuthToken } from './services/api';

const drawerWidth = 220;

const navItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/app' },
  { text: 'Analytics', icon: <AnalyticsIcon />, path: '/app/analytics' },
  { text: 'Content', icon: <ContentPasteIcon />, path: '/app/content' },
  { text: 'Platforms', icon: <LinkIcon />, path: '/app/platforms' },
  { text: 'Tasks', icon: <CloudQueueIcon />, path: '/app/tasks' },
  { text: 'Settings', icon: <SettingsIcon />, path: '/app/settings' },
  { text: 'Support', icon: <SupportAgentIcon />, path: '/app/support' },
];

// Protected Route Component for App Routes
const ProtectedRoute = ({ children }) => {
  const token = getAuthToken();
  return token ? children : <Navigate to="/login" replace />;
};

// App Layout Component (for authenticated users)
const AppLayout = ({ onLogout }) => {
  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <AppBar position="fixed" sx={{ zIndex: 1201 }}>
        <Toolbar sx={{ justifyContent: 'space-between' }}>
          <Typography variant="h6" noWrap component="div">
            AutoGuru Universal
          </Typography>
          <Button 
            color="inherit" 
            startIcon={<LogoutIcon />}
            onClick={onLogout}
          >
            Logout
          </Button>
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
        <Box sx={{ overflow: 'auto' }}>
          <List>
            {navItems.map((item) => (
              <ListItem button key={item.text} component="a" href={item.path}>
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>
      <Box component="main" sx={{ flexGrow: 1, bgcolor: 'background.default', p: 3, ml: `${drawerWidth}px` }}>
        <Toolbar />
        <Container maxWidth="xl">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/content" element={<Content />} />
            <Route path="/platforms" element={<Platforms />} />
            <Route path="/tasks" element={<Tasks />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/support" element={<Support />} />
          </Routes>
        </Container>
      </Box>
    </Box>
  );
};

// Placeholder components for marketing pages
const Features = () => (
  <Box sx={{ py: 8 }}>
    <Container maxWidth="lg">
      <Typography variant="h2" component="h1" gutterBottom fontWeight="bold" textAlign="center">
        Features
      </Typography>
      <Typography variant="h6" color="text.secondary" paragraph textAlign="center">
        Coming soon - Detailed features page
      </Typography>
    </Container>
  </Box>
);

const UseCases = () => (
  <Box sx={{ py: 8 }}>
    <Container maxWidth="lg">
      <Typography variant="h2" component="h1" gutterBottom fontWeight="bold" textAlign="center">
        Use Cases
      </Typography>
      <Typography variant="h6" color="text.secondary" paragraph textAlign="center">
        Coming soon - Industry-specific use cases
      </Typography>
    </Container>
  </Box>
);

const Resources = () => (
  <Box sx={{ py: 8 }}>
    <Container maxWidth="lg">
      <Typography variant="h2" component="h1" gutterBottom fontWeight="bold" textAlign="center">
        Resources
      </Typography>
      <Typography variant="h6" color="text.secondary" paragraph textAlign="center">
        Coming soon - Blog, tutorials, and resources
      </Typography>
    </Container>
  </Box>
);

const About = () => (
  <Box sx={{ py: 8 }}>
    <Container maxWidth="lg">
      <Typography variant="h2" component="h1" gutterBottom fontWeight="bold" textAlign="center">
        About AutoGuru Universal
      </Typography>
      <Typography variant="h6" color="text.secondary" paragraph textAlign="center">
        Coming soon - Our story and mission
      </Typography>
    </Container>
  </Box>
);

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const token = getAuthToken();
    setIsAuthenticated(!!token);
    setIsLoading(false);
  }, []);

  const handleLogout = () => {
    removeAuthToken();
    setIsAuthenticated(false);
  };

  const handleLogin = () => {
    setIsAuthenticated(true);
  };

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <Typography>Loading...</Typography>
      </Box>
    );
  }

  return (
    <Router>
      <CssBaseline />
      <Routes>
        {/* Marketing Routes (Public) */}
        <Route path="/" element={<MarketingLayout />}>
          <Route index element={<Landing />} />
          <Route path="features" element={<Features />} />
          <Route path="pricing" element={<Pricing />} />
          <Route path="use-cases" element={<UseCases />} />
          <Route path="resources" element={<Resources />} />
          <Route path="about" element={<About />} />
        </Route>

        {/* Authentication Routes */}
        <Route 
          path="/login" 
          element={
            isAuthenticated ? 
              <Navigate to="/app" replace /> : 
              <Login onLogin={handleLogin} />
          } 
        />
        <Route path="/signup" element={<Signup />} />

        {/* Protected App Routes */}
        <Route 
          path="/app/*" 
          element={
            <ProtectedRoute>
              <AppLayout onLogout={handleLogout} />
            </ProtectedRoute>
          } 
        />

        {/* Fallback Route */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
} 