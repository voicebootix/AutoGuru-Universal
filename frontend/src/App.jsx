import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { CssBaseline, AppBar, Toolbar, Typography, Drawer, List, ListItem, ListItemIcon, ListItemText, Box, Container } from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import ContentPasteIcon from '@mui/icons-material/ContentPaste';
import SettingsIcon from '@mui/icons-material/Settings';
import SupportAgentIcon from '@mui/icons-material/SupportAgent';
import CloudQueueIcon from '@mui/icons-material/CloudQueue';
import LinkIcon from '@mui/icons-material/Link';

// Import actual components instead of using placeholders
import Dashboard from './features/dashboard/Dashboard';

const drawerWidth = 220;

const navItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
  { text: 'Analytics', icon: <AnalyticsIcon />, path: '/analytics' },
  { text: 'Content', icon: <ContentPasteIcon />, path: '/content' },
  { text: 'Platforms', icon: <LinkIcon />, path: '/platforms' },
  { text: 'Tasks', icon: <CloudQueueIcon />, path: '/tasks' },
  { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
  { text: 'Support', icon: <SupportAgentIcon />, path: '/support' },
];

// Temporary placeholder for components not yet implemented
function Placeholder({ title }) {
  return (
    <Box p={4}>
      <Typography variant="h4" gutterBottom>{title}</Typography>
      <Typography variant="body1" color="textSecondary">
        This feature is being implemented. Dashboard is now functional!
      </Typography>
    </Box>
  );
}

export default function App() {
  return (
    <Router>
      <Box sx={{ display: 'flex' }}>
        <CssBaseline />
        <AppBar position="fixed" sx={{ zIndex: 1201 }}>
          <Toolbar>
            <Typography variant="h6" noWrap component="div">
              AutoGuru Universal
            </Typography>
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
                <ListItem button key={item.text} component={Link} to={item.path}>
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
              <Route path="/analytics" element={<Placeholder title="Analytics Dashboard" />} />
              <Route path="/content" element={<Placeholder title="Content Creation & Scheduling" />} />
              <Route path="/platforms" element={<Placeholder title="Platform Management" />} />
              <Route path="/tasks" element={<Placeholder title="Background Tasks" />} />
              <Route path="/settings" element={<Placeholder title="Settings & Security" />} />
              <Route path="/support" element={<Placeholder title="Support & Documentation" />} />
            </Routes>
          </Container>
        </Box>
      </Box>
    </Router>
  );
} 