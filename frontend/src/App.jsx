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

// Import the actual feature components
import Dashboard from './features/dashboard/Dashboard';
import Analytics from './features/analytics/Analytics';
import Content from './features/content/Content';
import Platforms from './features/platforms/Platforms';
import Tasks from './features/tasks/Tasks';
import Settings from './features/settings/Settings';
import Support from './features/support/Support';

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
    </Router>
  );
} 