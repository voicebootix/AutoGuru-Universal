import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Alert,
  CircularProgress,
  Tabs,
  Tab,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Switch,
  FormControlLabel,
  Tooltip,
  LinearProgress,
  Divider,
  Link,
  Badge
} from '@mui/material';
import {
  CloudSync,
  Add,
  Edit,
  Delete,
  Test,
  CheckCircle,
  Error,
  Warning,
  Help,
  Refresh,
  Settings,
  Key,
  Link as LinkIcon,
  Monitor,
  PlayArrow,
  Stop,
  ExpandMore,
  Visibility,
  VisibilityOff,
  Launch
} from '@mui/icons-material';

// Platform Icons
import {
  Facebook,
  Instagram,
  Twitter,
  LinkedIn,
  YouTube,
  Pinterest,
  Reddit,
  TikTok
} from '@mui/icons-material';

import { adminAPI } from '../services/adminAPI';

const platforms = [
  {
    id: 'facebook',
    name: 'Facebook',
    icon: <Facebook />,
    color: '#1877f2',
    description: 'Connect to Facebook pages and groups',
    credentials: [
      { name: 'app_id', label: 'App ID', type: 'text', required: true },
      { name: 'app_secret', label: 'App Secret', type: 'password', required: true },
      { name: 'access_token', label: 'Access Token', type: 'textarea', required: false }
    ],
    setupGuide: 'https://developers.facebook.com/docs/apps/register',
    permissions: ['pages_manage_posts', 'pages_read_engagement', 'pages_show_list']
  },
  {
    id: 'instagram',
    name: 'Instagram',
    icon: <Instagram />,
    color: '#E4405F',
    description: 'Connect to Instagram business accounts',
    credentials: [
      { name: 'app_id', label: 'App ID', type: 'text', required: true },
      { name: 'app_secret', label: 'App Secret', type: 'password', required: true },
      { name: 'access_token', label: 'Access Token', type: 'textarea', required: false }
    ],
    setupGuide: 'https://developers.facebook.com/docs/instagram-basic-display-api',
    permissions: ['instagram_basic', 'instagram_content_publish']
  },
  {
    id: 'twitter',
    name: 'Twitter',
    icon: <Twitter />,
    color: '#1DA1F2',
    description: 'Connect to Twitter accounts',
    credentials: [
      { name: 'api_key', label: 'API Key', type: 'text', required: true },
      { name: 'api_secret', label: 'API Secret', type: 'password', required: true },
      { name: 'access_token', label: 'Access Token', type: 'text', required: false },
      { name: 'access_token_secret', label: 'Access Token Secret', type: 'password', required: false }
    ],
    setupGuide: 'https://developer.twitter.com/en/docs/twitter-api/getting-started/guide',
    permissions: ['tweet.read', 'tweet.write', 'users.read']
  },
  {
    id: 'linkedin',
    name: 'LinkedIn',
    icon: <LinkedIn />,
    color: '#0077B5',
    description: 'Connect to LinkedIn company pages',
    credentials: [
      { name: 'client_id', label: 'Client ID', type: 'text', required: true },
      { name: 'client_secret', label: 'Client Secret', type: 'password', required: true },
      { name: 'access_token', label: 'Access Token', type: 'textarea', required: false }
    ],
    setupGuide: 'https://docs.microsoft.com/en-us/linkedin/shared/authentication/authorization-code-flow',
    permissions: ['r_liteprofile', 'r_organization_social', 'w_organization_social']
  },
  {
    id: 'youtube',
    name: 'YouTube',
    icon: <YouTube />,
    color: '#FF0000',
    description: 'Connect to YouTube channels',
    credentials: [
      { name: 'client_id', label: 'Client ID', type: 'text', required: true },
      { name: 'client_secret', label: 'Client Secret', type: 'password', required: true },
      { name: 'refresh_token', label: 'Refresh Token', type: 'textarea', required: false }
    ],
    setupGuide: 'https://developers.google.com/youtube/v3/getting-started',
    permissions: ['https://www.googleapis.com/auth/youtube.upload', 'https://www.googleapis.com/auth/youtube']
  },
  {
    id: 'tiktok',
    name: 'TikTok',
    icon: <TikTok />,
    color: '#000000',
    description: 'Connect to TikTok business accounts',
    credentials: [
      { name: 'app_id', label: 'App ID', type: 'text', required: true },
      { name: 'app_secret', label: 'App Secret', type: 'password', required: true },
      { name: 'access_token', label: 'Access Token', type: 'textarea', required: false }
    ],
    setupGuide: 'https://developers.tiktok.com/doc/login-kit-web',
    permissions: ['user.info.basic', 'video.list', 'video.upload']
  }
];

export default function PlatformManager() {
  const [platformStatuses, setPlatformStatuses] = useState({});
  const [selectedPlatform, setSelectedPlatform] = useState(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [testDialogOpen, setTestDialogOpen] = useState(false);
  const [credentials, setCredentials] = useState({});
  const [loading, setLoading] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResults, setTestResults] = useState(null);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [tabValue, setTabValue] = useState(0);
  const [showPasswords, setShowPasswords] = useState({});
  const [bulkConfigMode, setBulkConfigMode] = useState(false);
  const [bulkCredentials, setBulkCredentials] = useState({});

  useEffect(() => {
    loadPlatformStatuses();
  }, []);

  const loadPlatformStatuses = async () => {
    try {
      const statuses = await adminAPI.getPlatformStatuses();
      setPlatformStatuses(statuses);
    } catch (err) {
      setError('Failed to load platform statuses');
    }
  };

  const handleConfigurePlatform = (platform) => {
    setSelectedPlatform(platform);
    setCredentials({});
    setDialogOpen(true);
    setError('');
    setSuccess('');
  };

  const handleTestConnection = async (platform) => {
    setSelectedPlatform(platform);
    setTestDialogOpen(true);
    setTesting(true);
    setTestResults(null);

    try {
      const results = await adminAPI.testPlatformConnection(platform.id, {
        platform_type: platform.id,
        test_endpoints: [],
        timeout_seconds: 30
      });
      setTestResults(results);
    } catch (err) {
      setTestResults({
        success: false,
        error_message: err.message || 'Connection test failed',
        endpoint_results: []
      });
    } finally {
      setTesting(false);
    }
  };

  const handleSaveCredentials = async () => {
    if (!selectedPlatform) return;

    setLoading(true);
    setError('');

    try {
      const promises = [];
      
      // Save each credential
      for (const [credName, credValue] of Object.entries(credentials)) {
        if (credValue && credValue.trim()) {
          promises.push(
            adminAPI.storePlatformCredential(selectedPlatform.id, {
              platform_type: selectedPlatform.id,
              credential_name: credName,
              credential_value: credValue.trim(),
              metadata: {
                platform_name: selectedPlatform.name,
                configured_at: new Date().toISOString()
              }
            })
          );
        }
      }

      await Promise.all(promises);
      
      setSuccess(`${selectedPlatform.name} credentials saved successfully!`);
      setDialogOpen(false);
      
      // Refresh platform statuses
      await loadPlatformStatuses();
      
    } catch (err) {
      setError(err.message || 'Failed to save credentials');
    } finally {
      setLoading(false);
    }
  };

  const handleBulkConfigure = async () => {
    setLoading(true);
    setError('');

    try {
      const result = await adminAPI.bulkConfigurePlatforms(bulkCredentials);
      
      if (result.successful > 0) {
        setSuccess(`Successfully configured ${result.successful} platforms`);
      }
      
      if (result.failed > 0) {
        setError(`Failed to configure ${result.failed} platforms`);
      }
      
      setBulkConfigMode(false);
      setBulkCredentials({});
      await loadPlatformStatuses();
      
    } catch (err) {
      setError(err.message || 'Bulk configuration failed');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'connected': return '#4caf50';
      case 'configured': return '#2196f3';
      case 'error': return '#f44336';
      case 'not_configured': return '#9e9e9e';
      default: return '#ff9800';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'connected': return <CheckCircle />;
      case 'error': return <Error />;
      case 'not_configured': return <Warning />;
      default: return <Help />;
    }
  };

  const togglePasswordVisibility = (credName) => {
    setShowPasswords(prev => ({
      ...prev,
      [credName]: !prev[credName]
    }));
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1">
          Platform Manager
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={loadPlatformStatuses}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<Settings />}
            onClick={() => setBulkConfigMode(true)}
          >
            Bulk Configure
          </Button>
        </Box>
      </Box>

      {/* Success/Error Messages */}
      {success && (
        <Alert severity="success" sx={{ mb: 3 }} onClose={() => setSuccess('')}>
          {success}
        </Alert>
      )}
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      {/* Platform Overview */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Platform Overview
        </Typography>
        <Grid container spacing={2}>
          {Object.entries(platformStatuses).map(([platformId, status]) => (
            <Grid item xs={12} sm={6} md={3} key={platformId}>
              <Card
                sx={{
                  borderLeft: `4px solid ${getStatusColor(status.last_test_status)}`,
                  cursor: 'pointer',
                  '&:hover': { boxShadow: 3 }
                }}
                onClick={() => {
                  const platform = platforms.find(p => p.id === platformId);
                  if (platform) handleConfigurePlatform(platform);
                }}
              >
                <CardContent>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {getStatusIcon(status.last_test_status)}
                    <Typography variant="h6" sx={{ textTransform: 'capitalize' }}>
                      {platformId}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="textSecondary">
                    {status.configured ? 'Configured' : 'Not Configured'}
                  </Typography>
                  {status.response_time_ms && (
                    <Typography variant="caption" color="textSecondary">
                      Response: {status.response_time_ms}ms
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* Platform Configuration Cards */}
      <Grid container spacing={3}>
        {platforms.map((platform) => {
          const status = platformStatuses[platform.id] || {};
          const isConfigured = status.configured;
          const isConnected = status.last_test_status === 'connected';
          
          return (
            <Grid item xs={12} md={6} lg={4} key={platform.id}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  {/* Platform Header */}
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Box sx={{ color: platform.color, mr: 2 }}>
                      {platform.icon}
                    </Box>
                    <Box sx={{ flexGrow: 1 }}>
                      <Typography variant="h6">
                        {platform.name}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        {platform.description}
                      </Typography>
                    </Box>
                    <Badge
                      color={isConnected ? 'success' : isConfigured ? 'warning' : 'error'}
                      variant="dot"
                    />
                  </Box>

                  {/* Status Information */}
                  <Box sx={{ mb: 2 }}>
                    <Chip
                      icon={getStatusIcon(status.last_test_status)}
                      label={isConnected ? 'Connected' : isConfigured ? 'Configured' : 'Not Configured'}
                      color={isConnected ? 'success' : isConfigured ? 'warning' : 'default'}
                      size="small"
                    />
                    
                    {status.last_test_time && (
                      <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                        Last tested: {new Date(status.last_test_time).toLocaleString()}
                      </Typography>
                    )}
                    
                    {status.error_message && (
                      <Typography variant="caption" color="error" display="block" sx={{ mt: 1 }}>
                        {status.error_message}
                      </Typography>
                    )}
                  </Box>

                  {/* Required Credentials */}
                  <Accordion>
                    <AccordionSummary expandIcon={<ExpandMore />}>
                      <Typography variant="subtitle2">
                        Required Credentials ({platform.credentials.length})
                      </Typography>
                    </AccordionSummary>
                    <AccordionDetails>
                      <List dense>
                        {platform.credentials.map((cred, index) => (
                          <ListItem key={index} divider>
                            <ListItemIcon>
                              <Key fontSize="small" />
                            </ListItemIcon>
                            <ListItemText
                              primary={cred.label}
                              secondary={cred.required ? 'Required' : 'Optional'}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </AccordionDetails>
                  </Accordion>

                  {/* Action Buttons */}
                  <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                    <Button
                      variant="contained"
                      startIcon={isConfigured ? <Edit /> : <Add />}
                      onClick={() => handleConfigurePlatform(platform)}
                      fullWidth
                    >
                      {isConfigured ? 'Edit' : 'Configure'}
                    </Button>
                    
                    {isConfigured && (
                      <Button
                        variant="outlined"
                        startIcon={<Test />}
                        onClick={() => handleTestConnection(platform)}
                      >
                        Test
                      </Button>
                    )}
                    
                    <IconButton
                      component="a"
                      href={platform.setupGuide}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <Launch />
                    </IconButton>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {/* Platform Configuration Dialog */}
      <Dialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {selectedPlatform && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box sx={{ color: selectedPlatform.color }}>
                {selectedPlatform.icon}
              </Box>
              Configure {selectedPlatform.name}
            </Box>
          )}
        </DialogTitle>
        
        <DialogContent>
          {selectedPlatform && (
            <Box sx={{ mt: 2 }}>
              <Alert severity="info" sx={{ mb: 3 }}>
                <Typography variant="body2">
                  Follow the{' '}
                  <Link href={selectedPlatform.setupGuide} target="_blank" rel="noopener">
                    setup guide
                  </Link>{' '}
                  to create your app and get these credentials.
                </Typography>
              </Alert>

              <Grid container spacing={2}>
                {selectedPlatform.credentials.map((cred, index) => (
                  <Grid item xs={12} key={index}>
                    <TextField
                      fullWidth
                      label={cred.label}
                      type={cred.type === 'password' && !showPasswords[cred.name] ? 'password' : 'text'}
                      multiline={cred.type === 'textarea'}
                      rows={cred.type === 'textarea' ? 3 : 1}
                      required={cred.required}
                      value={credentials[cred.name] || ''}
                      onChange={(e) => setCredentials({
                        ...credentials,
                        [cred.name]: e.target.value
                      })}
                      InputProps={cred.type === 'password' ? {
                        endAdornment: (
                          <IconButton
                            onClick={() => togglePasswordVisibility(cred.name)}
                            edge="end"
                          >
                            {showPasswords[cred.name] ? <VisibilityOff /> : <Visibility />}
                          </IconButton>
                        )
                      } : {}}
                    />
                  </Grid>
                ))}
              </Grid>

              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Required Permissions:
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {selectedPlatform.permissions.map((permission, index) => (
                    <Chip
                      key={index}
                      label={permission}
                      size="small"
                      variant="outlined"
                    />
                  ))}
                </Box>
              </Box>
            </Box>
          )}
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleSaveCredentials}
            variant="contained"
            disabled={loading}
            startIcon={loading ? <CircularProgress size={20} /> : <CloudSync />}
          >
            {loading ? 'Saving...' : 'Save Credentials'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Connection Test Dialog */}
      <Dialog
        open={testDialogOpen}
        onClose={() => setTestDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Connection Test Results
        </DialogTitle>
        
        <DialogContent>
          {testing ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', py: 4 }}>
              <CircularProgress sx={{ mb: 2 }} />
              <Typography>Testing connection to {selectedPlatform?.name}...</Typography>
            </Box>
          ) : testResults ? (
            <Box sx={{ mt: 2 }}>
              <Alert severity={testResults.success ? 'success' : 'error'} sx={{ mb: 3 }}>
                {testResults.success ? 'Connection successful!' : 'Connection failed'}
                {testResults.response_time_ms && (
                  <Typography variant="body2">
                    Response time: {testResults.response_time_ms}ms
                  </Typography>
                )}
              </Alert>

              {testResults.error_message && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {testResults.error_message}
                </Alert>
              )}

              {testResults.endpoint_results && testResults.endpoint_results.length > 0 && (
                <Box>
                  <Typography variant="h6" gutterBottom>
                    Endpoint Test Results
                  </Typography>
                  <List>
                    {testResults.endpoint_results.map((result, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          {result.success ? <CheckCircle color="success" /> : <Error color="error" />}
                        </ListItemIcon>
                        <ListItemText
                          primary={result.endpoint}
                          secondary={result.success ? 
                            `HTTP ${result.status_code}` : 
                            result.error || 'Failed'
                          }
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
            </Box>
          ) : null}
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setTestDialogOpen(false)}>
            Close
          </Button>
        </DialogActions>
      </Dialog>

      {/* Bulk Configuration Dialog */}
      <Dialog
        open={bulkConfigMode}
        onClose={() => setBulkConfigMode(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          Bulk Platform Configuration
        </DialogTitle>
        
        <DialogContent>
          <Alert severity="info" sx={{ mb: 3 }}>
            Configure multiple platforms at once by providing their credentials.
          </Alert>
          
          <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)}>
            {platforms.map((platform, index) => (
              <Tab key={platform.id} label={platform.name} />
            ))}
          </Tabs>

          {platforms.map((platform, index) => (
            <TabPanel key={platform.id} value={tabValue} index={index}>
              <Grid container spacing={2}>
                {platform.credentials.map((cred, credIndex) => (
                  <Grid item xs={12} md={6} key={credIndex}>
                    <TextField
                      fullWidth
                      label={cred.label}
                      type={cred.type === 'password' ? 'password' : 'text'}
                      multiline={cred.type === 'textarea'}
                      rows={cred.type === 'textarea' ? 3 : 1}
                      value={bulkCredentials[platform.id]?.[cred.name] || ''}
                      onChange={(e) => setBulkCredentials({
                        ...bulkCredentials,
                        [platform.id]: {
                          ...bulkCredentials[platform.id],
                          [cred.name]: e.target.value
                        }
                      })}
                    />
                  </Grid>
                ))}
              </Grid>
            </TabPanel>
          ))}
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setBulkConfigMode(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleBulkConfigure}
            variant="contained"
            disabled={loading}
            startIcon={loading ? <CircularProgress size={20} /> : <CloudSync />}
          >
            {loading ? 'Configuring...' : 'Configure All'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

// Tab Panel Component
function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`platform-tabpanel-${index}`}
      aria-labelledby={`platform-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}