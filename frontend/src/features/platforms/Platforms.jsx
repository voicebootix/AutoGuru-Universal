import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  CircularProgress,
  Alert,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  Switch,
  LinearProgress,
} from '@mui/material';
import {
  Instagram,
  LinkedIn,
  YouTube,
  Twitter,
  Facebook,
  CheckCircle,
  Error,
  Refresh,
  Link,
  LinkOff,
} from '@mui/icons-material';
import usePlatformStore from '../../store/platformStore';

const Platforms = () => {
  const { status, loading, error, fetchStatus, connect, refreshToken } = usePlatformStore();
  const [openConnectDialog, setOpenConnectDialog] = useState(false);
  const [selectedPlatform, setSelectedPlatform] = useState(null);
  const [oauthData, setOauthData] = useState({
    clientId: '',
    clientSecret: '',
    accessToken: '',
  });

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const handleConnect = async () => {
    await connect(selectedPlatform, oauthData);
    setOpenConnectDialog(false);
    setOauthData({ clientId: '', clientSecret: '', accessToken: '' });
  };

  const handleRefreshToken = async (platform) => {
    await refreshToken(platform);
  };

  const platforms = [
    {
      name: 'Instagram',
      icon: <Instagram />,
      key: 'instagram',
      color: '#E4405F',
      description: 'Share photos and stories with your audience',
    },
    {
      name: 'LinkedIn',
      icon: <LinkedIn />,
      key: 'linkedin',
      color: '#0077B5',
      description: 'Professional networking and business content',
    },
    {
      name: 'YouTube',
      icon: <YouTube />,
      key: 'youtube',
      color: '#FF0000',
      description: 'Video content and channel management',
    },
    {
      name: 'Twitter',
      icon: <Twitter />,
      key: 'twitter',
      color: '#1DA1F2',
      description: 'Real-time updates and engagement',
    },
    {
      name: 'TikTok',
      icon: <YouTube />, // Using YouTube icon as placeholder
      key: 'tiktok',
      color: '#000000',
      description: 'Short-form video content',
    },
    {
      name: 'Facebook',
      icon: <Facebook />,
      key: 'facebook',
      color: '#1877F2',
      description: 'Social networking and page management',
    },
  ];

  const getPlatformStatus = (platformKey) => {
    const platformStatus = status[platformKey];
    if (!platformStatus) return { connected: false, status: 'disconnected' };
    return platformStatus;
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
        Failed to load platform status: {error.message}
      </Alert>
    );
  }

  const connectedCount = platforms.filter(p => getPlatformStatus(p.key).connected).length;

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Platform Connections
      </Typography>

      {/* Platform Stats */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Connected Platforms
              </Typography>
              <Typography variant="h4">
                {connectedCount}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                of {platforms.length} platforms
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Connection Rate
              </Typography>
              <Typography variant="h4">
                {Math.round((connectedCount / platforms.length) * 100)}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={(connectedCount / platforms.length) * 100} 
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Active Tokens
              </Typography>
              <Typography variant="h4">
                {platforms.filter(p => getPlatformStatus(p.key).tokenValid).length}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                valid tokens
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Last Sync
              </Typography>
              <Typography variant="h6">
                {status.lastSync ? new Date(status.lastSync).toLocaleDateString() : 'Never'}
              </Typography>
              <Button
                size="small"
                startIcon={<Refresh />}
                onClick={() => fetchStatus()}
              >
                Refresh
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Platform List */}
      <Grid container spacing={3}>
        {platforms.map((platform) => {
          const platformStatus = getPlatformStatus(platform.key);
          const isConnected = platformStatus.connected;
          const isTokenValid = platformStatus.tokenValid;

          return (
            <Grid item xs={12} md={6} key={platform.key}>
              <Card>
                <CardContent>
                  <Box display="flex" alignItems="center" justifyContent="space-between">
                    <Box display="flex" alignItems="center">
                      <Box
                        sx={{
                          color: platform.color,
                          mr: 2,
                          display: 'flex',
                          alignItems: 'center',
                        }}
                      >
                        {platform.icon}
                      </Box>
                      <Box>
                        <Typography variant="h6">{platform.name}</Typography>
                        <Typography variant="body2" color="textSecondary">
                          {platform.description}
                        </Typography>
                      </Box>
                    </Box>
                    <Box display="flex" alignItems="center" gap={1}>
                      {isConnected ? (
                        <>
                          <CheckCircle color="success" />
                          <Chip 
                            label="Connected" 
                            color="success" 
                            size="small"
                          />
                        </>
                      ) : (
                        <>
                          <Error color="error" />
                          <Chip 
                            label="Disconnected" 
                            color="error" 
                            size="small"
                          />
                        </>
                      )}
                    </Box>
                  </Box>

                  {isConnected && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="body2" color="textSecondary" gutterBottom>
                        Connection Details:
                      </Typography>
                      <List dense>
                        <ListItem>
                          <ListItemText
                            primary="Token Status"
                            secondary={
                              <Chip
                                label={isTokenValid ? 'Valid' : 'Expired'}
                                color={isTokenValid ? 'success' : 'error'}
                                size="small"
                              />
                            }
                          />
                          {!isTokenValid && (
                            <ListItemSecondaryAction>
                              <Button
                                size="small"
                                startIcon={<Refresh />}
                                onClick={() => handleRefreshToken(platform.key)}
                              >
                                Refresh Token
                              </Button>
                            </ListItemSecondaryAction>
                          )}
                        </ListItem>
                        {platformStatus.accountName && (
                          <ListItem>
                            <ListItemText
                              primary="Account"
                              secondary={platformStatus.accountName}
                            />
                          </ListItem>
                        )}
                        {platformStatus.lastPost && (
                          <ListItem>
                            <ListItemText
                              primary="Last Post"
                              secondary={new Date(platformStatus.lastPost).toLocaleDateString()}
                            />
                          </ListItem>
                        )}
                      </List>
                    </Box>
                  )}

                  <Box sx={{ mt: 2 }}>
                    {isConnected ? (
                      <Button
                        variant="outlined"
                        color="error"
                        startIcon={<LinkOff />}
                        onClick={() => {
                          setSelectedPlatform(platform);
                          setOpenConnectDialog(true);
                        }}
                      >
                        Reconnect
                      </Button>
                    ) : (
                      <Button
                        variant="contained"
                        startIcon={<Link />}
                        onClick={() => {
                          setSelectedPlatform(platform);
                          setOpenConnectDialog(true);
                        }}
                        sx={{ backgroundColor: platform.color }}
                      >
                        Connect
                      </Button>
                    )}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {/* Connect Platform Dialog */}
      <Dialog open={openConnectDialog} onClose={() => setOpenConnectDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>
          Connect {selectedPlatform?.name}
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
            Enter your {selectedPlatform?.name} API credentials to connect your account.
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Client ID"
                value={oauthData.clientId}
                onChange={(e) => setOauthData({ ...oauthData, clientId: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Client Secret"
                type="password"
                value={oauthData.clientSecret}
                onChange={(e) => setOauthData({ ...oauthData, clientSecret: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Access Token (Optional)"
                value={oauthData.accessToken}
                onChange={(e) => setOauthData({ ...oauthData, accessToken: e.target.value })}
                helperText="Leave empty to use OAuth flow"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenConnectDialog(false)}>Cancel</Button>
          <Button 
            onClick={handleConnect} 
            variant="contained"
            disabled={!oauthData.clientId || !oauthData.clientSecret}
          >
            Connect
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Platforms; 