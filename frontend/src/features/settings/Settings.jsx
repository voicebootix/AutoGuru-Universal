import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  CircularProgress,
  Alert,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Switch,
  Divider,
} from '@mui/material';
import {
  Person,
  Security,
  Api,
  Refresh,
  Visibility,
  VisibilityOff,
  Edit,
  Save,
  Cancel,
} from '@mui/icons-material';
import useSettingsStore from '../../store/settingsStore';

const Settings = () => {
  const { 
    profile, 
    apiKey, 
    oauthTokens, 
    loading, 
    error, 
    fetchProfile, 
    updateProfile, 
    fetchApiKey, 
    regenerateApiKey, 
    fetchOAuthTokens 
  } = useSettingsStore();
  
  const [editMode, setEditMode] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);
  const [profileData, setProfileData] = useState({});
  const [openRegenerateDialog, setOpenRegenerateDialog] = useState(false);

  useEffect(() => {
    fetchProfile();
    fetchApiKey();
    fetchOAuthTokens();
  }, [fetchProfile, fetchApiKey, fetchOAuthTokens]);

  useEffect(() => {
    if (profile) {
      setProfileData(profile);
    }
  }, [profile]);

  const handleSaveProfile = async () => {
    await updateProfile(profileData);
    setEditMode(false);
  };

  const handleCancelEdit = () => {
    setProfileData(profile);
    setEditMode(false);
  };

  const handleRegenerateApiKey = async () => {
    await regenerateApiKey();
    setOpenRegenerateDialog(false);
    fetchApiKey(); // Refresh the API key
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
        Failed to load settings: {error.message}
      </Alert>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>

      <Grid container spacing={3}>
        {/* Profile Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Box display="flex" alignItems="center">
                  <Person sx={{ mr: 1 }} />
                  <Typography variant="h6">Profile Settings</Typography>
                </Box>
                {!editMode ? (
                  <Button
                    startIcon={<Edit />}
                    onClick={() => setEditMode(true)}
                  >
                    Edit
                  </Button>
                ) : (
                  <Box>
                    <Button
                      startIcon={<Save />}
                      onClick={handleSaveProfile}
                      sx={{ mr: 1 }}
                    >
                      Save
                    </Button>
                    <Button
                      startIcon={<Cancel />}
                      onClick={handleCancelEdit}
                    >
                      Cancel
                    </Button>
                  </Box>
                )}
              </Box>

              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="First Name"
                    value={profileData.firstName || ''}
                    onChange={(e) => setProfileData({ ...profileData, firstName: e.target.value })}
                    disabled={!editMode}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Last Name"
                    value={profileData.lastName || ''}
                    onChange={(e) => setProfileData({ ...profileData, lastName: e.target.value })}
                    disabled={!editMode}
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Email"
                    type="email"
                    value={profileData.email || ''}
                    onChange={(e) => setProfileData({ ...profileData, email: e.target.value })}
                    disabled={!editMode}
                  />
                </Grid>
                <Grid item xs={12}>
                  <FormControl fullWidth>
                    <InputLabel>Business Niche</InputLabel>
                    <Select
                      value={profileData.businessNiche || 'general'}
                      label="Business Niche"
                      onChange={(e) => setProfileData({ ...profileData, businessNiche: e.target.value })}
                      disabled={!editMode}
                    >
                      <MenuItem value="general">General</MenuItem>
                      <MenuItem value="fitness">Fitness & Wellness</MenuItem>
                      <MenuItem value="education">Education</MenuItem>
                      <MenuItem value="consulting">Business Consulting</MenuItem>
                      <MenuItem value="creative">Creative Arts</MenuItem>
                      <MenuItem value="ecommerce">E-commerce</MenuItem>
                      <MenuItem value="technology">Technology</MenuItem>
                      <MenuItem value="nonprofit">Non-profit</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Company"
                    value={profileData.company || ''}
                    onChange={(e) => setProfileData({ ...profileData, company: e.target.value })}
                    disabled={!editMode}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* API Key Management */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Box display="flex" alignItems="center">
                  <Api sx={{ mr: 1 }} />
                  <Typography variant="h6">API Key</Typography>
                </Box>
                <Button
                  startIcon={<Refresh />}
                  onClick={() => setOpenRegenerateDialog(true)}
                  color="warning"
                >
                  Regenerate
                </Button>
              </Box>

              <TextField
                fullWidth
                label="API Key"
                value={apiKey || 'No API key available'}
                type={showApiKey ? 'text' : 'password'}
                InputProps={{
                  readOnly: true,
                  endAdornment: (
                    <IconButton
                      onClick={() => setShowApiKey(!showApiKey)}
                      edge="end"
                    >
                      {showApiKey ? <VisibilityOff /> : <Visibility />}
                    </IconButton>
                  ),
                }}
                sx={{ mb: 2 }}
              />

              <Typography variant="body2" color="textSecondary">
                Use this API key to authenticate with the AutoGuru Universal API.
                Keep it secure and never share it publicly.
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* OAuth Tokens */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Security sx={{ mr: 1 }} />
                <Typography variant="h6">OAuth Tokens</Typography>
              </Box>

              {oauthTokens && Object.keys(oauthTokens).length > 0 ? (
                <List>
                  {Object.entries(oauthTokens).map(([platform, token]) => (
                    <ListItem key={platform} divider>
                      <ListItemText
                        primary={platform.charAt(0).toUpperCase() + platform.slice(1)}
                        secondary={
                          <Box>
                            <Typography variant="body2" color="textSecondary">
                              Token expires: {token.expiresAt ? new Date(token.expiresAt).toLocaleDateString() : 'Never'}
                            </Typography>
                            <Chip
                              label={token.isValid ? 'Valid' : 'Expired'}
                              color={token.isValid ? 'success' : 'error'}
                              size="small"
                              sx={{ mt: 1 }}
                            />
                          </Box>
                        }
                      />
                      <ListItemSecondaryAction>
                        <Button
                          size="small"
                          startIcon={<Refresh />}
                          disabled={token.isValid}
                        >
                          Refresh
                        </Button>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Typography color="textSecondary" align="center" sx={{ py: 4 }}>
                  No OAuth tokens found. Connect your social media platforms to see tokens here.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Notification Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Notification Settings
              </Typography>
              
              <List>
                <ListItem>
                  <ListItemText
                    primary="Email Notifications"
                    secondary="Receive email updates about your content performance"
                  />
                  <ListItemSecondaryAction>
                    <Switch defaultChecked />
                  </ListItemSecondaryAction>
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText
                    primary="Task Completion Alerts"
                    secondary="Get notified when background tasks complete"
                  />
                  <ListItemSecondaryAction>
                    <Switch defaultChecked />
                  </ListItemSecondaryAction>
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText
                    primary="Weekly Reports"
                    secondary="Receive weekly performance summaries"
                  />
                  <ListItemSecondaryAction>
                    <Switch />
                  </ListItemSecondaryAction>
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText
                    primary="Error Alerts"
                    secondary="Get notified of system errors or failures"
                  />
                  <ListItemSecondaryAction>
                    <Switch defaultChecked />
                  </ListItemSecondaryAction>
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* System Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Settings
              </Typography>
              
              <List>
                <ListItem>
                  <ListItemText
                    primary="Auto-optimization"
                    secondary="Automatically optimize content based on performance"
                  />
                  <ListItemSecondaryAction>
                    <Switch defaultChecked />
                  </ListItemSecondaryAction>
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText
                    primary="Real-time Analytics"
                    secondary="Enable real-time analytics updates"
                  />
                  <ListItemSecondaryAction>
                    <Switch defaultChecked />
                  </ListItemSecondaryAction>
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText
                    primary="Content Scheduling"
                    secondary="Allow automatic content scheduling"
                  />
                  <ListItemSecondaryAction>
                    <Switch defaultChecked />
                  </ListItemSecondaryAction>
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText
                    primary="Data Export"
                    secondary="Allow data export functionality"
                  />
                  <ListItemSecondaryAction>
                    <Switch />
                  </ListItemSecondaryAction>
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Regenerate API Key Dialog */}
      <Dialog open={openRegenerateDialog} onClose={() => setOpenRegenerateDialog(false)}>
        <DialogTitle>Regenerate API Key</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="textSecondary">
            Are you sure you want to regenerate your API key? This will invalidate your current key and you'll need to update any applications using it.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenRegenerateDialog(false)}>Cancel</Button>
          <Button onClick={handleRegenerateApiKey} color="warning" variant="contained">
            Regenerate
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Settings; 