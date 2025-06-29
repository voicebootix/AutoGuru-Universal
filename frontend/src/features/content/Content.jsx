import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Alert,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Fab,
} from '@mui/material';
import {
  Add,
  Edit,
  Delete,
  Schedule,
  Publish,
  ContentCopy,
  Image,
  VideoLibrary,
  Article,
} from '@mui/icons-material';
import useContentStore from '../../store/contentStore';

const Content = () => {
  const { contentList, loading, error, createContent, scheduleContent, fetchContentHistory } = useContentStore();
  const [openCreateDialog, setOpenCreateDialog] = useState(false);
  const [openScheduleDialog, setOpenScheduleDialog] = useState(false);
  const [selectedContent, setSelectedContent] = useState(null);
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    platform: 'all',
    contentType: 'post',
    businessNiche: 'general',
  });

  useEffect(() => {
    fetchContentHistory();
  }, [fetchContentHistory]);

  const handleCreateContent = async () => {
    await createContent(formData);
    setOpenCreateDialog(false);
    setFormData({
      title: '',
      description: '',
      platform: 'all',
      contentType: 'post',
      businessNiche: 'general',
    });
  };

  const handleScheduleContent = async (contentId, scheduleData) => {
    await scheduleContent({ contentId, ...scheduleData });
    setOpenScheduleDialog(false);
  };

  const getContentIcon = (type) => {
    switch (type) {
      case 'image':
        return <Image />;
      case 'video':
        return <VideoLibrary />;
      case 'article':
        return <Article />;
      default:
        return <ContentCopy />;
    }
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
        Failed to load content: {error.message}
      </Alert>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Content Management</Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setOpenCreateDialog(true)}
        >
          Create Content
        </Button>
      </Box>

      {/* Content Stats */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Content
              </Typography>
              <Typography variant="h4">
                {contentList.length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Published
              </Typography>
              <Typography variant="h4">
                {contentList.filter(c => c.status === 'published').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Scheduled
              </Typography>
              <Typography variant="h4">
                {contentList.filter(c => c.status === 'scheduled').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Drafts
              </Typography>
              <Typography variant="h4">
                {contentList.filter(c => c.status === 'draft').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Content List */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Content History
          </Typography>
          {contentList.length > 0 ? (
            <List>
              {contentList.map((content) => (
                <ListItem key={content.id} divider>
                  <Box sx={{ mr: 2 }}>
                    {getContentIcon(content.contentType)}
                  </Box>
                  <ListItemText
                    primary={content.title}
                    secondary={
                      <Box>
                        <Typography variant="body2" color="textSecondary">
                          {content.description}
                        </Typography>
                        <Box sx={{ mt: 1 }}>
                          <Chip 
                            label={content.platform} 
                            size="small" 
                            sx={{ mr: 1 }}
                          />
                          <Chip 
                            label={content.status} 
                            size="small"
                            color={
                              content.status === 'published' ? 'success' :
                              content.status === 'scheduled' ? 'warning' : 'default'
                            }
                          />
                        </Box>
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton
                      edge="end"
                      onClick={() => {
                        setSelectedContent(content);
                        setOpenScheduleDialog(true);
                      }}
                      disabled={content.status === 'published'}
                    >
                      <Schedule />
                    </IconButton>
                    <IconButton edge="end">
                      <Edit />
                    </IconButton>
                    <IconButton edge="end">
                      <Delete />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          ) : (
            <Typography color="textSecondary" align="center" sx={{ py: 4 }}>
              No content created yet. Start by creating your first piece of content!
            </Typography>
          )}
        </CardContent>
      </Card>

      {/* Create Content Dialog */}
      <Dialog open={openCreateDialog} onClose={() => setOpenCreateDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Create New Content</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Content Title"
                value={formData.title}
                onChange={(e) => setFormData({ ...formData, title: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={4}
                label="Content Description"
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Platform</InputLabel>
                <Select
                  value={formData.platform}
                  label="Platform"
                  onChange={(e) => setFormData({ ...formData, platform: e.target.value })}
                >
                  <MenuItem value="all">All Platforms</MenuItem>
                  <MenuItem value="instagram">Instagram</MenuItem>
                  <MenuItem value="linkedin">LinkedIn</MenuItem>
                  <MenuItem value="tiktok">TikTok</MenuItem>
                  <MenuItem value="youtube">YouTube</MenuItem>
                  <MenuItem value="twitter">Twitter</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Content Type</InputLabel>
                <Select
                  value={formData.contentType}
                  label="Content Type"
                  onChange={(e) => setFormData({ ...formData, contentType: e.target.value })}
                >
                  <MenuItem value="post">Post</MenuItem>
                  <MenuItem value="image">Image</MenuItem>
                  <MenuItem value="video">Video</MenuItem>
                  <MenuItem value="article">Article</MenuItem>
                  <MenuItem value="story">Story</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Business Niche</InputLabel>
                <Select
                  value={formData.businessNiche}
                  label="Business Niche"
                  onChange={(e) => setFormData({ ...formData, businessNiche: e.target.value })}
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
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenCreateDialog(false)}>Cancel</Button>
          <Button onClick={handleCreateContent} variant="contained">
            Create Content
          </Button>
        </DialogActions>
      </Dialog>

      {/* Schedule Content Dialog */}
      <Dialog open={openScheduleDialog} onClose={() => setOpenScheduleDialog(false)}>
        <DialogTitle>Schedule Content</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            Schedule "{selectedContent?.title}" for publication
          </Typography>
          <TextField
            fullWidth
            type="datetime-local"
            label="Schedule Date & Time"
            InputLabelProps={{ shrink: true }}
            sx={{ mt: 1 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenScheduleDialog(false)}>Cancel</Button>
          <Button 
            onClick={() => handleScheduleContent(selectedContent?.id, { scheduledAt: new Date() })}
            variant="contained"
          >
            Schedule
          </Button>
        </DialogActions>
      </Dialog>

      {/* Floating Action Button */}
      <Fab
        color="primary"
        aria-label="add"
        sx={{ position: 'fixed', bottom: 16, right: 16 }}
        onClick={() => setOpenCreateDialog(true)}
      >
        <Add />
      </Fab>
    </Box>
  );
};

export default Content; 