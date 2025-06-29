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
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Snackbar,
} from '@mui/material';
import {
  Help,
  Email,
  Phone,
  Chat,
  Send,
  ExpandMore,
  Article,
  VideoLibrary,
  School,
  BugReport,
  Lightbulb,
  QuestionAnswer,
} from '@mui/icons-material';
import useSupportStore from '../../store/supportStore';

const Support = () => {
  const { info, feedbackStatus, loading, error, fetchInfo, submitFeedback } = useSupportStore();
  const [openFeedbackDialog, setOpenFeedbackDialog] = useState(false);
  const [feedbackData, setFeedbackData] = useState({
    type: 'general',
    subject: '',
    message: '',
    priority: 'medium',
  });
  const [showSnackbar, setShowSnackbar] = useState(false);

  useEffect(() => {
    fetchInfo();
  }, [fetchInfo]);

  const handleSubmitFeedback = async () => {
    await submitFeedback(feedbackData);
    setOpenFeedbackDialog(false);
    setFeedbackData({
      type: 'general',
      subject: '',
      message: '',
      priority: 'medium',
    });
    setShowSnackbar(true);
  };

  const faqs = [
    {
      question: "How do I connect my social media accounts?",
      answer: "Go to the Platforms section and click 'Connect' next to the platform you want to connect. Follow the OAuth flow to authorize AutoGuru Universal to access your account."
    },
    {
      question: "How does the AI content creation work?",
      answer: "Our AI analyzes your business niche, target audience, and past performance to generate optimized content. You can customize the content before publishing or scheduling."
    },
    {
      question: "What analytics are available?",
      answer: "We provide comprehensive analytics including engagement rates, reach, impressions, conversion tracking, and AI-powered insights to help optimize your strategy."
    },
    {
      question: "How do I schedule content?",
      answer: "Create content in the Content section, then click the schedule button to set a specific date and time for publication across your connected platforms."
    },
    {
      question: "Is my data secure?",
      answer: "Yes, all data is encrypted and stored securely. We use industry-standard security practices and never share your information with third parties."
    },
    {
      question: "Can I export my data?",
      answer: "Yes, you can export your analytics data in various formats including CSV, JSON, Excel, and PDF from the Analytics section."
    }
  ];

  const resources = [
    {
      title: "Getting Started Guide",
      type: "guide",
      icon: <School />,
      description: "Complete guide to setting up and using AutoGuru Universal",
      url: "#"
    },
    {
      title: "API Documentation",
      type: "api",
      icon: <Article />,
      description: "Comprehensive API reference and examples",
      url: "#"
    },
    {
      title: "Video Tutorials",
      type: "video",
      icon: <VideoLibrary />,
      description: "Step-by-step video tutorials for all features",
      url: "#"
    },
    {
      title: "Best Practices",
      type: "guide",
      icon: <Lightbulb />,
      description: "Tips and best practices for social media success",
      url: "#"
    }
  ];

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
        Failed to load support information: {error.message}
      </Alert>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Support & Help Center
      </Typography>

      {/* Contact Information */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Email sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Email Support
              </Typography>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                {info?.email || 'support@autoguru.com'}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Response within 24 hours
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Chat sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Live Chat
              </Typography>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Available 9AM-6PM EST
              </Typography>
              <Button variant="contained" size="small">
                Start Chat
              </Button>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Phone sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Phone Support
              </Typography>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                {info?.phone || '+1 (555) 123-4567'}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Mon-Fri 9AM-6PM EST
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Help sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
              <Typography variant="h6" gutterBottom>
                Help Center
              </Typography>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Self-service resources
              </Typography>
              <Button variant="outlined" size="small">
                Browse Articles
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* FAQ Section */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Frequently Asked Questions
              </Typography>
              {faqs.map((faq, index) => (
                <Accordion key={index}>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Typography variant="subtitle1">{faq.question}</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Typography variant="body2" color="textSecondary">
                      {faq.answer}
                    </Typography>
                  </AccordionDetails>
                </Accordion>
              ))}
            </CardContent>
          </Card>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              <List>
                <ListItem button>
                  <ListItemIcon>
                    <BugReport />
                  </ListItemIcon>
                  <ListItemText primary="Report a Bug" />
                </ListItem>
                <ListItem button>
                  <ListItemIcon>
                    <Lightbulb />
                  </ListItemIcon>
                  <ListItemText primary="Feature Request" />
                </ListItem>
                <ListItem button>
                  <ListItemIcon>
                    <QuestionAnswer />
                  </ListItemIcon>
                  <ListItemText primary="Ask a Question" />
                </ListItem>
                <ListItem button onClick={() => setOpenFeedbackDialog(true)}>
                  <ListItemIcon>
                    <Send />
                  </ListItemIcon>
                  <ListItemText primary="Send Feedback" />
                </ListItem>
              </List>
            </CardContent>
          </Card>

          {/* System Status */}
          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Status
              </Typography>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <Chip label="All Systems Operational" color="success" size="small" />
              </Box>
              <Typography variant="body2" color="textSecondary">
                Last updated: {new Date().toLocaleString()}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Resources */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Helpful Resources
              </Typography>
              <Grid container spacing={2}>
                {resources.map((resource, index) => (
                  <Grid item xs={12} sm={6} md={3} key={index}>
                    <Card variant="outlined">
                      <CardContent>
                        <Box display="flex" alignItems="center" mb={1}>
                          {resource.icon}
                          <Typography variant="subtitle2" sx={{ ml: 1 }}>
                            {resource.title}
                          </Typography>
                        </Box>
                        <Typography variant="body2" color="textSecondary" gutterBottom>
                          {resource.description}
                        </Typography>
                        <Button size="small" variant="outlined">
                          View Resource
                        </Button>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Feedback Dialog */}
      <Dialog open={openFeedbackDialog} onClose={() => setOpenFeedbackDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>Send Feedback</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Feedback Type</InputLabel>
                <Select
                  value={feedbackData.type}
                  label="Feedback Type"
                  onChange={(e) => setFeedbackData({ ...feedbackData, type: e.target.value })}
                >
                  <MenuItem value="general">General Feedback</MenuItem>
                  <MenuItem value="bug">Bug Report</MenuItem>
                  <MenuItem value="feature">Feature Request</MenuItem>
                  <MenuItem value="improvement">Improvement Suggestion</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Priority</InputLabel>
                <Select
                  value={feedbackData.priority}
                  label="Priority"
                  onChange={(e) => setFeedbackData({ ...feedbackData, priority: e.target.value })}
                >
                  <MenuItem value="low">Low</MenuItem>
                  <MenuItem value="medium">Medium</MenuItem>
                  <MenuItem value="high">High</MenuItem>
                  <MenuItem value="critical">Critical</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Subject"
                value={feedbackData.subject}
                onChange={(e) => setFeedbackData({ ...feedbackData, subject: e.target.value })}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                multiline
                rows={4}
                label="Message"
                value={feedbackData.message}
                onChange={(e) => setFeedbackData({ ...feedbackData, message: e.target.value })}
                placeholder="Please describe your feedback in detail..."
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenFeedbackDialog(false)}>Cancel</Button>
          <Button 
            onClick={handleSubmitFeedback} 
            variant="contained"
            disabled={!feedbackData.subject || !feedbackData.message}
          >
            Send Feedback
          </Button>
        </DialogActions>
      </Dialog>

      {/* Success Snackbar */}
      <Snackbar
        open={showSnackbar}
        autoHideDuration={6000}
        onClose={() => setShowSnackbar(false)}
        message="Feedback submitted successfully!"
      />
    </Box>
  );
};

export default Support; 