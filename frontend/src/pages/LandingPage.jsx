import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Container,
  Grid,
  Card,
  CardContent,
  Paper,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Avatar,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  AutoAwesome,
  TrendingUp,
  AttachMoney,
  Campaign,
  Analytics,
  Psychology,
  Speed,
  Security,
  CheckCircle,
  Star,
  People,
  Business,
  School,
  FitnessCenter,
  Palette,
  Store,
  Handyman,
  Computer,
  VolunteerActivism,
  ExpandMore,
  PlayArrow,
  Rocket,
  Timeline,
  Shield,
  CloudUpload,
  SmartToy,
  Group,
  MonetizationOn,
  Insights,
  CampaignOutlined,
  BarChart,
  AdminPanelSettings,
} from '@mui/icons-material';
import { Link, useNavigate } from 'react-router-dom';

const LandingPage = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const navigate = useNavigate();
  const [demoDialogOpen, setDemoDialogOpen] = useState(false);

  const businessNiches = [
    { name: 'Educational Business', icon: <School />, color: 'primary' },
    { name: 'Business Consulting', icon: <Business />, color: 'secondary' },
    { name: 'Fitness & Wellness', icon: <FitnessCenter />, color: 'success' },
    { name: 'Creative Professional', icon: <Palette />, color: 'warning' },
    { name: 'E-commerce', icon: <Store />, color: 'info' },
    { name: 'Local Services', icon: <Handyman />, color: 'error' },
    { name: 'Technology/SaaS', icon: <Computer />, color: 'primary' },
    { name: 'Non-profit', icon: <VolunteerActivism />, color: 'secondary' },
  ];

  const features = [
    {
      icon: <AutoAwesome sx={{ fontSize: 40, color: 'primary.main' }} />,
      title: 'AI-Powered Content Creation',
      description: 'Generate viral-ready content optimized for your specific business niche with our advanced AI engine.',
      details: ['Multi-platform content adaptation', 'Psychological trigger integration', 'Viral potential optimization']
    },
    {
      icon: <AttachMoney sx={{ fontSize: 40, color: 'success.main' }} />,
      title: 'Revenue Tracking & Attribution',
      description: 'Track every dollar earned from your social media posts with multi-touch attribution analysis.',
      details: ['Real-time revenue tracking', 'Post-level attribution', 'ROI optimization insights']
    },
    {
      icon: <Campaign sx={{ fontSize: 40, color: 'warning.main' }} />,
      title: 'Advanced Ad Creative Engine',
      description: 'Generate high-converting ad creatives with psychological triggers and A/B testing capabilities.',
      details: ['Psychology-based optimization', 'Platform-specific creatives', 'Automated A/B testing']
    },
    {
      icon: <Analytics sx={{ fontSize: 40, color: 'info.main' }} />,
      title: 'Business Intelligence Suite',
      description: 'Comprehensive analytics dashboard with AI-generated insights and performance monitoring.',
      details: ['Predictive analytics', 'Performance optimization', 'Competitor analysis']
    },
    {
      icon: <Psychology sx={{ fontSize: 40, color: 'secondary.main' }} />,
      title: 'Psychological Optimization',
      description: 'Leverage proven psychological triggers to maximize engagement and conversion rates.',
      details: ['Scarcity & urgency tactics', 'Social proof integration', 'Authority positioning']
    },
    {
      icon: <Speed sx={{ fontSize: 40, color: 'error.main' }} />,
      title: 'Automated Scheduling',
      description: 'Smart scheduling system that posts at optimal times for maximum reach and engagement.',
      details: ['Optimal timing analysis', 'Multi-platform scheduling', 'Audience activity tracking']
    },
  ];

  const platforms = [
    'Instagram', 'LinkedIn', 'Facebook', 'TikTok', 'YouTube', 'Twitter', 'Pinterest', 'Reddit'
  ];

  const testimonials = [
    {
      name: 'Sarah Johnson',
      role: 'Fitness Coach',
      avatar: 'SJ',
      content: 'AutoGuru Universal transformed my social media presence. I went from 5k to 50k followers in 6 months and my revenue increased by 300%.',
      rating: 5,
      revenue: '$15,000/month'
    },
    {
      name: 'Marcus Thompson',
      role: 'Business Consultant',
      avatar: 'MT',
      content: 'The AI insights are incredible. It predicted market trends that helped me position my services perfectly. My client acquisition cost dropped by 60%.',
      rating: 5,
      revenue: '$25,000/month'
    },
    {
      name: 'Lisa Chen',
      role: 'E-commerce Owner',
      avatar: 'LC',
      content: 'The revenue tracking feature is a game-changer. I can see exactly which posts drive sales and optimize accordingly. ROI increased by 400%.',
      rating: 5,
      revenue: '$50,000/month'
    },
  ];

  const stats = [
    { value: '10,000+', label: 'Active Users' },
    { value: '500M+', label: 'Content Generated' },
    { value: '$50M+', label: 'Revenue Tracked' },
    { value: '95%', label: 'Success Rate' },
  ];

  const pricingPlans = [
    {
      name: 'Starter',
      price: '$29',
      period: '/month',
      features: [
        '5 Social Media Platforms',
        '100 AI-Generated Posts/month',
        'Basic Analytics',
        'Email Support',
        'Revenue Tracking'
      ],
      popular: false
    },
    {
      name: 'Professional',
      price: '$99',
      period: '/month',
      features: [
        'All Platforms',
        'Unlimited AI Content',
        'Advanced Analytics',
        'Ad Creative Engine',
        'Priority Support',
        'API Access'
      ],
      popular: true
    },
    {
      name: 'Enterprise',
      price: '$299',
      period: '/month',
      features: [
        'Everything in Professional',
        'Custom AI Training',
        'White-label Solution',
        'Dedicated Account Manager',
        'Custom Integrations',
        'SLA Guarantee'
      ],
      popular: false
    }
  ];

  const handleGetStarted = () => {
    navigate('/signup');
  };

  const handleWatchDemo = () => {
    setDemoDialogOpen(true);
  };

  return (
    <Box>
      {/* Hero Section */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          py: { xs: 8, md: 12 },
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={6}>
              <Typography
                variant="h1"
                sx={{
                  fontSize: { xs: '2.5rem', md: '3.5rem' },
                  fontWeight: 'bold',
                  mb: 2,
                  lineHeight: 1.2,
                }}
              >
                The Universal Social Media
                <Box component="span" sx={{ color: '#FFD700' }}>
                  {' '}Automation Platform
                </Box>
              </Typography>
              <Typography
                variant="h5"
                sx={{
                  mb: 4,
                  opacity: 0.9,
                  fontSize: { xs: '1.1rem', md: '1.3rem' },
                }}
              >
                AI-powered content creation, revenue tracking, and advertising optimization
                for ANY business niche. From fitness coaches to business consultants to artists.
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, mb: 4, flexWrap: 'wrap' }}>
                <Button
                  variant="contained"
                  size="large"
                  onClick={handleGetStarted}
                  sx={{
                    backgroundColor: '#FFD700',
                    color: 'black',
                    fontWeight: 'bold',
                    px: 4,
                    py: 1.5,
                    '&:hover': { backgroundColor: '#FFC107' },
                  }}
                >
                  Start Free Trial
                </Button>
                <Button
                  variant="outlined"
                  size="large"
                  onClick={handleWatchDemo}
                  startIcon={<PlayArrow />}
                  sx={{
                    borderColor: 'white',
                    color: 'white',
                    px: 4,
                    py: 1.5,
                    '&:hover': { borderColor: '#FFD700', color: '#FFD700' },
                  }}
                >
                  Watch Demo
                </Button>
              </Box>
              <Box sx={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                {stats.map((stat, index) => (
                  <Box key={index} sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#FFD700' }}>
                      {stat.value}
                    </Typography>
                    <Typography variant="body2" sx={{ opacity: 0.8 }}>
                      {stat.label}
                    </Typography>
                  </Box>
                ))}
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box
                sx={{
                  position: 'relative',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                }}
              >
                <Paper
                  elevation={20}
                  sx={{
                    p: 4,
                    borderRadius: 4,
                    background: 'rgba(255, 255, 255, 0.1)',
                    backdropFilter: 'blur(10px)',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                  }}
                >
                  <Typography variant="h6" sx={{ mb: 2, color: '#FFD700' }}>
                    Live Platform Demo
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                    <Chip label="Revenue: $12,450 this month" color="success" />
                    <Chip label="ROI: 340%" color="primary" />
                  </Box>
                  <Typography variant="body2" sx={{ mb: 2 }}>
                    AI-generated content for fitness coach achieving 3.2x ROI
                  </Typography>
                  <Button
                    variant="contained"
                    fullWidth
                    onClick={handleWatchDemo}
                    sx={{ backgroundColor: '#FFD700', color: 'black' }}
                  >
                    See Full Demo
                  </Button>
                </Paper>
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Business Niches Section */}
      <Box sx={{ py: 8, backgroundColor: 'grey.50' }}>
        <Container maxWidth="lg">
          <Typography variant="h2" align="center" gutterBottom>
            Works for <Box component="span" sx={{ color: 'primary.main' }}>ANY</Box> Business Niche
          </Typography>
          <Typography variant="h6" align="center" color="textSecondary" sx={{ mb: 6 }}>
            Our AI automatically adapts to your specific business type and audience
          </Typography>
          <Grid container spacing={3}>
            {businessNiches.map((niche, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <Card
                  sx={{
                    height: '100%',
                    textAlign: 'center',
                    transition: 'transform 0.3s ease',
                    '&:hover': { transform: 'translateY(-5px)' },
                  }}
                >
                  <CardContent>
                    <Avatar
                      sx={{
                        mx: 'auto',
                        mb: 2,
                        bgcolor: `${niche.color}.main`,
                        width: 60,
                        height: 60,
                      }}
                    >
                      {niche.icon}
                    </Avatar>
                    <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                      {niche.name}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* Features Section */}
      <Box sx={{ py: 8 }}>
        <Container maxWidth="lg">
          <Typography variant="h2" align="center" gutterBottom>
            Revolutionary Features
          </Typography>
          <Typography variant="h6" align="center" color="textSecondary" sx={{ mb: 6 }}>
            Everything you need to dominate social media and maximize revenue
          </Typography>
          <Grid container spacing={4}>
            {features.map((feature, index) => (
              <Grid item xs={12} md={6} key={index}>
                <Card
                  sx={{
                    height: '100%',
                    p: 3,
                    transition: 'transform 0.3s ease',
                    '&:hover': { transform: 'translateY(-5px)' },
                  }}
                >
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      {feature.icon}
                      <Typography variant="h5" sx={{ ml: 2, fontWeight: 'bold' }}>
                        {feature.title}
                      </Typography>
                    </Box>
                    <Typography variant="body1" color="textSecondary" sx={{ mb: 3 }}>
                      {feature.description}
                    </Typography>
                    <List>
                      {feature.details.map((detail, detailIndex) => (
                        <ListItem key={detailIndex} sx={{ py: 0.5 }}>
                          <ListItemIcon sx={{ minWidth: 30 }}>
                            <CheckCircle color="success" fontSize="small" />
                          </ListItemIcon>
                          <ListItemText primary={detail} />
                        </ListItem>
                      ))}
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* Platform Integration Section */}
      <Box sx={{ py: 8, backgroundColor: 'grey.50' }}>
        <Container maxWidth="lg">
          <Typography variant="h2" align="center" gutterBottom>
            Seamless Platform Integration
          </Typography>
          <Typography variant="h6" align="center" color="textSecondary" sx={{ mb: 6 }}>
            Manage all your social media platforms from one powerful dashboard
          </Typography>
          <Grid container spacing={2} justifyContent="center">
            {platforms.map((platform, index) => (
              <Grid item key={index}>
                <Chip
                  label={platform}
                  size="large"
                  variant="outlined"
                  sx={{
                    p: 2,
                    fontSize: '1.1rem',
                    fontWeight: 'bold',
                    borderWidth: 2,
                    '&:hover': { borderColor: 'primary.main', backgroundColor: 'primary.light' },
                  }}
                />
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* Testimonials Section */}
      <Box sx={{ py: 8 }}>
        <Container maxWidth="lg">
          <Typography variant="h2" align="center" gutterBottom>
            Success Stories
          </Typography>
          <Typography variant="h6" align="center" color="textSecondary" sx={{ mb: 6 }}>
            Real results from real businesses across different niches
          </Typography>
          <Grid container spacing={4}>
            {testimonials.map((testimonial, index) => (
              <Grid item xs={12} md={4} key={index}>
                <Card sx={{ height: '100%', p: 3 }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <Avatar sx={{ mr: 2, bgcolor: 'primary.main' }}>
                        {testimonial.avatar}
                      </Avatar>
                      <Box>
                        <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                          {testimonial.name}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          {testimonial.role}
                        </Typography>
                      </Box>
                    </Box>
                    <Box sx={{ display: 'flex', mb: 2 }}>
                      {[...Array(testimonial.rating)].map((_, i) => (
                        <Star key={i} sx={{ color: '#FFD700' }} />
                      ))}
                    </Box>
                    <Typography variant="body1" sx={{ mb: 2, fontStyle: 'italic' }}>
                      "{testimonial.content}"
                    </Typography>
                    <Chip
                      label={`Revenue: ${testimonial.revenue}`}
                      color="success"
                      variant="outlined"
                    />
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* Pricing Section */}
      <Box sx={{ py: 8, backgroundColor: 'grey.50' }}>
        <Container maxWidth="lg">
          <Typography variant="h2" align="center" gutterBottom>
            Simple, Transparent Pricing
          </Typography>
          <Typography variant="h6" align="center" color="textSecondary" sx={{ mb: 6 }}>
            Choose the plan that fits your business needs
          </Typography>
          <Grid container spacing={4} justifyContent="center">
            {pricingPlans.map((plan, index) => (
              <Grid item xs={12} md={4} key={index}>
                <Card
                  sx={{
                    height: '100%',
                    position: 'relative',
                    ...(plan.popular && {
                      border: '2px solid',
                      borderColor: 'primary.main',
                      transform: 'scale(1.05)',
                    }),
                  }}
                >
                  {plan.popular && (
                    <Chip
                      label="Most Popular"
                      color="primary"
                      sx={{
                        position: 'absolute',
                        top: -10,
                        left: '50%',
                        transform: 'translateX(-50%)',
                        fontWeight: 'bold',
                      }}
                    />
                  )}
                  <CardContent sx={{ p: 4, textAlign: 'center' }}>
                    <Typography variant="h4" sx={{ fontWeight: 'bold', mb: 1 }}>
                      {plan.name}
                    </Typography>
                    <Typography variant="h3" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                      {plan.price}
                      <Typography component="span" variant="h6" color="textSecondary">
                        {plan.period}
                      </Typography>
                    </Typography>
                    <List sx={{ mt: 3 }}>
                      {plan.features.map((feature, featureIndex) => (
                        <ListItem key={featureIndex} sx={{ py: 0.5 }}>
                          <ListItemIcon sx={{ minWidth: 30 }}>
                            <CheckCircle color="success" fontSize="small" />
                          </ListItemIcon>
                          <ListItemText primary={feature} />
                        </ListItem>
                      ))}
                    </List>
                    <Button
                      variant={plan.popular ? 'contained' : 'outlined'}
                      size="large"
                      fullWidth
                      sx={{ mt: 3, py: 1.5 }}
                      onClick={handleGetStarted}
                    >
                      Get Started
                    </Button>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* FAQ Section */}
      <Box sx={{ py: 8 }}>
        <Container maxWidth="md">
          <Typography variant="h2" align="center" gutterBottom>
            Frequently Asked Questions
          </Typography>
          <Box sx={{ mt: 4 }}>
            {[
              {
                question: 'How does AutoGuru Universal work for different business niches?',
                answer: 'Our AI system automatically detects your business niche and adapts content creation, psychological triggers, and optimization strategies specifically for your industry. Whether you\'re a fitness coach, business consultant, or artist, the platform learns your unique audience and market dynamics.'
              },
              {
                question: 'Can I track revenue from social media posts?',
                answer: 'Yes! Our advanced revenue attribution system tracks every dollar earned from your social media posts using multi-touch attribution analysis. You can see exactly which posts, platforms, and content types generate the most revenue for your business.'
              },
              {
                question: 'What makes the AI advertising engine different?',
                answer: 'Our advertising engine uses psychological triggers, platform-specific optimization, and automated A/B testing to create high-converting ad creatives. It analyzes millions of successful campaigns to generate creatives that outperform industry standards.'
              },
              {
                question: 'Is there a free trial available?',
                answer: 'Yes! We offer a 14-day free trial with full access to all features. You can test the platform, generate content, and see results before committing to a paid plan.'
              },
              {
                question: 'How quickly can I see results?',
                answer: 'Most users see improved engagement within the first week and significant revenue growth within 30 days. The AI learns your audience quickly and optimizes content for maximum impact.'
              }
            ].map((faq, index) => (
              <Accordion key={index} sx={{ mb: 2 }}>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                    {faq.question}
                  </Typography>
                </AccordionSummary>
                <AccordionDetails>
                  <Typography variant="body1" color="textSecondary">
                    {faq.answer}
                  </Typography>
                </AccordionDetails>
              </Accordion>
            ))}
          </Box>
        </Container>
      </Box>

      {/* CTA Section */}
      <Box
        sx={{
          py: 8,
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          textAlign: 'center',
        }}
      >
        <Container maxWidth="md">
          <Typography variant="h2" gutterBottom>
            Ready to Transform Your Social Media?
          </Typography>
          <Typography variant="h6" sx={{ mb: 4, opacity: 0.9 }}>
            Join thousands of businesses already growing with AutoGuru Universal
          </Typography>
          <Button
            variant="contained"
            size="large"
            onClick={handleGetStarted}
            sx={{
              backgroundColor: '#FFD700',
              color: 'black',
              fontWeight: 'bold',
              px: 6,
              py: 2,
              fontSize: '1.2rem',
              '&:hover': { backgroundColor: '#FFC107' },
            }}
          >
            Start Your Free Trial Now
          </Button>
          <Typography variant="body2" sx={{ mt: 2, opacity: 0.8 }}>
            No credit card required • 14-day free trial • Cancel anytime
          </Typography>
        </Container>
      </Box>

      {/* Demo Dialog */}
      <Dialog
        open={demoDialogOpen}
        onClose={() => setDemoDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Typography variant="h4" align="center">
            AutoGuru Universal Demo
          </Typography>
        </DialogTitle>
        <DialogContent>
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="h6" gutterBottom>
              See how AutoGuru Universal transforms your social media strategy
            </Typography>
            <Typography variant="body1" color="textSecondary" sx={{ mb: 4 }}>
              Watch our comprehensive demo showcasing:
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <List>
                  <ListItem>
                    <ListItemIcon><SmartToy color="primary" /></ListItemIcon>
                    <ListItemText primary="AI Content Generation" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon><MonetizationOn color="primary" /></ListItemIcon>
                    <ListItemText primary="Revenue Tracking" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon><Insights color="primary" /></ListItemIcon>
                    <ListItemText primary="Advanced Analytics" />
                  </ListItem>
                </List>
              </Grid>
              <Grid item xs={12} md={6}>
                <List>
                  <ListItem>
                    <ListItemIcon><CampaignOutlined color="primary" /></ListItemIcon>
                    <ListItemText primary="Ad Creative Engine" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon><BarChart color="primary" /></ListItemIcon>
                    <ListItemText primary="Performance Optimization" />
                  </ListItem>
                  <ListItem>
                    <ListItemIcon><AdminPanelSettings color="primary" /></ListItemIcon>
                    <ListItemText primary="Admin Tools" />
                  </ListItem>
                </List>
              </Grid>
            </Grid>
          </Box>
        </DialogContent>
        <DialogActions sx={{ justifyContent: 'center', pb: 3 }}>
          <Button onClick={() => setDemoDialogOpen(false)} size="large">
            Close
          </Button>
          <Button
            variant="contained"
            size="large"
            onClick={() => {
              setDemoDialogOpen(false);
              handleGetStarted();
            }}
            sx={{ ml: 2 }}
          >
            Start Free Trial
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default LandingPage;