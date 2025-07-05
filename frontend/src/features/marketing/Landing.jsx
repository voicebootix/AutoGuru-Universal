import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  Avatar,
  Rating,
  Chip,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  IconButton,
  Fade,
  Slide,
  useTheme,
  useMediaQuery,
  styled
} from '@mui/material';
import {
  TrendingUp,
  AutoAwesome,
  Speed,
  Security,
  CheckCircle,
  ArrowForward,
  PlayArrow,
  Instagram,
  LinkedIn,
  Facebook,
  Twitter,
  YouTube,
  MusicNote,
  Star,
  Groups,
  Analytics,
  Rocket,
  AttachMoney,
  EmojiEmotions,
  BusinessCenter,
  FitnessCenter,
  Palette,
  School,
  Store,
  Engineering,
  Favorite
} from '@mui/icons-material';
import { brandColors, brandGradients, brandStats, brandTestimonials, brandContent } from '../../assets/brand';

// Styled Components
const GradientBox = styled(Box)(({ theme }) => ({
  background: brandGradients.hero,
  borderRadius: '16px',
  padding: '2px'
}));

const HeroSection = styled(Box)(({ theme }) => ({
  background: `linear-gradient(135deg, ${brandColors.primary[50]} 0%, ${brandColors.secondary[50]} 100%)`,
  minHeight: '90vh',
  display: 'flex',
  alignItems: 'center',
  position: 'relative',
  overflow: 'hidden',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'radial-gradient(circle at 30% 20%, rgba(14, 165, 233, 0.1) 0%, transparent 50%), radial-gradient(circle at 70% 80%, rgba(217, 70, 239, 0.1) 0%, transparent 50%)',
    zIndex: 1
  }
}));

const StatsCard = styled(Card)(({ theme }) => ({
  background: 'rgba(255, 255, 255, 0.1)',
  backdropFilter: 'blur(10px)',
  border: '1px solid rgba(255, 255, 255, 0.2)',
  borderRadius: '16px',
  textAlign: 'center',
  padding: '24px',
  height: '100%',
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'translateY(-8px)',
    boxShadow: '0 20px 40px rgba(0, 0, 0, 0.1)'
  }
}));

const FeatureCard = styled(Card)(({ theme }) => ({
  height: '100%',
  borderRadius: '16px',
  border: '1px solid rgba(14, 165, 233, 0.1)',
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'translateY(-8px)',
    boxShadow: '0 20px 40px rgba(0, 0, 0, 0.1)',
    borderColor: brandColors.primary[300]
  }
}));

const TestimonialCard = styled(Card)(({ theme }) => ({
  height: '100%',
  borderRadius: '16px',
  border: '1px solid rgba(217, 70, 239, 0.1)',
  background: 'linear-gradient(145deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%)',
  transition: 'all 0.3s ease',
  '&:hover': {
    transform: 'translateY(-8px)',
    boxShadow: '0 20px 40px rgba(0, 0, 0, 0.1)'
  }
}));

const PlatformChip = styled(Chip)(({ theme }) => ({
  margin: '4px',
  background: brandGradients.primary,
  color: 'white',
  fontWeight: 600,
  '&:hover': {
    background: brandGradients.secondary
  }
}));

const BusinessNicheCard = styled(Card)(({ theme }) => ({
  height: '100%',
  borderRadius: '16px',
  background: 'linear-gradient(145deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%)',
  border: '1px solid rgba(14, 165, 233, 0.1)',
  transition: 'all 0.3s ease',
  cursor: 'pointer',
  '&:hover': {
    transform: 'translateY(-8px)',
    boxShadow: '0 20px 40px rgba(0, 0, 0, 0.1)',
    borderColor: brandColors.primary[300]
  }
}));

const Landing = () => {
  const navigate = useNavigate();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [currentStat, setCurrentStat] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStat(prev => (prev + 1) % 4);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleGetStarted = () => {
    navigate('/signup');
  };

  const handleWatchDemo = () => {
    // TODO: Implement demo video modal
    console.log('Watch demo clicked');
  };

  const stats = [
    { label: 'Active Businesses', value: brandStats.businesses, icon: <BusinessCenter /> },
    { label: 'Posts Generated', value: brandStats.posts, icon: <Analytics /> },
    { label: 'Platforms Supported', value: brandStats.platforms, icon: <Rocket /> },
    { label: 'Countries Served', value: brandStats.countries, icon: <Groups /> }
  ];

  const features = [
    {
      icon: <AutoAwesome sx={{ fontSize: 48, color: brandColors.primary[500] }} />,
      title: 'AI-Powered Intelligence',
      description: 'Our advanced AI automatically detects your business niche and creates content that resonates with your specific audience.',
      benefits: ['Automatic niche detection', 'Content optimization', 'Viral potential scoring']
    },
    {
      icon: <Speed sx={{ fontSize: 48, color: brandColors.secondary[500] }} />,
      title: 'Universal Business Support',
      description: 'From fitness coaches to Fortune 500 companies, AutoGuru Universal adapts to ANY business type automatically.',
      benefits: ['Works for any niche', 'No manual setup required', 'Scalable solutions']
    },
    {
      icon: <TrendingUp sx={{ fontSize: 48, color: brandColors.success[500] }} />,
      title: 'Proven Results',
      description: 'Our clients see average engagement increases of 300% and follower growth of 150% within the first 90 days.',
      benefits: ['3x more engagement', '150% follower growth', 'Measurable ROI']
    },
    {
      icon: <Security sx={{ fontSize: 48, color: brandColors.warning[500] }} />,
      title: 'Enterprise Security',
      description: 'Bank-grade security with encrypted credentials, SOC 2 compliance, and 99.9% uptime guarantee.',
      benefits: ['Bank-grade security', 'SOC 2 compliant', '99.9% uptime SLA']
    }
  ];

  const businessNiches = [
    { icon: <FitnessCenter />, title: 'Fitness & Wellness', description: 'Transform your fitness business' },
    { icon: <BusinessCenter />, title: 'Business Consulting', description: 'Scale your consulting practice' },
    { icon: <Palette />, title: 'Creative Services', description: 'Showcase your creative work' },
    { icon: <School />, title: 'Education', description: 'Grow your educational impact' },
    { icon: <Store />, title: 'E-commerce', description: 'Boost your online sales' },
    { icon: <Engineering />, title: 'Technology', description: 'Amplify your tech solutions' },
    { icon: <Favorite />, title: 'Non-Profit', description: 'Increase your social impact' },
    { icon: <EmojiEmotions />, title: 'Any Business', description: 'AI adapts to your niche' }
  ];

  const platforms = [
    { name: 'Instagram', icon: <Instagram />, color: '#E4405F' },
    { name: 'LinkedIn', icon: <LinkedIn />, color: '#0077B5' },
    { name: 'Facebook', icon: <Facebook />, color: '#1877F2' },
    { name: 'Twitter', icon: <Twitter />, color: '#1DA1F2' },
    { name: 'YouTube', icon: <YouTube />, color: '#FF0000' },
    { name: 'TikTok', icon: <MusicNote />, color: '#000000' }
  ];

  return (
    <Box sx={{ overflow: 'hidden' }}>
      {/* Hero Section */}
      <HeroSection>
        <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 2 }}>
          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={6}>
              <Fade in timeout={1000}>
                <Box>
                  <Typography
                    variant={isMobile ? 'h3' : 'h2'}
                    component="h1"
                    gutterBottom
                    sx={{
                      fontWeight: 800,
                      background: brandGradients.hero,
                      backgroundClip: 'text',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      lineHeight: 1.2
                    }}
                  >
                    {brandContent.hero.title}
                  </Typography>
                  <Typography
                    variant="h6"
                    color="text.secondary"
                    paragraph
                    sx={{ fontSize: isMobile ? '1.1rem' : '1.25rem', lineHeight: 1.6 }}
                  >
                    {brandContent.hero.subtitle}
                  </Typography>
                  <Box sx={{ mt: 4, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                    <Button
                      variant="contained"
                      size="large"
                      onClick={handleGetStarted}
                      sx={{
                        background: brandGradients.primary,
                        px: 4,
                        py: 1.5,
                        borderRadius: '12px',
                        fontWeight: 600,
                        fontSize: '1.1rem',
                        '&:hover': {
                          background: brandGradients.secondary,
                          transform: 'translateY(-2px)'
                        }
                      }}
                      endIcon={<ArrowForward />}
                    >
                      Start Free Trial
                    </Button>
                    <Button
                      variant="outlined"
                      size="large"
                      onClick={handleWatchDemo}
                      sx={{
                        px: 4,
                        py: 1.5,
                        borderRadius: '12px',
                        borderColor: brandColors.primary[500],
                        color: brandColors.primary[500],
                        fontWeight: 600,
                        fontSize: '1.1rem',
                        '&:hover': {
                          borderColor: brandColors.primary[600],
                          background: brandColors.primary[50]
                        }
                      }}
                      startIcon={<PlayArrow />}
                    >
                      Watch Demo
                    </Button>
                  </Box>
                </Box>
              </Fade>
            </Grid>
            <Grid item xs={12} md={6}>
              <Slide direction="left" in timeout={1000}>
                <Box>
                  <Grid container spacing={2}>
                    {stats.map((stat, index) => (
                      <Grid item xs={6} key={index}>
                        <StatsCard>
                          <Box sx={{ color: brandColors.primary[500], mb: 1 }}>
                            {stat.icon}
                          </Box>
                          <Typography variant="h4" fontWeight="bold" color="primary">
                            {stat.value}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {stat.label}
                          </Typography>
                        </StatsCard>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              </Slide>
            </Grid>
          </Grid>
        </Container>
      </HeroSection>

      {/* Business Niches Section */}
      <Box sx={{ py: 8, background: 'linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)' }}>
        <Container maxWidth="lg">
          <Box textAlign="center" mb={6}>
            <Typography variant="h3" component="h2" gutterBottom fontWeight="bold">
              Works for Every Business Type
            </Typography>
            <Typography variant="h6" color="text.secondary" paragraph>
              AI automatically adapts to your industry - no manual setup required
            </Typography>
          </Box>
          <Grid container spacing={3}>
            {businessNiches.map((niche, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <Fade in timeout={1000 + index * 100}>
                  <BusinessNicheCard>
                    <CardContent sx={{ textAlign: 'center', p: 3 }}>
                      <Box sx={{ color: brandColors.primary[500], mb: 2 }}>
                        {React.cloneElement(niche.icon, { sx: { fontSize: 40 } })}
                      </Box>
                      <Typography variant="h6" fontWeight="bold" gutterBottom>
                        {niche.title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {niche.description}
                      </Typography>
                    </CardContent>
                  </BusinessNicheCard>
                </Fade>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* Features Section */}
      <Box sx={{ py: 8, background: 'white' }}>
        <Container maxWidth="lg">
          <Box textAlign="center" mb={6}>
            <Typography variant="h3" component="h2" gutterBottom fontWeight="bold">
              {brandContent.features.title}
            </Typography>
            <Typography variant="h6" color="text.secondary" paragraph>
              {brandContent.features.subtitle}
            </Typography>
          </Box>
          <Grid container spacing={4}>
            {features.map((feature, index) => (
              <Grid item xs={12} md={6} key={index}>
                <Fade in timeout={1000 + index * 200}>
                  <FeatureCard>
                    <CardContent sx={{ p: 4 }}>
                      <Box sx={{ mb: 3 }}>
                        {feature.icon}
                      </Box>
                      <Typography variant="h5" fontWeight="bold" gutterBottom>
                        {feature.title}
                      </Typography>
                      <Typography variant="body1" color="text.secondary" paragraph>
                        {feature.description}
                      </Typography>
                      <List>
                        {feature.benefits.map((benefit, benefitIndex) => (
                          <ListItem key={benefitIndex} sx={{ px: 0 }}>
                            <ListItemIcon sx={{ minWidth: 36 }}>
                              <CheckCircle sx={{ color: brandColors.success[500] }} />
                            </ListItemIcon>
                            <ListItemText primary={benefit} />
                          </ListItem>
                        ))}
                      </List>
                    </CardContent>
                  </FeatureCard>
                </Fade>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* Platforms Section */}
      <Box sx={{ py: 8, background: brandColors.gray[50] }}>
        <Container maxWidth="lg">
          <Box textAlign="center" mb={6}>
            <Typography variant="h3" component="h2" gutterBottom fontWeight="bold">
              One Platform, All Networks
            </Typography>
            <Typography variant="h6" color="text.secondary" paragraph>
              Publish and optimize content across all major social media platforms
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 2 }}>
            {platforms.map((platform, index) => (
              <Fade in timeout={1000 + index * 100} key={index}>
                <PlatformChip
                  icon={React.cloneElement(platform.icon, { sx: { color: 'white !important' } })}
                  label={platform.name}
                  sx={{ fontSize: '1rem', px: 2, py: 1 }}
                />
              </Fade>
            ))}
          </Box>
        </Container>
      </Box>

      {/* Testimonials Section */}
      <Box sx={{ py: 8, background: 'white' }}>
        <Container maxWidth="lg">
          <Box textAlign="center" mb={6}>
            <Typography variant="h3" component="h2" gutterBottom fontWeight="bold">
              {brandContent.social.title}
            </Typography>
            <Typography variant="h6" color="text.secondary" paragraph>
              {brandContent.social.subtitle}
            </Typography>
          </Box>
          <Grid container spacing={4}>
            {brandTestimonials.map((testimonial, index) => (
              <Grid item xs={12} md={4} key={index}>
                <Fade in timeout={1000 + index * 200}>
                  <TestimonialCard>
                    <CardContent sx={{ p: 4 }}>
                      <Rating value={testimonial.rating} readOnly sx={{ mb: 2 }} />
                      <Typography variant="body1" paragraph>
                        "{testimonial.content}"
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', mt: 3 }}>
                        <Avatar
                          src={testimonial.image}
                          alt={testimonial.name}
                          sx={{ width: 56, height: 56, mr: 2 }}
                        />
                        <Box>
                          <Typography variant="h6" fontWeight="bold">
                            {testimonial.name}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {testimonial.role}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {testimonial.company}
                          </Typography>
                        </Box>
                      </Box>
                    </CardContent>
                  </TestimonialCard>
                </Fade>
              </Grid>
            ))}
          </Grid>
        </Container>
      </Box>

      {/* CTA Section */}
      <Box sx={{ py: 8, background: brandGradients.hero, color: 'white' }}>
        <Container maxWidth="md">
          <Box textAlign="center">
            <Typography variant="h3" component="h2" gutterBottom fontWeight="bold">
              Ready to Transform Your Social Media?
            </Typography>
            <Typography variant="h6" paragraph sx={{ opacity: 0.9 }}>
              Join thousands of businesses already growing with AutoGuru Universal
            </Typography>
            <Box sx={{ mt: 4 }}>
              <Button
                variant="contained"
                size="large"
                onClick={handleGetStarted}
                sx={{
                  background: 'white',
                  color: brandColors.primary[600],
                  px: 6,
                  py: 2,
                  borderRadius: '12px',
                  fontWeight: 600,
                  fontSize: '1.2rem',
                  '&:hover': {
                    background: brandColors.gray[100],
                    transform: 'translateY(-2px)'
                  }
                }}
                endIcon={<ArrowForward />}
              >
                Start Your Free Trial
              </Button>
            </Box>
            <Typography variant="body2" sx={{ mt: 2, opacity: 0.8 }}>
              No credit card required • 14-day free trial • Cancel anytime
            </Typography>
          </Box>
        </Container>
      </Box>
    </Box>
  );
};

export default Landing;