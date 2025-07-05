import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Switch,
  FormControlLabel,
  Chip,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  TextField,
  InputAdornment,
  Slider,
  useTheme,
  styled
} from '@mui/material';
import {
  CheckCircle,
  Star,
  TrendingUp,
  Security,
  Speed,
  ArrowForward,
  ExpandMore,
  AttachMoney,
  BusinessCenter,
  Groups,
  Analytics,
  AutoAwesome,
  Rocket,
  Diamond,
  WorkspacePremium
} from '@mui/icons-material';
import { brandColors, brandGradients } from '../../assets/brand';

const PricingCard = styled(Card)(({ theme, featured }) => ({
  height: '100%',
  borderRadius: '20px',
  border: featured ? `3px solid ${brandColors.primary[500]}` : '1px solid rgba(0, 0, 0, 0.1)',
  position: 'relative',
  transition: 'all 0.3s ease',
  transform: featured ? 'scale(1.05)' : 'scale(1)',
  background: featured ? 'linear-gradient(145deg, rgba(255,255,255,1) 0%, rgba(248,250,252,1) 100%)' : 'white',
  '&:hover': {
    transform: featured ? 'scale(1.05)' : 'scale(1.02)',
    boxShadow: '0 20px 40px rgba(0, 0, 0, 0.15)'
  }
}));

const FeaturedBadge = styled(Chip)(({ theme }) => ({
  position: 'absolute',
  top: -12,
  left: '50%',
  transform: 'translateX(-50%)',
  background: brandGradients.primary,
  color: 'white',
  fontWeight: 'bold',
  fontSize: '0.9rem',
  zIndex: 1
}));

const ROICalculator = styled(Paper)(({ theme }) => ({
  padding: '32px',
  borderRadius: '20px',
  background: 'linear-gradient(145deg, rgba(14, 165, 233, 0.05) 0%, rgba(217, 70, 239, 0.05) 100%)',
  border: '1px solid rgba(14, 165, 233, 0.1)'
}));

const Pricing = () => {
  const navigate = useNavigate();
  const theme = useTheme();
  const [isAnnual, setIsAnnual] = useState(true);
  const [followers, setFollowers] = useState(10000);
  const [postsPerWeek, setPostsPerWeek] = useState(5);

  const plans = [
    {
      name: 'Starter',
      description: 'Perfect for small businesses and solo entrepreneurs',
      monthlyPrice: 29,
      annualPrice: 290,
      icon: <Rocket sx={{ fontSize: 40, color: brandColors.success[500] }} />,
      features: [
        '1 Business Profile',
        '3 Social Media Platforms',
        '50 AI-Generated Posts/month',
        'Basic Analytics',
        'Content Calendar',
        'Email Support',
        'Mobile App Access'
      ],
      limits: {
        profiles: 1,
        platforms: 3,
        posts: 50,
        analytics: 'Basic',
        support: 'Email'
      },
      popular: false
    },
    {
      name: 'Professional',
      description: 'Ideal for growing businesses and agencies',
      monthlyPrice: 79,
      annualPrice: 790,
      icon: <Star sx={{ fontSize: 40, color: brandColors.primary[500] }} />,
      features: [
        '5 Business Profiles',
        '6 Social Media Platforms',
        '200 AI-Generated Posts/month',
        'Advanced Analytics',
        'A/B Testing',
        'Priority Support',
        'Team Collaboration',
        'Custom Branding',
        'Revenue Tracking'
      ],
      limits: {
        profiles: 5,
        platforms: 6,
        posts: 200,
        analytics: 'Advanced',
        support: 'Priority'
      },
      popular: true
    },
    {
      name: 'Enterprise',
      description: 'For large organizations and agencies',
      monthlyPrice: 199,
      annualPrice: 1990,
      icon: <Diamond sx={{ fontSize: 40, color: brandColors.secondary[500] }} />,
      features: [
        'Unlimited Business Profiles',
        'All Social Media Platforms',
        'Unlimited AI-Generated Posts',
        'Enterprise Analytics',
        'White-label Solution',
        'Dedicated Success Manager',
        'API Access',
        'Custom Integrations',
        'Advanced Security',
        'SLA Guarantee'
      ],
      limits: {
        profiles: 'Unlimited',
        platforms: 'All',
        posts: 'Unlimited',
        analytics: 'Enterprise',
        support: 'Dedicated Manager'
      },
      popular: false
    }
  ];

  const features = [
    {
      category: 'Core Features',
      items: [
        { name: 'AI Content Generation', starter: true, pro: true, enterprise: true },
        { name: 'Multi-Platform Publishing', starter: true, pro: true, enterprise: true },
        { name: 'Content Calendar', starter: true, pro: true, enterprise: true },
        { name: 'Basic Analytics', starter: true, pro: false, enterprise: false },
        { name: 'Advanced Analytics', starter: false, pro: true, enterprise: true },
        { name: 'Enterprise Analytics', starter: false, pro: false, enterprise: true },
        { name: 'A/B Testing', starter: false, pro: true, enterprise: true },
        { name: 'Revenue Tracking', starter: false, pro: true, enterprise: true }
      ]
    },
    {
      category: 'Platforms & Integrations',
      items: [
        { name: 'Instagram', starter: true, pro: true, enterprise: true },
        { name: 'Facebook', starter: true, pro: true, enterprise: true },
        { name: 'LinkedIn', starter: true, pro: true, enterprise: true },
        { name: 'Twitter', starter: false, pro: true, enterprise: true },
        { name: 'YouTube', starter: false, pro: true, enterprise: true },
        { name: 'TikTok', starter: false, pro: true, enterprise: true },
        { name: 'API Access', starter: false, pro: false, enterprise: true },
        { name: 'Custom Integrations', starter: false, pro: false, enterprise: true }
      ]
    },
    {
      category: 'Business Features',
      items: [
        { name: 'Team Collaboration', starter: false, pro: true, enterprise: true },
        { name: 'Custom Branding', starter: false, pro: true, enterprise: true },
        { name: 'White-label Solution', starter: false, pro: false, enterprise: true },
        { name: 'Multiple Business Profiles', starter: '1', pro: '5', enterprise: 'Unlimited' },
        { name: 'Client Management', starter: false, pro: true, enterprise: true },
        { name: 'Reporting Dashboard', starter: false, pro: true, enterprise: true }
      ]
    },
    {
      category: 'Support & Security',
      items: [
        { name: 'Email Support', starter: true, pro: true, enterprise: true },
        { name: 'Priority Support', starter: false, pro: true, enterprise: true },
        { name: 'Dedicated Success Manager', starter: false, pro: false, enterprise: true },
        { name: 'SOC 2 Compliance', starter: false, pro: true, enterprise: true },
        { name: 'SLA Guarantee', starter: false, pro: false, enterprise: true },
        { name: 'Advanced Security', starter: false, pro: false, enterprise: true }
      ]
    }
  ];

  const faqs = [
    {
      question: 'What makes AutoGuru Universal different from other social media tools?',
      answer: 'AutoGuru Universal is the only platform that works for ANY business niche automatically. Our AI detects your business type and creates content specifically for your industry, whether you\'re a fitness coach, business consultant, or artist.'
    },
    {
      question: 'How does the AI content generation work?',
      answer: 'Our AI analyzes your business, audience, and industry trends to create viral-optimized content. It automatically adapts writing style, hashtags, and posting strategies for each social media platform.'
    },
    {
      question: 'Can I upgrade or downgrade my plan anytime?',
      answer: 'Yes! You can change your plan at any time. Upgrades take effect immediately, and downgrades take effect at the next billing cycle. We\'ll prorate any differences.'
    },
    {
      question: 'Is there a free trial?',
      answer: 'Yes! We offer a 14-day free trial with full access to Professional features. No credit card required to start.'
    },
    {
      question: 'What platforms do you support?',
      answer: 'We support Instagram, Facebook, LinkedIn, Twitter, YouTube, and TikTok. More platforms are added regularly based on user demand.'
    },
    {
      question: 'How does billing work?',
      answer: 'You can choose monthly or annual billing. Annual plans save 20%. All plans include a 30-day money-back guarantee.'
    }
  ];

  const calculateROI = () => {
    const timeSpent = postsPerWeek * 2; // 2 hours per post manually
    const timeSaved = timeSpent * 0.8; // 80% time savings
    const hourlyRate = 50; // Average hourly rate
    const monthlySavings = timeSaved * 4 * hourlyRate; // 4 weeks per month
    const engagementBoost = followers * 0.03; // 3% engagement boost
    const conversionValue = engagementBoost * 0.02 * 100; // 2% conversion at $100 value
    const totalValue = monthlySavings + conversionValue;
    const planCost = isAnnual ? plans[1].annualPrice / 12 : plans[1].monthlyPrice;
    const roi = ((totalValue - planCost) / planCost) * 100;

    return {
      timeSaved: Math.round(timeSaved),
      monthlySavings: Math.round(monthlySavings),
      engagementBoost: Math.round(engagementBoost),
      conversionValue: Math.round(conversionValue),
      totalValue: Math.round(totalValue),
      roi: Math.round(roi)
    };
  };

  const roiData = calculateROI();

  const handleSelectPlan = (planName) => {
    navigate('/signup', { state: { plan: planName, isAnnual } });
  };

  return (
    <Box sx={{ py: 8 }}>
      <Container maxWidth="lg">
        {/* Header */}
        <Box textAlign="center" mb={8}>
          <Typography variant="h2" component="h1" gutterBottom fontWeight="bold">
            Simple, Transparent Pricing
          </Typography>
          <Typography variant="h5" color="text.secondary" paragraph>
            Choose the perfect plan for your business growth
          </Typography>
          
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={isAnnual}
                  onChange={(e) => setIsAnnual(e.target.checked)}
                  color="primary"
                />
              }
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography>Annual Billing</Typography>
                  <Chip
                    label="Save 20%"
                    size="small"
                    sx={{
                      background: brandGradients.success,
                      color: 'white',
                      fontWeight: 'bold'
                    }}
                  />
                </Box>
              }
            />
          </Box>
        </Box>

        {/* Pricing Cards */}
        <Grid container spacing={4} justifyContent="center" mb={8}>
          {plans.map((plan, index) => (
            <Grid item xs={12} md={4} key={index}>
              <PricingCard featured={plan.popular}>
                {plan.popular && <FeaturedBadge label="Most Popular" />}
                <CardContent sx={{ p: 4, textAlign: 'center' }}>
                  <Box sx={{ mb: 3 }}>
                    {plan.icon}
                  </Box>
                  <Typography variant="h4" fontWeight="bold" gutterBottom>
                    {plan.name}
                  </Typography>
                  <Typography variant="body1" color="text.secondary" paragraph>
                    {plan.description}
                  </Typography>
                  <Box sx={{ mb: 3 }}>
                    <Typography
                      variant="h3"
                      component="span"
                      fontWeight="bold"
                      color="primary"
                    >
                      ${isAnnual ? Math.round(plan.annualPrice / 12) : plan.monthlyPrice}
                    </Typography>
                    <Typography variant="body1" component="span" color="text.secondary">
                      /month
                    </Typography>
                    {isAnnual && (
                      <Typography variant="body2" color="text.secondary">
                        Billed annually (${plan.annualPrice})
                      </Typography>
                    )}
                  </Box>
                  <List sx={{ textAlign: 'left' }}>
                    {plan.features.map((feature, featureIndex) => (
                      <ListItem key={featureIndex} sx={{ px: 0 }}>
                        <ListItemIcon sx={{ minWidth: 36 }}>
                          <CheckCircle sx={{ color: brandColors.success[500] }} />
                        </ListItemIcon>
                        <ListItemText primary={feature} />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
                <CardActions sx={{ p: 4, pt: 0 }}>
                  <Button
                    variant={plan.popular ? "contained" : "outlined"}
                    fullWidth
                    size="large"
                    onClick={() => handleSelectPlan(plan.name)}
                    sx={{
                      py: 1.5,
                      borderRadius: '12px',
                      fontWeight: 600,
                      ...(plan.popular && {
                        background: brandGradients.primary,
                        '&:hover': {
                          background: brandGradients.secondary
                        }
                      })
                    }}
                    endIcon={<ArrowForward />}
                  >
                    {plan.name === 'Enterprise' ? 'Contact Sales' : 'Start Free Trial'}
                  </Button>
                </CardActions>
              </PricingCard>
            </Grid>
          ))}
        </Grid>

        {/* ROI Calculator */}
        <ROICalculator sx={{ mb: 8 }}>
          <Typography variant="h4" fontWeight="bold" gutterBottom textAlign="center">
            Calculate Your ROI
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph textAlign="center">
            See how much AutoGuru Universal can save your business
          </Typography>

          <Grid container spacing={4} alignItems="center">
            <Grid item xs={12} md={6}>
              <Box sx={{ mb: 4 }}>
                <Typography variant="h6" gutterBottom>
                  Current Followers: {followers.toLocaleString()}
                </Typography>
                <Slider
                  value={followers}
                  onChange={(e, value) => setFollowers(value)}
                  min={1000}
                  max={1000000}
                  step={1000}
                  marks={[
                    { value: 1000, label: '1K' },
                    { value: 100000, label: '100K' },
                    { value: 1000000, label: '1M' }
                  ]}
                  sx={{ color: brandColors.primary[500] }}
                />
              </Box>
              <Box sx={{ mb: 4 }}>
                <Typography variant="h6" gutterBottom>
                  Posts per Week: {postsPerWeek}
                </Typography>
                <Slider
                  value={postsPerWeek}
                  onChange={(e, value) => setPostsPerWeek(value)}
                  min={1}
                  max={20}
                  step={1}
                  marks={[
                    { value: 1, label: '1' },
                    { value: 10, label: '10' },
                    { value: 20, label: '20' }
                  ]}
                  sx={{ color: brandColors.primary[500] }}
                />
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Paper sx={{ p: 2, textAlign: 'center', borderRadius: '12px' }}>
                    <Typography variant="h4" fontWeight="bold" color="primary">
                      {roiData.timeSaved}h
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Hours Saved/Month
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={6}>
                  <Paper sx={{ p: 2, textAlign: 'center', borderRadius: '12px' }}>
                    <Typography variant="h4" fontWeight="bold" color="success.main">
                      ${roiData.monthlySavings}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Monthly Savings
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={6}>
                  <Paper sx={{ p: 2, textAlign: 'center', borderRadius: '12px' }}>
                    <Typography variant="h4" fontWeight="bold" color="secondary.main">
                      +{roiData.engagementBoost}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Extra Engagements
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={6}>
                  <Paper sx={{ p: 2, textAlign: 'center', borderRadius: '12px' }}>
                    <Typography variant="h4" fontWeight="bold" color="warning.main">
                      {roiData.roi}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Monthly ROI
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </ROICalculator>

        {/* Feature Comparison Table */}
        <Box sx={{ mb: 8 }}>
          <Typography variant="h4" fontWeight="bold" gutterBottom textAlign="center">
            Feature Comparison
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph textAlign="center" mb={4}>
            Compare all features across our plans
          </Typography>

          <TableContainer component={Paper} sx={{ borderRadius: '16px' }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 'bold', fontSize: '1.1rem' }}>Features</TableCell>
                  <TableCell align="center" sx={{ fontWeight: 'bold', fontSize: '1.1rem' }}>Starter</TableCell>
                  <TableCell align="center" sx={{ fontWeight: 'bold', fontSize: '1.1rem', background: brandColors.primary[50] }}>
                    Professional
                  </TableCell>
                  <TableCell align="center" sx={{ fontWeight: 'bold', fontSize: '1.1rem' }}>Enterprise</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {features.map((category, categoryIndex) => (
                  <React.Fragment key={categoryIndex}>
                    <TableRow>
                      <TableCell
                        colSpan={4}
                        sx={{
                          background: brandColors.gray[50],
                          fontWeight: 'bold',
                          fontSize: '1rem'
                        }}
                      >
                        {category.category}
                      </TableCell>
                    </TableRow>
                    {category.items.map((item, itemIndex) => (
                      <TableRow key={itemIndex}>
                        <TableCell>{item.name}</TableCell>
                        <TableCell align="center">
                          {typeof item.starter === 'boolean' ? (
                            item.starter ? (
                              <CheckCircle sx={{ color: brandColors.success[500] }} />
                            ) : (
                              '—'
                            )
                          ) : (
                            item.starter
                          )}
                        </TableCell>
                        <TableCell align="center" sx={{ background: brandColors.primary[25] }}>
                          {typeof item.pro === 'boolean' ? (
                            item.pro ? (
                              <CheckCircle sx={{ color: brandColors.success[500] }} />
                            ) : (
                              '—'
                            )
                          ) : (
                            item.pro
                          )}
                        </TableCell>
                        <TableCell align="center">
                          {typeof item.enterprise === 'boolean' ? (
                            item.enterprise ? (
                              <CheckCircle sx={{ color: brandColors.success[500] }} />
                            ) : (
                              '—'
                            )
                          ) : (
                            item.enterprise
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </React.Fragment>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>

        {/* FAQ Section */}
        <Box>
          <Typography variant="h4" fontWeight="bold" gutterBottom textAlign="center">
            Frequently Asked Questions
          </Typography>
          <Typography variant="body1" color="text.secondary" paragraph textAlign="center" mb={4}>
            Everything you need to know about AutoGuru Universal
          </Typography>

          {faqs.map((faq, index) => (
            <Accordion key={index} sx={{ mb: 2, borderRadius: '12px', '&:before': { display: 'none' } }}>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6" fontWeight="600">
                  {faq.question}
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="body1" color="text.secondary">
                  {faq.answer}
                </Typography>
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>

        {/* Final CTA */}
        <Box sx={{ mt: 8, textAlign: 'center', p: 6, background: brandGradients.hero, borderRadius: '20px', color: 'white' }}>
          <Typography variant="h4" fontWeight="bold" gutterBottom>
            Ready to Transform Your Social Media?
          </Typography>
          <Typography variant="h6" paragraph sx={{ opacity: 0.9 }}>
            Start your 14-day free trial today. No credit card required.
          </Typography>
          <Button
            variant="contained"
            size="large"
            onClick={() => handleSelectPlan('Professional')}
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
            Start Free Trial
          </Button>
        </Box>
      </Container>
    </Box>
  );
};

export default Pricing;