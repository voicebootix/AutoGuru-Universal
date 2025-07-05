import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  TextField,
  Stepper,
  Step,
  StepLabel,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  Alert,
  LinearProgress,
  Chip,
  Divider,
  IconButton,
  InputAdornment,
  useTheme,
  styled
} from '@mui/material';
import {
  Google,
  Visibility,
  VisibilityOff,
  ArrowForward,
  CheckCircle,
  Business,
  Person,
  Lock,
  Email,
  Phone,
  FitnessCenter,
  BusinessCenter,
  Palette,
  School,
  Store,
  Engineering,
  Favorite,
  EmojiEmotions
} from '@mui/icons-material';
import { brandColors, brandGradients } from '../../assets/brand';

const SignupCard = styled(Card)(({ theme }) => ({
  maxWidth: 500,
  margin: '0 auto',
  borderRadius: '20px',
  background: 'rgba(255, 255, 255, 0.95)',
  backdropFilter: 'blur(10px)',
  border: '1px solid rgba(255, 255, 255, 0.2)',
  boxShadow: '0 20px 40px rgba(0, 0, 0, 0.1)'
}));

const BenefitChip = styled(Chip)(({ theme }) => ({
  margin: '4px',
  background: brandColors.success[50],
  color: brandColors.success[700],
  border: `1px solid ${brandColors.success[200]}`,
  '& .MuiChip-icon': {
    color: brandColors.success[500]
  }
}));

const BusinessTypeCard = styled(Card)(({ theme, selected }) => ({
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  border: selected ? `2px solid ${brandColors.primary[500]}` : '1px solid rgba(0, 0, 0, 0.1)',
  background: selected ? brandColors.primary[50] : 'white',
  '&:hover': {
    transform: 'translateY(-4px)',
    boxShadow: '0 8px 20px rgba(0, 0, 0, 0.1)',
    borderColor: brandColors.primary[300]
  }
}));

const Signup = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  
  // Get plan and billing info from navigation state
  const { plan = 'Professional', isAnnual = true } = location.state || {};
  
  const [activeStep, setActiveStep] = useState(0);
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const [formData, setFormData] = useState({
    // Step 1: Account
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: '',
    phone: '',
    
    // Step 2: Business
    businessName: '',
    businessType: '',
    businessSize: '',
    website: '',
    description: '',
    
    // Step 3: Goals
    goals: [],
    currentChallenges: [],
    monthlyBudget: '',
    expectedGrowth: '',
    
    // Plan selection
    selectedPlan: plan,
    billingCycle: isAnnual ? 'annual' : 'monthly'
  });

  const steps = ['Account', 'Business', 'Goals'];
  
  const businessTypes = [
    { id: 'fitness', name: 'Fitness & Wellness', icon: <FitnessCenter />, description: 'Gyms, trainers, wellness coaches' },
    { id: 'consulting', name: 'Business Consulting', icon: <BusinessCenter />, description: 'Business coaches, consultants' },
    { id: 'creative', name: 'Creative Services', icon: <Palette />, description: 'Designers, artists, photographers' },
    { id: 'education', name: 'Education', icon: <School />, description: 'Courses, tutoring, coaching' },
    { id: 'ecommerce', name: 'E-commerce', icon: <Store />, description: 'Online stores, retail' },
    { id: 'technology', name: 'Technology', icon: <Engineering />, description: 'SaaS, tech companies' },
    { id: 'nonprofit', name: 'Non-profit', icon: <Favorite />, description: 'Charitable organizations' },
    { id: 'other', name: 'Other', icon: <EmojiEmotions />, description: 'AI will adapt to your niche' }
  ];

  const goals = [
    'Increase followers',
    'Boost engagement',
    'Generate leads',
    'Drive sales',
    'Build brand awareness',
    'Save time on content creation',
    'Improve content quality',
    'Scale social media presence'
  ];

  const challenges = [
    'No time for content creation',
    'Poor engagement rates',
    'Inconsistent posting',
    'Lack of viral content ideas',
    'Managing multiple platforms',
    'Measuring ROI',
    'Creating quality visuals',
    'Understanding algorithms'
  ];

  const benefits = [
    'AI creates content for your specific niche',
    'Works across all major platforms',
    '14-day free trial',
    'No setup fees',
    'Cancel anytime',
    '300% average engagement increase'
  ];

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({
        ...prev,
        [field]: ''
      }));
    }
  };

  const handleMultiSelectChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: prev[field].includes(value)
        ? prev[field].filter(item => item !== value)
        : [...prev[field], value]
    }));
  };

  const validateStep = (step) => {
    const newErrors = {};
    
    switch (step) {
      case 0: // Account
        if (!formData.firstName.trim()) newErrors.firstName = 'First name is required';
        if (!formData.lastName.trim()) newErrors.lastName = 'Last name is required';
        if (!formData.email.trim()) newErrors.email = 'Email is required';
        else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) newErrors.email = 'Invalid email format';
        if (!formData.password) newErrors.password = 'Password is required';
        else if (formData.password.length < 8) newErrors.password = 'Password must be at least 8 characters';
        if (formData.password !== formData.confirmPassword) newErrors.confirmPassword = 'Passwords do not match';
        break;
        
      case 1: // Business
        if (!formData.businessName.trim()) newErrors.businessName = 'Business name is required';
        if (!formData.businessType) newErrors.businessType = 'Business type is required';
        if (!formData.businessSize) newErrors.businessSize = 'Business size is required';
        break;
        
      case 2: // Goals
        if (formData.goals.length === 0) newErrors.goals = 'Please select at least one goal';
        if (formData.currentChallenges.length === 0) newErrors.currentChallenges = 'Please select at least one challenge';
        break;
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleNext = () => {
    if (validateStep(activeStep)) {
      setActiveStep(prev => prev + 1);
    }
  };

  const handleBack = () => {
    setActiveStep(prev => prev - 1);
  };

  const handleSubmit = async () => {
    if (!validateStep(activeStep)) return;
    
    setLoading(true);
    
    try {
      // TODO: Implement actual signup API call
      const signupData = {
        ...formData,
        plan: formData.selectedPlan,
        billing: formData.billingCycle
      };
      
      console.log('Signup data:', signupData);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Navigate to onboarding or dashboard
      navigate('/onboarding', { state: { signupData } });
      
    } catch (error) {
      setErrors({ submit: 'Signup failed. Please try again.' });
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignup = () => {
    // TODO: Implement Google OAuth
    console.log('Google signup clicked');
  };

  const renderAccountStep = () => (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        Create Your Account
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Let's get started with your AutoGuru Universal account
      </Typography>
      
      <Button
        fullWidth
        variant="outlined"
        size="large"
        onClick={handleGoogleSignup}
        startIcon={<Google />}
        sx={{
          mb: 3,
          py: 1.5,
          borderRadius: '12px',
          borderColor: brandColors.gray[300],
          '&:hover': {
            borderColor: brandColors.primary[500],
            background: brandColors.primary[50]
          }
        }}
      >
        Sign up with Google
      </Button>
      
      <Divider sx={{ my: 3 }}>
        <Typography variant="body2" color="text.secondary">or</Typography>
      </Divider>
      
      <Grid container spacing={2}>
        <Grid item xs={6}>
          <TextField
            fullWidth
            label="First Name"
            value={formData.firstName}
            onChange={(e) => handleInputChange('firstName', e.target.value)}
            error={!!errors.firstName}
            helperText={errors.firstName}
            sx={{ mb: 2 }}
          />
        </Grid>
        <Grid item xs={6}>
          <TextField
            fullWidth
            label="Last Name"
            value={formData.lastName}
            onChange={(e) => handleInputChange('lastName', e.target.value)}
            error={!!errors.lastName}
            helperText={errors.lastName}
            sx={{ mb: 2 }}
          />
        </Grid>
      </Grid>
      
      <TextField
        fullWidth
        label="Email Address"
        type="email"
        value={formData.email}
        onChange={(e) => handleInputChange('email', e.target.value)}
        error={!!errors.email}
        helperText={errors.email}
        sx={{ mb: 2 }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <Email color="action" />
            </InputAdornment>
          )
        }}
      />
      
      <TextField
        fullWidth
        label="Password"
        type={showPassword ? 'text' : 'password'}
        value={formData.password}
        onChange={(e) => handleInputChange('password', e.target.value)}
        error={!!errors.password}
        helperText={errors.password}
        sx={{ mb: 2 }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <Lock color="action" />
            </InputAdornment>
          ),
          endAdornment: (
            <InputAdornment position="end">
              <IconButton onClick={() => setShowPassword(!showPassword)}>
                {showPassword ? <VisibilityOff /> : <Visibility />}
              </IconButton>
            </InputAdornment>
          )
        }}
      />
      
      <TextField
        fullWidth
        label="Confirm Password"
        type="password"
        value={formData.confirmPassword}
        onChange={(e) => handleInputChange('confirmPassword', e.target.value)}
        error={!!errors.confirmPassword}
        helperText={errors.confirmPassword}
        sx={{ mb: 2 }}
      />
      
      <TextField
        fullWidth
        label="Phone Number (Optional)"
        value={formData.phone}
        onChange={(e) => handleInputChange('phone', e.target.value)}
        sx={{ mb: 3 }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <Phone color="action" />
            </InputAdornment>
          )
        }}
      />
    </Box>
  );

  const renderBusinessStep = () => (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        Tell Us About Your Business
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        This helps our AI create better content for your specific industry
      </Typography>
      
      <TextField
        fullWidth
        label="Business Name"
        value={formData.businessName}
        onChange={(e) => handleInputChange('businessName', e.target.value)}
        error={!!errors.businessName}
        helperText={errors.businessName}
        sx={{ mb: 3 }}
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <Business color="action" />
            </InputAdornment>
          )
        }}
      />
      
      <Typography variant="h6" fontWeight="600" gutterBottom>
        What type of business do you have?
      </Typography>
      {errors.businessType && (
        <Alert severity="error" sx={{ mb: 2 }}>{errors.businessType}</Alert>
      )}
      
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {businessTypes.map((type) => (
          <Grid item xs={12} sm={6} key={type.id}>
            <BusinessTypeCard
              selected={formData.businessType === type.id}
              onClick={() => handleInputChange('businessType', type.id)}
            >
              <CardContent sx={{ p: 2, textAlign: 'center' }}>
                <Box sx={{ color: brandColors.primary[500], mb: 1 }}>
                  {React.cloneElement(type.icon, { sx: { fontSize: 32 } })}
                </Box>
                <Typography variant="subtitle1" fontWeight="600">
                  {type.name}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {type.description}
                </Typography>
              </CardContent>
            </BusinessTypeCard>
          </Grid>
        ))}
      </Grid>
      
      <Grid container spacing={2}>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Business Size</InputLabel>
            <Select
              value={formData.businessSize}
              onChange={(e) => handleInputChange('businessSize', e.target.value)}
              error={!!errors.businessSize}
            >
              <MenuItem value="solo">Just me</MenuItem>
              <MenuItem value="small">2-10 employees</MenuItem>
              <MenuItem value="medium">11-50 employees</MenuItem>
              <MenuItem value="large">50+ employees</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={6}>
          <TextField
            fullWidth
            label="Website (Optional)"
            value={formData.website}
            onChange={(e) => handleInputChange('website', e.target.value)}
            sx={{ mb: 2 }}
          />
        </Grid>
      </Grid>
      
      <TextField
        fullWidth
        label="Business Description (Optional)"
        multiline
        rows={3}
        value={formData.description}
        onChange={(e) => handleInputChange('description', e.target.value)}
        placeholder="Tell us more about your business to help our AI create better content..."
        sx={{ mb: 2 }}
      />
    </Box>
  );

  const renderGoalsStep = () => (
    <Box>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        What Are Your Goals?
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Help us customize your experience based on what you want to achieve
      </Typography>
      
      <Typography variant="h6" fontWeight="600" gutterBottom>
        What do you want to achieve? (Select all that apply)
      </Typography>
      {errors.goals && (
        <Alert severity="error" sx={{ mb: 2 }}>{errors.goals}</Alert>
      )}
      
      <Box sx={{ mb: 4 }}>
        {goals.map((goal, index) => (
          <BenefitChip
            key={index}
            icon={formData.goals.includes(goal) ? <CheckCircle /> : undefined}
            label={goal}
            onClick={() => handleMultiSelectChange('goals', goal)}
            variant={formData.goals.includes(goal) ? 'filled' : 'outlined'}
            sx={{ 
              m: 0.5,
              ...(formData.goals.includes(goal) && {
                background: brandColors.primary[100],
                color: brandColors.primary[800]
              })
            }}
          />
        ))}
      </Box>
      
      <Typography variant="h6" fontWeight="600" gutterBottom>
        What are your current challenges? (Select all that apply)
      </Typography>
      {errors.currentChallenges && (
        <Alert severity="error" sx={{ mb: 2 }}>{errors.currentChallenges}</Alert>
      )}
      
      <Box sx={{ mb: 4 }}>
        {challenges.map((challenge, index) => (
          <BenefitChip
            key={index}
            icon={formData.currentChallenges.includes(challenge) ? <CheckCircle /> : undefined}
            label={challenge}
            onClick={() => handleMultiSelectChange('currentChallenges', challenge)}
            variant={formData.currentChallenges.includes(challenge) ? 'filled' : 'outlined'}
            sx={{ 
              m: 0.5,
              ...(formData.currentChallenges.includes(challenge) && {
                background: brandColors.secondary[100],
                color: brandColors.secondary[800]
              })
            }}
          />
        ))}
      </Box>
      
      <Grid container spacing={2}>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Monthly Marketing Budget</InputLabel>
            <Select
              value={formData.monthlyBudget}
              onChange={(e) => handleInputChange('monthlyBudget', e.target.value)}
            >
              <MenuItem value="under-500">Under $500</MenuItem>
              <MenuItem value="500-1000">$500 - $1,000</MenuItem>
              <MenuItem value="1000-2500">$1,000 - $2,500</MenuItem>
              <MenuItem value="2500-5000">$2,500 - $5,000</MenuItem>
              <MenuItem value="over-5000">Over $5,000</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Expected Growth</InputLabel>
            <Select
              value={formData.expectedGrowth}
              onChange={(e) => handleInputChange('expectedGrowth', e.target.value)}
            >
              <MenuItem value="slow">Steady growth (10-25%)</MenuItem>
              <MenuItem value="moderate">Moderate growth (25-50%)</MenuItem>
              <MenuItem value="fast">Fast growth (50-100%)</MenuItem>
              <MenuItem value="explosive">Explosive growth (100%+)</MenuItem>
            </Select>
          </FormControl>
        </Grid>
      </Grid>
    </Box>
  );

  return (
    <Box
      sx={{
        minHeight: '100vh',
        background: `linear-gradient(135deg, ${brandColors.primary[50]} 0%, ${brandColors.secondary[50]} 100%)`,
        py: 4
      }}
    >
      <Container maxWidth="md">
        <Box textAlign="center" mb={4}>
          <Typography variant="h3" fontWeight="bold" gutterBottom>
            Start Your Free Trial
          </Typography>
          <Typography variant="h6" color="text.secondary">
            Selected Plan: {formData.selectedPlan} ({formData.billingCycle})
          </Typography>
        </Box>

        {/* Benefits Banner */}
        <Box sx={{ textAlign: 'center', mb: 4 }}>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', gap: 1 }}>
            {benefits.map((benefit, index) => (
              <BenefitChip
                key={index}
                icon={<CheckCircle />}
                label={benefit}
                size="small"
              />
            ))}
          </Box>
        </Box>

        <SignupCard>
          <CardContent sx={{ p: 4 }}>
            {/* Stepper */}
            <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
              {steps.map((label) => (
                <Step key={label}>
                  <StepLabel>{label}</StepLabel>
                </Step>
              ))}
            </Stepper>

            {/* Progress Bar */}
            <LinearProgress
              variant="determinate"
              value={(activeStep / (steps.length - 1)) * 100}
              sx={{
                mb: 4,
                height: 8,
                borderRadius: 4,
                backgroundColor: brandColors.gray[200],
                '& .MuiLinearProgress-bar': {
                  background: brandGradients.primary,
                  borderRadius: 4
                }
              }}
            />

            {/* Step Content */}
            {activeStep === 0 && renderAccountStep()}
            {activeStep === 1 && renderBusinessStep()}
            {activeStep === 2 && renderGoalsStep()}

            {/* Error Message */}
            {errors.submit && (
              <Alert severity="error" sx={{ mb: 3 }}>
                {errors.submit}
              </Alert>
            )}

            {/* Navigation Buttons */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
              <Button
                onClick={handleBack}
                disabled={activeStep === 0}
                sx={{ visibility: activeStep === 0 ? 'hidden' : 'visible' }}
              >
                Back
              </Button>
              
              <Button
                variant="contained"
                onClick={activeStep === steps.length - 1 ? handleSubmit : handleNext}
                disabled={loading}
                sx={{
                  background: brandGradients.primary,
                  px: 4,
                  py: 1.5,
                  borderRadius: '12px',
                  fontWeight: 600,
                  '&:hover': {
                    background: brandGradients.secondary
                  }
                }}
                endIcon={loading ? null : <ArrowForward />}
              >
                {loading ? 'Creating Account...' : activeStep === steps.length - 1 ? 'Start Free Trial' : 'Continue'}
              </Button>
            </Box>

            {/* Terms */}
            <Typography variant="body2" color="text.secondary" textAlign="center" sx={{ mt: 3 }}>
              By signing up, you agree to our{' '}
              <Button color="primary" sx={{ p: 0, textDecoration: 'underline' }}>
                Terms of Service
              </Button>{' '}
              and{' '}
              <Button color="primary" sx={{ p: 0, textDecoration: 'underline' }}>
                Privacy Policy
              </Button>
            </Typography>
          </CardContent>
        </SignupCard>
      </Container>
    </Box>
  );
};

export default Signup;