import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  Container,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemText,
  useTheme,
  useMediaQuery,
  styled
} from '@mui/material';
import {
  Menu as MenuIcon,
  Close as CloseIcon,
  ArrowForward,
  Login
} from '@mui/icons-material';
import { brandColors, brandGradients } from '../../assets/brand';

const MarketingAppBar = styled(AppBar)(({ theme }) => ({
  background: 'rgba(255, 255, 255, 0.95)',
  backdropFilter: 'blur(10px)',
  borderBottom: '1px solid rgba(0, 0, 0, 0.1)',
  boxShadow: '0 2px 20px rgba(0, 0, 0, 0.05)',
  color: brandColors.gray[800]
}));

const Logo = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  cursor: 'pointer',
  '&:hover': {
    opacity: 0.8
  }
}));

const NavButton = styled(Button)(({ theme }) => ({
  color: brandColors.gray[600],
  fontWeight: 500,
  fontSize: '1rem',
  textTransform: 'none',
  padding: '8px 16px',
  borderRadius: '8px',
  '&:hover': {
    background: brandColors.gray[50],
    color: brandColors.primary[600]
  }
}));

const CTAButton = styled(Button)(({ theme }) => ({
  background: brandGradients.primary,
  color: 'white',
  fontWeight: 600,
  textTransform: 'none',
  padding: '10px 24px',
  borderRadius: '12px',
  '&:hover': {
    background: brandGradients.secondary,
    transform: 'translateY(-1px)',
    boxShadow: '0 4px 12px rgba(14, 165, 233, 0.3)'
  }
}));

const MarketingHeader = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navigationItems = [
    { label: 'Features', path: '/features' },
    { label: 'Pricing', path: '/pricing' },
    { label: 'Use Cases', path: '/use-cases' },
    { label: 'Resources', path: '/resources' },
    { label: 'About', path: '/about' }
  ];

  const handleNavigate = (path) => {
    navigate(path);
    setMobileMenuOpen(false);
  };

  const handleLogin = () => {
    navigate('/login');
  };

  const handleGetStarted = () => {
    navigate('/signup');
  };

  const LogoComponent = () => (
    <Logo onClick={() => handleNavigate('/')}>
      <Box
        sx={{
          width: 40,
          height: 40,
          borderRadius: '50%',
          background: brandGradients.hero,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          mr: 2
        }}
      >
        <Typography
          variant="h6"
          sx={{
            color: 'white',
            fontWeight: 'bold',
            fontSize: '1.2rem'
          }}
        >
          AG
        </Typography>
      </Box>
      <Box>
        <Typography
          variant="h6"
          sx={{
            fontWeight: 700,
            fontSize: '1.4rem',
            background: brandGradients.hero,
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}
        >
          AutoGuru
        </Typography>
        <Typography
          variant="caption"
          sx={{
            color: brandColors.gray[500],
            fontWeight: 500,
            fontSize: '0.75rem',
            lineHeight: 1,
            display: 'block',
            mt: -0.5
          }}
        >
          Universal
        </Typography>
      </Box>
    </Logo>
  );

  const DesktopNavigation = () => (
    <Box sx={{ display: { xs: 'none', md: 'flex' }, alignItems: 'center', gap: 1 }}>
      {navigationItems.map((item) => (
        <NavButton
          key={item.label}
          onClick={() => handleNavigate(item.path)}
          sx={{
            ...(location.pathname === item.path && {
              color: brandColors.primary[600],
              background: brandColors.primary[50]
            })
          }}
        >
          {item.label}
        </NavButton>
      ))}
    </Box>
  );

  const DesktopActions = () => (
    <Box sx={{ display: { xs: 'none', md: 'flex' }, alignItems: 'center', gap: 2 }}>
      <Button
        variant="text"
        onClick={handleLogin}
        startIcon={<Login />}
        sx={{
          color: brandColors.gray[600],
          fontWeight: 500,
          textTransform: 'none',
          '&:hover': {
            background: brandColors.gray[50]
          }
        }}
      >
        Sign In
      </Button>
      <CTAButton
        onClick={handleGetStarted}
        endIcon={<ArrowForward />}
      >
        Start Free Trial
      </CTAButton>
    </Box>
  );

  const MobileMenu = () => (
    <Drawer
      anchor="right"
      open={mobileMenuOpen}
      onClose={() => setMobileMenuOpen(false)}
      PaperProps={{
        sx: {
          width: 280,
          background: 'white'
        }
      }}
    >
      <Box sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <LogoComponent />
          <IconButton onClick={() => setMobileMenuOpen(false)}>
            <CloseIcon />
          </IconButton>
        </Box>
        
        <List>
          {navigationItems.map((item) => (
            <ListItem
              key={item.label}
              button
              onClick={() => handleNavigate(item.path)}
              sx={{
                borderRadius: '8px',
                mb: 1,
                ...(location.pathname === item.path && {
                  background: brandColors.primary[50],
                  color: brandColors.primary[600]
                })
              }}
            >
              <ListItemText
                primary={item.label}
                primaryTypographyProps={{
                  fontWeight: 500
                }}
              />
            </ListItem>
          ))}
        </List>

        <Box sx={{ mt: 4, px: 2 }}>
          <Button
            fullWidth
            variant="outlined"
            onClick={handleLogin}
            startIcon={<Login />}
            sx={{
              mb: 2,
              py: 1.5,
              borderRadius: '12px',
              textTransform: 'none',
              fontWeight: 500
            }}
          >
            Sign In
          </Button>
          <CTAButton
            fullWidth
            onClick={handleGetStarted}
            endIcon={<ArrowForward />}
            sx={{ py: 1.5 }}
          >
            Start Free Trial
          </CTAButton>
        </Box>
      </Box>
    </Drawer>
  );

  return (
    <>
      <MarketingAppBar position="fixed" elevation={0}>
        <Container maxWidth="lg">
          <Toolbar sx={{ justifyContent: 'space-between', py: 1 }}>
            <LogoComponent />
            
            {!isMobile && <DesktopNavigation />}
            
            {!isMobile && <DesktopActions />}
            
            {isMobile && (
              <IconButton
                onClick={() => setMobileMenuOpen(true)}
                sx={{ color: brandColors.gray[600] }}
              >
                <MenuIcon />
              </IconButton>
            )}
          </Toolbar>
        </Container>
      </MarketingAppBar>
      
      <MobileMenu />
      
      {/* Add top spacing for fixed header */}
      <Toolbar />
    </>
  );
};

export default MarketingHeader;