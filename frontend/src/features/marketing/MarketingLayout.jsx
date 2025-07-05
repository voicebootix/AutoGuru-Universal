import React from 'react';
import { Outlet } from 'react-router-dom';
import { Box, Container, Typography, Grid, Link, Divider, IconButton } from '@mui/material';
import { LinkedIn, Twitter, Facebook, Instagram, YouTube } from '@mui/icons-material';
import MarketingHeader from './MarketingHeader';
import { brandColors, brandGradients } from '../../assets/brand';

const MarketingLayout = () => {
  const currentYear = new Date().getFullYear();

  const footerLinks = {
    Product: [
      { name: 'Features', href: '/features' },
      { name: 'Pricing', href: '/pricing' },
      { name: 'Use Cases', href: '/use-cases' },
      { name: 'API', href: '/api' },
      { name: 'Integrations', href: '/integrations' }
    ],
    Resources: [
      { name: 'Blog', href: '/blog' },
      { name: 'Help Center', href: '/help' },
      { name: 'Tutorials', href: '/tutorials' },
      { name: 'Templates', href: '/templates' },
      { name: 'Webinars', href: '/webinars' }
    ],
    Company: [
      { name: 'About', href: '/about' },
      { name: 'Careers', href: '/careers' },
      { name: 'Contact', href: '/contact' },
      { name: 'Partners', href: '/partners' },
      { name: 'Press Kit', href: '/press' }
    ],
    Legal: [
      { name: 'Privacy Policy', href: '/privacy' },
      { name: 'Terms of Service', href: '/terms' },
      { name: 'Cookie Policy', href: '/cookies' },
      { name: 'Security', href: '/security' },
      { name: 'GDPR', href: '/gdpr' }
    ]
  };

  const socialLinks = [
    { name: 'LinkedIn', icon: <LinkedIn />, href: 'https://linkedin.com/company/autoguru-universal' },
    { name: 'Twitter', icon: <Twitter />, href: 'https://twitter.com/autoguruai' },
    { name: 'Facebook', icon: <Facebook />, href: 'https://facebook.com/autoguru.universal' },
    { name: 'Instagram', icon: <Instagram />, href: 'https://instagram.com/autoguru.universal' },
    { name: 'YouTube', icon: <YouTube />, href: 'https://youtube.com/@autoguruuniversal' }
  ];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* Header */}
      <MarketingHeader />
      
      {/* Main Content */}
      <Box component="main" sx={{ flexGrow: 1 }}>
        <Outlet />
      </Box>
      
      {/* Footer */}
      <Box
        component="footer"
        sx={{
          background: brandColors.dark[900],
          color: 'white',
          py: 6,
          mt: 8
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={4}>
            {/* Company Info */}
            <Grid item xs={12} md={4}>
              <Box sx={{ mb: 3 }}>
                <Box
                  sx={{
                    width: 48,
                    height: 48,
                    borderRadius: '50%',
                    background: brandGradients.hero,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    mb: 2
                  }}
                >
                  <Typography
                    variant="h5"
                    sx={{
                      color: 'white',
                      fontWeight: 'bold'
                    }}
                  >
                    AG
                  </Typography>
                </Box>
                <Typography
                  variant="h6"
                  sx={{
                    fontWeight: 700,
                    mb: 1,
                    background: brandGradients.hero,
                    backgroundClip: 'text',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent'
                  }}
                >
                  AutoGuru Universal
                </Typography>
                <Typography variant="body2" color="gray.400" paragraph>
                  Universal social media automation for ANY business niche. AI-powered content creation that adapts to your industry automatically.
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  {socialLinks.map((social) => (
                    <IconButton
                      key={social.name}
                      href={social.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      sx={{
                        color: brandColors.gray[400],
                        '&:hover': {
                          color: 'white',
                          background: brandColors.primary[600]
                        }
                      }}
                    >
                      {social.icon}
                    </IconButton>
                  ))}
                </Box>
              </Box>
            </Grid>

            {/* Footer Links */}
            {Object.entries(footerLinks).map(([category, links]) => (
              <Grid item xs={6} sm={3} md={2} key={category}>
                <Typography variant="h6" fontWeight="600" gutterBottom>
                  {category}
                </Typography>
                <Box>
                  {links.map((link) => (
                    <Link
                      key={link.name}
                      href={link.href}
                      sx={{
                        display: 'block',
                        color: brandColors.gray[400],
                        textDecoration: 'none',
                        mb: 1,
                        fontSize: '0.9rem',
                        '&:hover': {
                          color: 'white'
                        }
                      }}
                    >
                      {link.name}
                    </Link>
                  ))}
                </Box>
              </Grid>
            ))}
          </Grid>

          <Divider sx={{ my: 4, borderColor: brandColors.gray[700] }} />

          {/* Bottom Footer */}
          <Box
            sx={{
              display: 'flex',
              flexDirection: { xs: 'column', sm: 'row' },
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: 2
            }}
          >
            <Typography variant="body2" color="gray.400">
              © {currentYear} AutoGuru Universal. All rights reserved.
            </Typography>
            <Box sx={{ display: 'flex', gap: 3 }}>
              <Typography variant="body2" color="gray.400">
                Made with ❤️ for businesses of all types
              </Typography>
            </Box>
          </Box>
        </Container>
      </Box>
    </Box>
  );
};

export default MarketingLayout;