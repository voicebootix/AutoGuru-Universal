// AutoGuru Universal Brand Assets & Design System

export const brandColors = {
  primary: {
    50: '#f0f9ff',
    100: '#e0f2fe',
    200: '#bae6fd',
    300: '#7dd3fc',
    400: '#38bdf8',
    500: '#0ea5e9', // Main brand color
    600: '#0284c7',
    700: '#0369a1',
    800: '#075985',
    900: '#0c4a6e',
  },
  secondary: {
    50: '#fdf4ff',
    100: '#fae8ff',
    200: '#f5d0fe',
    300: '#f0abfc',
    400: '#e879f9',
    500: '#d946ef', // Secondary accent
    600: '#c026d3',
    700: '#a21caf',
    800: '#86198f',
    900: '#701a75',
  },
  success: {
    50: '#f0fdf4',
    100: '#dcfce7',
    200: '#bbf7d0',
    300: '#86efac',
    400: '#4ade80',
    500: '#22c55e',
    600: '#16a34a',
    700: '#15803d',
    800: '#166534',
    900: '#14532d',
  },
  warning: {
    50: '#fffbeb',
    100: '#fef3c7',
    200: '#fde68a',
    300: '#fcd34d',
    400: '#fbbf24',
    500: '#f59e0b',
    600: '#d97706',
    700: '#b45309',
    800: '#92400e',
    900: '#78350f',
  },
  error: {
    50: '#fef2f2',
    100: '#fee2e2',
    200: '#fecaca',
    300: '#fca5a5',
    400: '#f87171',
    500: '#ef4444',
    600: '#dc2626',
    700: '#b91c1c',
    800: '#991b1b',
    900: '#7f1d1d',
  },
  gray: {
    50: '#f9fafb',
    100: '#f3f4f6',
    200: '#e5e7eb',
    300: '#d1d5db',
    400: '#9ca3af',
    500: '#6b7280',
    600: '#4b5563',
    700: '#374151',
    800: '#1f2937',
    900: '#111827',
  },
  dark: {
    50: '#f8fafc',
    100: '#f1f5f9',
    200: '#e2e8f0',
    300: '#cbd5e1',
    400: '#94a3b8',
    500: '#64748b',
    600: '#475569',
    700: '#334155',
    800: '#1e293b',
    900: '#0f172a',
  }
};

export const brandFonts = {
  primary: {
    name: 'Inter',
    import: '@import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap");',
    fallback: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
  },
  secondary: {
    name: 'Poppins',
    import: '@import url("https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap");',
    fallback: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
  },
  mono: {
    name: 'JetBrains Mono',
    import: '@import url("https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&display=swap");',
    fallback: 'ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace'
  }
};

export const brandGradients = {
  primary: 'linear-gradient(135deg, #0ea5e9 0%, #d946ef 100%)',
  secondary: 'linear-gradient(135deg, #d946ef 0%, #f59e0b 100%)',
  success: 'linear-gradient(135deg, #22c55e 0%, #0ea5e9 100%)',
  dark: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
  light: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)',
  hero: 'linear-gradient(135deg, #0ea5e9 0%, #d946ef 50%, #f59e0b 100%)',
  card: 'linear-gradient(145deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%)'
};

export const brandShadows = {
  sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
  md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
  lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
  xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
  '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
  inner: 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.06)',
  glow: '0 0 20px rgba(14, 165, 233, 0.3)',
  glowSecondary: '0 0 20px rgba(217, 70, 239, 0.3)'
};

export const brandSpacing = {
  section: '80px',
  container: '120px',
  element: '24px',
  tight: '16px',
  loose: '32px'
};

export const brandBreakpoints = {
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px'
};

export const brandAnimations = {
  fadeIn: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6 }
  },
  slideIn: {
    initial: { opacity: 0, x: -50 },
    animate: { opacity: 1, x: 0 },
    transition: { duration: 0.6 }
  },
  scaleIn: {
    initial: { opacity: 0, scale: 0.8 },
    animate: { opacity: 1, scale: 1 },
    transition: { duration: 0.6 }
  },
  stagger: {
    animate: {
      transition: {
        staggerChildren: 0.1
      }
    }
  }
};

export const brandIcons = {
  logo: `<svg viewBox="0 0 200 60" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:#0ea5e9;stop-opacity:1" />
        <stop offset="50%" style="stop-color:#d946ef;stop-opacity:1" />
        <stop offset="100%" style="stop-color:#f59e0b;stop-opacity:1" />
      </linearGradient>
    </defs>
    <circle cx="30" cy="30" r="25" fill="url(#logoGradient)" />
    <path d="M20 20 L25 30 L35 30 L40 20 M25 30 L30 40 L35 30" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
    <text x="70" y="25" font-family="Inter, sans-serif" font-size="18" font-weight="700" fill="#1e293b">AutoGuru</text>
    <text x="70" y="45" font-family="Inter, sans-serif" font-size="12" font-weight="500" fill="#64748b">Universal</text>
  </svg>`,
  
  logoSmall: `<svg viewBox="0 0 60 60" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="logoGradientSmall" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:#0ea5e9;stop-opacity:1" />
        <stop offset="50%" style="stop-color:#d946ef;stop-opacity:1" />
        <stop offset="100%" style="stop-color:#f59e0b;stop-opacity:1" />
      </linearGradient>
    </defs>
    <circle cx="30" cy="30" r="25" fill="url(#logoGradientSmall)" />
    <path d="M20 20 L25 30 L35 30 L40 20 M25 30 L30 40 L35 30" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
  </svg>`
};

export const brandContent = {
  tagline: "Universal social media automation for ANY business niche",
  hero: {
    title: "Stop Wrestling with Social Media. Start Growing Your Business.",
    subtitle: "AutoGuru Universal creates viral content for ANY business niche automatically. Fitness coaches. Business consultants. Artists. E-commerce. Anyone.",
    cta: "Start Free Trial"
  },
  features: {
    title: "Why AutoGuru Universal?",
    subtitle: "The only AI-powered social media platform that works for every business type automatically"
  },
  social: {
    title: "Join 10,000+ businesses growing with AutoGuru Universal",
    subtitle: "From fitness coaches to Fortune 500 companies, everyone's scaling their social media presence"
  }
};

export const brandStats = {
  businesses: "10,000+",
  posts: "1M+",
  platforms: "6",
  countries: "50+"
};

export const brandTestimonials = [
  {
    name: "Sarah Johnson",
    role: "Fitness Coach",
    company: "FitLife Studio",
    image: "/api/placeholder/100/100",
    content: "AutoGuru Universal transformed my Instagram from 2K to 50K followers in 6 months. The AI creates content that my fitness audience actually engages with.",
    rating: 5
  },
  {
    name: "Michael Chen",
    role: "Business Consultant",
    company: "Growth Strategies Inc",
    image: "/api/placeholder/100/100",
    content: "Finally, a platform that understands B2B content. My LinkedIn posts now generate 10x more leads than before.",
    rating: 5
  },
  {
    name: "Emily Rodriguez",
    role: "Creative Director",
    company: "Pixel Perfect Design",
    image: "/api/placeholder/100/100",
    content: "The AI understands creative industry nuances. My portfolio posts on Instagram and Behance get incredible engagement.",
    rating: 5
  }
];

export default {
  colors: brandColors,
  fonts: brandFonts,
  gradients: brandGradients,
  shadows: brandShadows,
  spacing: brandSpacing,
  breakpoints: brandBreakpoints,
  animations: brandAnimations,
  icons: brandIcons,
  content: brandContent,
  stats: brandStats,
  testimonials: brandTestimonials
};