// Role-based access control utilities for AutoGuru Universal

export const USER_ROLES = {
  REGULAR: 'regular',
  BUSINESS_OWNER: 'business_owner', 
  ADMIN: 'admin',
  SUPER_ADMIN: 'super_admin'
};

export const PLAN_TYPES = {
  FREE: 'free',
  STARTER: 'starter',
  PROFESSIONAL: 'professional',
  ENTERPRISE: 'enterprise'
};

// Define feature access based on roles and plans
export const FEATURE_ACCESS = {
  // Main features - available to all authenticated users
  dashboard: {
    roles: [USER_ROLES.REGULAR, USER_ROLES.BUSINESS_OWNER, USER_ROLES.ADMIN, USER_ROLES.SUPER_ADMIN],
    plans: [PLAN_TYPES.FREE, PLAN_TYPES.STARTER, PLAN_TYPES.PROFESSIONAL, PLAN_TYPES.ENTERPRISE]
  },
  
  analytics: {
    roles: [USER_ROLES.REGULAR, USER_ROLES.BUSINESS_OWNER, USER_ROLES.ADMIN, USER_ROLES.SUPER_ADMIN],
    plans: [PLAN_TYPES.STARTER, PLAN_TYPES.PROFESSIONAL, PLAN_TYPES.ENTERPRISE]
  },
  
  content: {
    roles: [USER_ROLES.REGULAR, USER_ROLES.BUSINESS_OWNER, USER_ROLES.ADMIN, USER_ROLES.SUPER_ADMIN],
    plans: [PLAN_TYPES.FREE, PLAN_TYPES.STARTER, PLAN_TYPES.PROFESSIONAL, PLAN_TYPES.ENTERPRISE]
  },
  
  // Revenue features - business focused
  revenue_tracking: {
    roles: [USER_ROLES.BUSINESS_OWNER, USER_ROLES.ADMIN, USER_ROLES.SUPER_ADMIN],
    plans: [PLAN_TYPES.PROFESSIONAL, PLAN_TYPES.ENTERPRISE]
  },
  
  advertising: {
    roles: [USER_ROLES.BUSINESS_OWNER, USER_ROLES.ADMIN, USER_ROLES.SUPER_ADMIN],
    plans: [PLAN_TYPES.PROFESSIONAL, PLAN_TYPES.ENTERPRISE]
  },
  
  // AI features - premium only
  ai_insights: {
    roles: [USER_ROLES.BUSINESS_OWNER, USER_ROLES.ADMIN, USER_ROLES.SUPER_ADMIN],
    plans: [PLAN_TYPES.PROFESSIONAL, PLAN_TYPES.ENTERPRISE]
  },
  
  // Admin features - admin only
  admin_tools: {
    roles: [USER_ROLES.ADMIN, USER_ROLES.SUPER_ADMIN],
    plans: [PLAN_TYPES.ENTERPRISE] // Or admin accounts
  },
  
  user_management: {
    roles: [USER_ROLES.ADMIN, USER_ROLES.SUPER_ADMIN],
    plans: [PLAN_TYPES.ENTERPRISE]
  },
  
  system_monitoring: {
    roles: [USER_ROLES.ADMIN, USER_ROLES.SUPER_ADMIN],
    plans: [PLAN_TYPES.ENTERPRISE]
  }
};

// Demo user profiles for testing
export const DEMO_USERS = {
  regular: {
    email: 'user@example.com',
    role: USER_ROLES.REGULAR,
    plan: PLAN_TYPES.FREE,
    name: 'Regular User',
    features: ['dashboard', 'content']
  },
  
  business: {
    email: 'business@fitnessguru.com', 
    role: USER_ROLES.BUSINESS_OWNER,
    plan: PLAN_TYPES.PROFESSIONAL,
    name: 'Fitness Business Owner',
    features: ['dashboard', 'analytics', 'content', 'revenue_tracking', 'advertising', 'ai_insights']
  },
  
  admin: {
    email: 'admin@autoguru.com',
    role: USER_ROLES.ADMIN, 
    plan: PLAN_TYPES.ENTERPRISE,
    name: 'Platform Administrator',
    features: ['dashboard', 'analytics', 'content', 'revenue_tracking', 'advertising', 'ai_insights', 'admin_tools', 'user_management', 'system_monitoring']
  }
};

// Get user info from email (demo mode)
export const getUserFromEmail = (email) => {
  // In demo mode, determine user type from email
  if (email.includes('admin')) {
    return DEMO_USERS.admin;
  } else if (email.includes('business') || email.includes('fitness') || email.includes('coach')) {
    return DEMO_USERS.business;
  } else {
    return DEMO_USERS.regular;
  }
};

// Check if user has access to a feature
export const hasFeatureAccess = (userRole, userPlan, featureName) => {
  const feature = FEATURE_ACCESS[featureName];
  if (!feature) return false;
  
  const hasRoleAccess = feature.roles.includes(userRole);
  const hasPlanAccess = feature.plans.includes(userPlan);
  
  return hasRoleAccess && hasPlanAccess;
};

// Get navigation items based on user role and plan
export const getNavItemsForUser = (userRole, userPlan) => {
  const navItems = [
    { 
      text: 'Dashboard', 
      path: '/', 
      category: 'main',
      feature: 'dashboard'
    },
    { 
      text: 'Content', 
      path: '/content', 
      category: 'main',
      feature: 'content'
    }
  ];

  // Add features based on access
  if (hasFeatureAccess(userRole, userPlan, 'analytics')) {
    navItems.push({
      text: 'Analytics',
      path: '/analytics', 
      category: 'main',
      feature: 'analytics'
    });
  }

  if (hasFeatureAccess(userRole, userPlan, 'advertising')) {
    navItems.push({
      text: 'Ad Creative Engine',
      path: '/advertising',
      category: 'revenue', 
      badge: 'New',
      feature: 'advertising'
    });
  }

  if (hasFeatureAccess(userRole, userPlan, 'revenue_tracking')) {
    navItems.push({
      text: 'Revenue Tracking',
      path: '/revenue',
      category: 'revenue',
      feature: 'revenue_tracking' 
    });
  }

  if (hasFeatureAccess(userRole, userPlan, 'ai_insights')) {
    navItems.push({
      text: 'AI Insights', 
      path: '/insights',
      category: 'ai',
      feature: 'ai_insights'
    });
  }

  if (hasFeatureAccess(userRole, userPlan, 'admin_tools')) {
    navItems.push({
      text: 'Admin Tools',
      path: '/admin', 
      category: 'admin',
      badge: 'Pro',
      feature: 'admin_tools'
    });
  }

  // Always add settings
  navItems.push({
    text: 'Settings',
    path: '/settings',
    category: 'settings',
    feature: 'settings'
  });

  return navItems;
};

// Get dashboard content based on user role
export const getDashboardConfigForUser = (userRole, userPlan) => {
  const config = {
    showRevenue: hasFeatureAccess(userRole, userPlan, 'revenue_tracking'),
    showAIInsights: hasFeatureAccess(userRole, userPlan, 'ai_insights'),
    showPerformance: hasFeatureAccess(userRole, userPlan, 'system_monitoring'),
    showBasicAnalytics: true,
    maxPosts: userPlan === PLAN_TYPES.FREE ? 10 : userPlan === PLAN_TYPES.STARTER ? 100 : 999,
    platforms: userPlan === PLAN_TYPES.FREE ? 2 : userPlan === PLAN_TYPES.STARTER ? 5 : 8
  };

  return config;
};

// Role-specific welcome messages
export const getWelcomeMessage = (userRole, userName) => {
  switch (userRole) {
    case USER_ROLES.ADMIN:
    case USER_ROLES.SUPER_ADMIN:
      return `Welcome back, ${userName}! Monitor system health and user activity.`;
    case USER_ROLES.BUSINESS_OWNER:
      return `Welcome back, ${userName}! Track your revenue and optimize your campaigns.`;
    case USER_ROLES.REGULAR:
    default:
      return `Welcome back, ${userName}! Create amazing content for your audience.`;
  }
};