import api from './api';

// Get platform connection status
export async function fetchPlatformStatus() {
  try {
    const response = await api.get('/api/v1/platforms/status');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch platform status:', error);
    // Return mock data for now since backend endpoint might not exist
    return {
      instagram: { connected: false, tokenValid: false },
      linkedin: { connected: false, tokenValid: false },
      twitter: { connected: false, tokenValid: false },
      facebook: { connected: false, tokenValid: false },
      youtube: { connected: false, tokenValid: false },
      tiktok: { connected: false, tokenValid: false },
      lastSync: null
    };
  }
}

// Connect to a platform
export async function connectPlatform(platform, credentials) {
  try {
    const response = await api.post(`/api/v1/platforms/${platform}/connect`, {
      client_id: credentials.clientId,
      client_secret: credentials.clientSecret,
      access_token: credentials.accessToken,
      redirect_uri: credentials.redirectUri
    });
    return response.data;
  } catch (error) {
    console.error('Failed to connect platform:', error);
    throw error;
  }
}

// Disconnect from a platform
export async function disconnectPlatform(platform) {
  try {
    const response = await api.post(`/api/v1/platforms/${platform}/disconnect`);
    return response.data;
  } catch (error) {
    console.error('Failed to disconnect platform:', error);
    throw error;
  }
}

// Refresh platform token
export async function refreshPlatformToken(platform) {
  try {
    const response = await api.post(`/api/v1/platforms/${platform}/refresh-token`);
    return response.data;
  } catch (error) {
    console.error('Failed to refresh platform token:', error);
    throw error;
  }
}

// Get platform capabilities
export async function getPlatformCapabilities(platform) {
  try {
    const response = await api.get(`/api/v1/platforms/${platform}/capabilities`);
    return response.data;
  } catch (error) {
    console.error('Failed to get platform capabilities:', error);
    throw error;
  }
}

// Get platform account info
export async function getPlatformAccountInfo(platform) {
  try {
    const response = await api.get(`/api/v1/platforms/${platform}/account`);
    return response.data;
  } catch (error) {
    console.error('Failed to get platform account info:', error);
    throw error;
  }
}

// Publish content to a platform
export async function publishToPlatform(platform, content) {
  try {
    const response = await api.post(`/api/v1/platforms/${platform}/publish`, {
      content: content.text,
      media_urls: content.mediaUrls || [],
      hashtags: content.hashtags || [],
      mentions: content.mentions || [],
      location: content.location,
      schedule_time: content.scheduleTime
    });
    return response.data;
  } catch (error) {
    console.error('Failed to publish to platform:', error);
    throw error;
  }
}

// Get platform analytics
export async function getPlatformAnalytics(platform, timeframe = 'month') {
  try {
    const response = await api.get(`/api/v1/platforms/${platform}/analytics`, {
      params: { timeframe }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to get platform analytics:', error);
    throw error;
  }
}

// Get platform posting schedule
export async function getPlatformSchedule(platform) {
  try {
    const response = await api.get(`/api/v1/platforms/${platform}/schedule`);
    return response.data;
  } catch (error) {
    console.error('Failed to get platform schedule:', error);
    throw error;
  }
}

// Update platform settings
export async function updatePlatformSettings(platform, settings) {
  try {
    const response = await api.patch(`/api/v1/platforms/${platform}/settings`, settings);
    return response.data;
  } catch (error) {
    console.error('Failed to update platform settings:', error);
    throw error;
  }
}

// Get platform content guidelines
export async function getPlatformGuidelines(platform) {
  try {
    const response = await api.get(`/api/v1/platforms/${platform}/guidelines`);
    return response.data;
  } catch (error) {
    console.error('Failed to get platform guidelines:', error);
    // Return default guidelines
    return {
      platform: platform,
      character_limits: {
        post: 2200,
        bio: 150,
        hashtags: 30
      },
      media_requirements: {
        image: { max_size: '10MB', formats: ['jpg', 'png', 'gif'] },
        video: { max_size: '100MB', formats: ['mp4', 'mov'] }
      },
      best_practices: []
    };
  }
}

// Validate content for platform
export async function validateContent(platform, content) {
  try {
    const response = await api.post(`/api/v1/platforms/${platform}/validate`, {
      content: content.text,
      media_urls: content.mediaUrls || [],
      hashtags: content.hashtags || []
    });
    return response.data;
  } catch (error) {
    console.error('Failed to validate content:', error);
    throw error;
  }
}

// Get platform audience insights
export async function getPlatformAudienceInsights(platform) {
  try {
    const response = await api.get(`/api/v1/platforms/${platform}/audience-insights`);
    return response.data;
  } catch (error) {
    console.error('Failed to get platform audience insights:', error);
    throw error;
  }
}

// Get platform trending hashtags
export async function getPlatformTrendingHashtags(platform, location = 'global') {
  try {
    const response = await api.get(`/api/v1/platforms/${platform}/trending-hashtags`, {
      params: { location }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to get trending hashtags:', error);
    throw error;
  }
}

// Get platform optimal posting times
export async function getPlatformOptimalTimes(platform) {
  try {
    const response = await api.get(`/api/v1/platforms/${platform}/optimal-times`);
    return response.data;
  } catch (error) {
    console.error('Failed to get optimal posting times:', error);
    throw error;
  }
}

// Cross-platform publishing
export async function crossPlatformPublish(platforms, content) {
  try {
    const response = await api.post('/api/v1/platforms/cross-publish', {
      platforms: platforms,
      content: content,
      optimize_per_platform: true
    });
    return response.data;
  } catch (error) {
    console.error('Failed to cross-publish content:', error);
    throw error;
  }
}

// Platform-specific content optimization
export async function optimizeContentForPlatform(platform, content) {
  try {
    const response = await api.post(`/api/v1/platforms/${platform}/optimize-content`, {
      content: content.text,
      media_urls: content.mediaUrls || [],
      target_audience: content.targetAudience || {},
      business_niche: content.businessNiche || 'general'
    });
    return response.data;
  } catch (error) {
    console.error('Failed to optimize content for platform:', error);
    throw error;
  }
}

// Get platform rate limits
export async function getPlatformRateLimits(platform) {
  try {
    const response = await api.get(`/api/v1/platforms/${platform}/rate-limits`);
    return response.data;
  } catch (error) {
    console.error('Failed to get platform rate limits:', error);
    throw error;
  }
}

// Platform health check
export async function checkPlatformHealth(platform) {
  try {
    const response = await api.get(`/api/v1/platforms/${platform}/health`);
    return response.data;
  } catch (error) {
    console.error('Failed to check platform health:', error);
    throw error;
  }
}

// Get all supported platforms
export async function getSupportedPlatforms() {
  try {
    const response = await api.get('/api/v1/platforms/supported');
    return response.data;
  } catch (error) {
    console.error('Failed to get supported platforms:', error);
    // Return default supported platforms
    return [
      {
        name: 'Instagram',
        key: 'instagram',
        features: ['posts', 'stories', 'reels', 'igtv'],
        authentication: 'oauth2'
      },
      {
        name: 'LinkedIn',
        key: 'linkedin',
        features: ['posts', 'articles', 'company_updates'],
        authentication: 'oauth2'
      },
      {
        name: 'Twitter',
        key: 'twitter',
        features: ['tweets', 'threads', 'spaces'],
        authentication: 'oauth2'
      },
      {
        name: 'Facebook',
        key: 'facebook',
        features: ['posts', 'stories', 'live'],
        authentication: 'oauth2'
      },
      {
        name: 'YouTube',
        key: 'youtube',
        features: ['videos', 'shorts', 'live'],
        authentication: 'oauth2'
      },
      {
        name: 'TikTok',
        key: 'tiktok',
        features: ['videos', 'live'],
        authentication: 'oauth2'
      }
    ];
  }
}

// Platform automation settings
export async function getPlatformAutomationSettings(platform) {
  try {
    const response = await api.get(`/api/v1/platforms/${platform}/automation`);
    return response.data;
  } catch (error) {
    console.error('Failed to get platform automation settings:', error);
    throw error;
  }
}

export async function updatePlatformAutomationSettings(platform, settings) {
  try {
    const response = await api.patch(`/api/v1/platforms/${platform}/automation`, settings);
    return response.data;
  } catch (error) {
    console.error('Failed to update platform automation settings:', error);
    throw error;
  }
} 