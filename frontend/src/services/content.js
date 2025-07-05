import api from './api';

// Content creation and management
export async function createContent(contentData) {
  try {
    const response = await api.post('/api/v1/create-viral-content', {
      topic: contentData.title,
      business_niche: contentData.businessNiche || 'general',
      platforms: contentData.platform === 'all' ? ['instagram', 'linkedin', 'twitter', 'facebook'] : [contentData.platform],
      content_type: contentData.contentType || 'post',
      target_audience: {
        description: contentData.description,
        demographics: {},
        interests: []
      }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to create content:', error);
    throw error;
  }
}

// Schedule content for publishing
export async function scheduleContent(scheduleData) {
  try {
    const response = await api.post('/api/v1/publish', {
      content: {
        text: scheduleData.content || '',
        platform: scheduleData.platform || 'all',
        media_urls: scheduleData.mediaUrls || [],
        hashtags: scheduleData.hashtags || []
      },
      schedule_time: scheduleData.scheduledAt,
      cross_post: scheduleData.crossPost || false
    });
    return response.data;
  } catch (error) {
    console.error('Failed to schedule content:', error);
    throw error;
  }
}

// Get content history
export async function fetchContentHistory(filters = {}) {
  try {
    const response = await api.get('/api/v1/content/history', {
      params: {
        limit: filters.limit || 50,
        offset: filters.offset || 0,
        platform: filters.platform || 'all',
        status: filters.status || 'all',
        date_from: filters.dateFrom,
        date_to: filters.dateTo
      }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch content history:', error);
    // Return mock data for now since backend endpoint might not exist
    return {
      content: [],
      total: 0,
      page: 1,
      per_page: 50
    };
  }
}

// Analyze content performance
export async function analyzeContent(contentText, context = {}) {
  try {
    const response = await api.post('/api/v1/analyze', {
      content: contentText,
      context: {
        business_niche: context.businessNiche || 'general',
        target_audience: context.targetAudience || {},
        platform: context.platform || 'all',
        ...context
      },
      platforms: context.platforms || ['instagram', 'linkedin', 'twitter', 'facebook']
    });
    return response.data;
  } catch (error) {
    console.error('Failed to analyze content:', error);
    throw error;
  }
}

// Generate content variations
export async function generateContentVariations(originalContent, options = {}) {
  try {
    const response = await api.post('/api/v1/content/variations', {
      original_content: originalContent,
      variation_count: options.count || 3,
      platforms: options.platforms || ['instagram', 'linkedin', 'twitter'],
      tone: options.tone || 'professional',
      business_niche: options.businessNiche || 'general'
    });
    return response.data;
  } catch (error) {
    console.error('Failed to generate content variations:', error);
    throw error;
  }
}

// Get content suggestions based on trending topics
export async function getContentSuggestions(businessNiche = 'general', platform = 'all') {
  try {
    const response = await api.post('/api/v1/content/suggestions', {
      business_niche: businessNiche,
      platform: platform,
      include_trending: true,
      include_seasonal: true,
      count: 10
    });
    return response.data;
  } catch (error) {
    console.error('Failed to get content suggestions:', error);
    throw error;
  }
}

// Optimize content for specific platform
export async function optimizeContentForPlatform(content, platform) {
  try {
    const response = await api.post('/api/v1/content/optimize', {
      content: content,
      target_platform: platform,
      include_hashtags: true,
      include_timing: true,
      include_format_suggestions: true
    });
    return response.data;
  } catch (error) {
    console.error('Failed to optimize content:', error);
    throw error;
  }
}

// Get content calendar
export async function fetchContentCalendar(month, year) {
  try {
    const response = await api.get('/api/v1/content/calendar', {
      params: {
        month: month,
        year: year,
        include_scheduled: true,
        include_published: true
      }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch content calendar:', error);
    // Return mock data for now
    return {
      events: [],
      scheduled_posts: [],
      published_posts: []
    };
  }
}

// Update content status
export async function updateContentStatus(contentId, status) {
  try {
    const response = await api.patch(`/api/v1/content/${contentId}/status`, {
      status: status
    });
    return response.data;
  } catch (error) {
    console.error('Failed to update content status:', error);
    throw error;
  }
}

// Delete content
export async function deleteContent(contentId) {
  try {
    const response = await api.delete(`/api/v1/content/${contentId}`);
    return response.data;
  } catch (error) {
    console.error('Failed to delete content:', error);
    throw error;
  }
}

// Get content performance metrics
export async function getContentMetrics(contentId) {
  try {
    const response = await api.get(`/api/v1/content/${contentId}/metrics`);
    return response.data;
  } catch (error) {
    console.error('Failed to get content metrics:', error);
    throw error;
  }
}

// Generate persona-based content
export async function generatePersonaContent(personaData) {
  try {
    // First generate persona
    const personaResponse = await api.post('/api/v1/generate-persona', {
      business_description: personaData.businessDescription,
      target_market: personaData.targetMarket,
      goals: personaData.goals || []
    });
    
    // Then create content based on persona
    const contentResponse = await api.post('/api/v1/create-viral-content', {
      topic: personaData.topic,
      business_niche: personaData.businessNiche || 'general',
      platforms: personaData.platforms || ['instagram', 'linkedin'],
      target_audience: personaResponse.data
    });
    
    return {
      persona: personaResponse.data,
      content: contentResponse.data
    };
  } catch (error) {
    console.error('Failed to generate persona content:', error);
    throw error;
  }
}

// Bulk upload content
export async function bulkUploadContent(contentList) {
  try {
    const response = await api.post('/api/v1/content/bulk-upload', {
      content_list: contentList,
      auto_optimize: true,
      auto_schedule: false
    });
    return response.data;
  } catch (error) {
    console.error('Failed to bulk upload content:', error);
    throw error;
  }
} 