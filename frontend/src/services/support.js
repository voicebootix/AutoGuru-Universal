import api from './api';

// Support Information
export async function fetchSupportInfo() {
  try {
    const response = await api.get('/support');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch support info:', error);
    // Return default support information
    return {
      email: 'support@autoguru.com',
      phone: '+1 (555) 123-4567',
      url: 'https://autoguru.com/support',
      hours: 'Monday-Friday, 9AM-6PM EST',
      chat_available: true
    };
  }
}

// Feedback Submission
export async function submitFeedback(feedbackData) {
  try {
    const response = await api.post('/api/v1/support/feedback', {
      type: feedbackData.type,
      subject: feedbackData.subject,
      message: feedbackData.message,
      priority: feedbackData.priority || 'medium',
      category: feedbackData.category || 'general',
      user_agent: navigator.userAgent,
      url: window.location.href
    });
    return response.data;
  } catch (error) {
    console.error('Failed to submit feedback:', error);
    throw error;
  }
}

// Bug Reports
export async function submitBugReport(bugData) {
  try {
    const response = await api.post('/api/v1/support/bug-report', {
      title: bugData.title,
      description: bugData.description,
      steps_to_reproduce: bugData.stepsToReproduce,
      expected_behavior: bugData.expectedBehavior,
      actual_behavior: bugData.actualBehavior,
      severity: bugData.severity || 'medium',
      browser: bugData.browser || navigator.userAgent,
      operating_system: bugData.operatingSystem || navigator.platform,
      screenshot_urls: bugData.screenshotUrls || [],
      console_logs: bugData.consoleLogs || []
    });
    return response.data;
  } catch (error) {
    console.error('Failed to submit bug report:', error);
    throw error;
  }
}

// Feature Requests
export async function submitFeatureRequest(featureData) {
  try {
    const response = await api.post('/api/v1/support/feature-request', {
      title: featureData.title,
      description: featureData.description,
      use_case: featureData.useCase,
      priority: featureData.priority || 'medium',
      business_impact: featureData.businessImpact,
      additional_context: featureData.additionalContext
    });
    return response.data;
  } catch (error) {
    console.error('Failed to submit feature request:', error);
    throw error;
  }
}

// Support Tickets
export async function createSupportTicket(ticketData) {
  try {
    const response = await api.post('/api/v1/support/tickets', {
      subject: ticketData.subject,
      description: ticketData.description,
      category: ticketData.category,
      priority: ticketData.priority || 'medium',
      attachments: ticketData.attachments || []
    });
    return response.data;
  } catch (error) {
    console.error('Failed to create support ticket:', error);
    throw error;
  }
}

export async function fetchSupportTickets(filters = {}) {
  try {
    const response = await api.get('/api/v1/support/tickets', {
      params: {
        status: filters.status || 'all',
        limit: filters.limit || 20,
        offset: filters.offset || 0
      }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch support tickets:', error);
    return { tickets: [], total: 0 };
  }
}

export async function fetchSupportTicket(ticketId) {
  try {
    const response = await api.get(`/api/v1/support/tickets/${ticketId}`);
    return response.data;
  } catch (error) {
    console.error('Failed to fetch support ticket:', error);
    throw error;
  }
}

export async function updateSupportTicket(ticketId, updateData) {
  try {
    const response = await api.patch(`/api/v1/support/tickets/${ticketId}`, updateData);
    return response.data;
  } catch (error) {
    console.error('Failed to update support ticket:', error);
    throw error;
  }
}

export async function closeSupportTicket(ticketId) {
  try {
    const response = await api.post(`/api/v1/support/tickets/${ticketId}/close`);
    return response.data;
  } catch (error) {
    console.error('Failed to close support ticket:', error);
    throw error;
  }
}

// Help Articles & Documentation
export async function fetchHelpArticles(category = 'all') {
  try {
    const response = await api.get('/api/v1/support/help-articles', {
      params: { category }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch help articles:', error);
    // Return mock help articles
    return {
      articles: [
        {
          id: 'getting-started',
          title: 'Getting Started with AutoGuru Universal',
          category: 'basics',
          content: 'Learn how to set up your account and start automating your social media.',
          last_updated: new Date().toISOString()
        },
        {
          id: 'connecting-platforms',
          title: 'Connecting Your Social Media Platforms',
          category: 'platforms',
          content: 'Step-by-step guide to connect Instagram, LinkedIn, Twitter, and more.',
          last_updated: new Date().toISOString()
        },
        {
          id: 'content-creation',
          title: 'AI-Powered Content Creation',
          category: 'content',
          content: 'How to use AI to create engaging content for any business niche.',
          last_updated: new Date().toISOString()
        }
      ],
      categories: ['basics', 'platforms', 'content', 'analytics', 'billing']
    };
  }
}

export async function fetchHelpArticle(articleId) {
  try {
    const response = await api.get(`/api/v1/support/help-articles/${articleId}`);
    return response.data;
  } catch (error) {
    console.error('Failed to fetch help article:', error);
    throw error;
  }
}

export async function searchHelpArticles(query) {
  try {
    const response = await api.get('/api/v1/support/help-articles/search', {
      params: { q: query }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to search help articles:', error);
    return { articles: [] };
  }
}

// FAQ
export async function fetchFAQs(category = 'all') {
  try {
    const response = await api.get('/api/v1/support/faqs', {
      params: { category }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch FAQs:', error);
    // Return mock FAQs
    return {
      faqs: [
        {
          question: "How do I connect my social media accounts?",
          answer: "Go to the Platforms section and click 'Connect' next to the platform you want to connect. Follow the OAuth flow to authorize AutoGuru Universal to access your account.",
          category: 'platforms'
        },
        {
          question: "How does the AI content creation work?",
          answer: "Our AI analyzes your business niche, target audience, and past performance to generate optimized content. You can customize the content before publishing or scheduling.",
          category: 'content'
        },
        {
          question: "What analytics are available?",
          answer: "We provide comprehensive analytics including engagement rates, reach, impressions, conversion tracking, and AI-powered insights to help optimize your strategy.",
          category: 'analytics'
        }
      ]
    };
  }
}

// System Status
export async function fetchSystemStatus() {
  try {
    const response = await api.get('/api/v1/support/system-status');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch system status:', error);
    // Return default system status
    return {
      status: 'operational',
      services: {
        api: 'operational',
        dashboard: 'operational',
        content_creation: 'operational',
        analytics: 'operational',
        platform_connections: 'operational'
      },
      last_updated: new Date().toISOString(),
      incidents: []
    };
  }
}

// Live Chat
export async function initiateLiveChat() {
  try {
    const response = await api.post('/api/v1/support/live-chat/start');
    return response.data;
  } catch (error) {
    console.error('Failed to initiate live chat:', error);
    throw error;
  }
}

export async function sendChatMessage(sessionId, message) {
  try {
    const response = await api.post(`/api/v1/support/live-chat/${sessionId}/message`, {
      message: message,
      timestamp: new Date().toISOString()
    });
    return response.data;
  } catch (error) {
    console.error('Failed to send chat message:', error);
    throw error;
  }
}

export async function endLiveChat(sessionId) {
  try {
    const response = await api.post(`/api/v1/support/live-chat/${sessionId}/end`);
    return response.data;
  } catch (error) {
    console.error('Failed to end live chat:', error);
    throw error;
  }
}

// Contact Form
export async function submitContactForm(contactData) {
  try {
    const response = await api.post('/api/v1/support/contact', {
      name: contactData.name,
      email: contactData.email,
      company: contactData.company,
      subject: contactData.subject,
      message: contactData.message,
      inquiry_type: contactData.inquiryType || 'general'
    });
    return response.data;
  } catch (error) {
    console.error('Failed to submit contact form:', error);
    throw error;
  }
}

// Video Tutorials
export async function fetchVideoTutorials(category = 'all') {
  try {
    const response = await api.get('/api/v1/support/video-tutorials', {
      params: { category }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch video tutorials:', error);
    // Return mock video tutorials
    return {
      tutorials: [
        {
          id: 'setup-tutorial',
          title: 'Setting Up Your AutoGuru Account',
          description: 'Complete walkthrough of account setup and initial configuration',
          duration: '10:30',
          thumbnail: '/assets/thumbnails/setup.jpg',
          video_url: 'https://example.com/videos/setup',
          category: 'getting-started'
        },
        {
          id: 'content-creation-tutorial',
          title: 'Creating Your First AI-Generated Content',
          description: 'Learn how to use AI to create engaging content for your business',
          duration: '15:45',
          thumbnail: '/assets/thumbnails/content.jpg',
          video_url: 'https://example.com/videos/content-creation',
          category: 'content'
        }
      ]
    };
  }
}

// Feedback on Help Content
export async function rateHelpContent(contentId, rating, feedback = '') {
  try {
    const response = await api.post(`/api/v1/support/help-content/${contentId}/rate`, {
      rating: rating, // 1-5 stars
      feedback: feedback
    });
    return response.data;
  } catch (error) {
    console.error('Failed to rate help content:', error);
    throw error;
  }
}

// Knowledge Base Search
export async function searchKnowledgeBase(query, filters = {}) {
  try {
    const response = await api.get('/api/v1/support/knowledge-base/search', {
      params: {
        q: query,
        category: filters.category,
        content_type: filters.contentType, // 'article', 'faq', 'video'
        limit: filters.limit || 10
      }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to search knowledge base:', error);
    return { results: [] };
  }
} 