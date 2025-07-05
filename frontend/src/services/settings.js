import api from './api';

// User Profile Management
export async function fetchUserProfile() {
  try {
    const response = await api.get('/api/v1/user/profile');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch user profile:', error);
    // Return mock data for now since backend endpoint might not exist
    return {
      id: 'user_123',
      firstName: 'John',
      lastName: 'Doe',
      email: 'john.doe@example.com',
      businessNiche: 'general',
      company: 'AutoGuru User',
      avatar: null,
      createdAt: new Date().toISOString(),
      lastLogin: new Date().toISOString()
    };
  }
}

export async function updateUserProfile(profileData) {
  try {
    const response = await api.patch('/api/v1/user/profile', {
      firstName: profileData.firstName,
      lastName: profileData.lastName,
      email: profileData.email,
      businessNiche: profileData.businessNiche,
      company: profileData.company,
      avatar: profileData.avatar
    });
    return response.data;
  } catch (error) {
    console.error('Failed to update user profile:', error);
    throw error;
  }
}

export async function uploadAvatar(avatarFile) {
  try {
    const formData = new FormData();
    formData.append('avatar', avatarFile);
    
    const response = await api.post('/api/v1/user/avatar', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to upload avatar:', error);
    throw error;
  }
}

// API Key Management
export async function fetchApiKey() {
  try {
    const response = await api.get('/api/v1/user/api-key');
    return response.data.api_key;
  } catch (error) {
    console.error('Failed to fetch API key:', error);
    // Return mock data for now
    return 'ak_demo_1234567890abcdef';
  }
}

export async function regenerateApiKey() {
  try {
    const response = await api.post('/api/v1/user/api-key/regenerate');
    return response.data;
  } catch (error) {
    console.error('Failed to regenerate API key:', error);
    throw error;
  }
}

export async function deleteApiKey() {
  try {
    const response = await api.delete('/api/v1/user/api-key');
    return response.data;
  } catch (error) {
    console.error('Failed to delete API key:', error);
    throw error;
  }
}

// OAuth Token Management
export async function fetchOAuthTokens() {
  try {
    const response = await api.get('/api/v1/user/oauth-tokens');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch OAuth tokens:', error);
    // Return mock data for now
    return {
      instagram: { isValid: false, expiresAt: null },
      linkedin: { isValid: false, expiresAt: null },
      twitter: { isValid: false, expiresAt: null },
      facebook: { isValid: false, expiresAt: null },
      youtube: { isValid: false, expiresAt: null },
      tiktok: { isValid: false, expiresAt: null }
    };
  }
}

export async function refreshOAuthToken(platform) {
  try {
    const response = await api.post(`/api/v1/user/oauth-tokens/${platform}/refresh`);
    return response.data;
  } catch (error) {
    console.error('Failed to refresh OAuth token:', error);
    throw error;
  }
}

export async function revokeOAuthToken(platform) {
  try {
    const response = await api.delete(`/api/v1/user/oauth-tokens/${platform}`);
    return response.data;
  } catch (error) {
    console.error('Failed to revoke OAuth token:', error);
    throw error;
  }
}

// Account Settings
export async function fetchAccountSettings() {
  try {
    const response = await api.get('/api/v1/user/settings');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch account settings:', error);
    // Return default settings
    return {
      notifications: {
        email: true,
        push: true,
        sms: false,
        weeklyReports: false,
        errorAlerts: true,
        taskCompletion: true
      },
      automation: {
        autoOptimization: true,
        realTimeAnalytics: true,
        contentScheduling: true,
        dataExport: false
      },
      privacy: {
        shareAnalytics: false,
        publicProfile: false,
        trackingAllowed: true
      },
      preferences: {
        theme: 'light',
        language: 'en',
        timezone: 'UTC',
        dateFormat: 'MM/DD/YYYY'
      }
    };
  }
}

export async function updateAccountSettings(settings) {
  try {
    const response = await api.patch('/api/v1/user/settings', settings);
    return response.data;
  } catch (error) {
    console.error('Failed to update account settings:', error);
    throw error;
  }
}

// Subscription Management
export async function fetchSubscriptionInfo() {
  try {
    const response = await api.get('/api/v1/user/subscription');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch subscription info:', error);
    // Return mock data
    return {
      plan: 'starter',
      status: 'active',
      billingCycle: 'monthly',
      nextBillingDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
      features: ['basic_analytics', 'content_creation', 'platform_integration'],
      limits: {
        posts_per_month: 100,
        platforms: 3,
        analytics_retention: 30
      }
    };
  }
}

export async function updateSubscription(planId) {
  try {
    const response = await api.post('/api/v1/user/subscription/update', {
      plan_id: planId
    });
    return response.data;
  } catch (error) {
    console.error('Failed to update subscription:', error);
    throw error;
  }
}

export async function cancelSubscription() {
  try {
    const response = await api.post('/api/v1/user/subscription/cancel');
    return response.data;
  } catch (error) {
    console.error('Failed to cancel subscription:', error);
    throw error;
  }
}

// Billing Information
export async function fetchBillingInfo() {
  try {
    const response = await api.get('/api/v1/user/billing');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch billing info:', error);
    // Return mock data
    return {
      paymentMethod: {
        type: 'credit_card',
        last4: '4242',
        brand: 'visa',
        expiryMonth: 12,
        expiryYear: 2025
      },
      billingAddress: {
        line1: '123 Main St',
        city: 'New York',
        state: 'NY',
        postalCode: '10001',
        country: 'US'
      }
    };
  }
}

export async function updateBillingInfo(billingData) {
  try {
    const response = await api.patch('/api/v1/user/billing', billingData);
    return response.data;
  } catch (error) {
    console.error('Failed to update billing info:', error);
    throw error;
  }
}

export async function fetchInvoiceHistory() {
  try {
    const response = await api.get('/api/v1/user/invoices');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch invoice history:', error);
    return [];
  }
}

export async function downloadInvoice(invoiceId) {
  try {
    const response = await api.get(`/api/v1/user/invoices/${invoiceId}/download`, {
      responseType: 'blob'
    });
    
    // Create download link
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `invoice_${invoiceId}.pdf`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    
    return { success: true, message: 'Invoice downloaded successfully' };
  } catch (error) {
    console.error('Failed to download invoice:', error);
    throw error;
  }
}

// Security Settings
export async function changePassword(currentPassword, newPassword) {
  try {
    const response = await api.post('/api/v1/user/change-password', {
      current_password: currentPassword,
      new_password: newPassword
    });
    return response.data;
  } catch (error) {
    console.error('Failed to change password:', error);
    throw error;
  }
}

export async function enableTwoFactorAuth() {
  try {
    const response = await api.post('/api/v1/user/2fa/enable');
    return response.data;
  } catch (error) {
    console.error('Failed to enable 2FA:', error);
    throw error;
  }
}

export async function disableTwoFactorAuth(verificationCode) {
  try {
    const response = await api.post('/api/v1/user/2fa/disable', {
      verification_code: verificationCode
    });
    return response.data;
  } catch (error) {
    console.error('Failed to disable 2FA:', error);
    throw error;
  }
}

export async function fetchLoginHistory() {
  try {
    const response = await api.get('/api/v1/user/login-history');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch login history:', error);
    return [];
  }
}

export async function revokeSessionToken(sessionId) {
  try {
    const response = await api.post(`/api/v1/user/sessions/${sessionId}/revoke`);
    return response.data;
  } catch (error) {
    console.error('Failed to revoke session token:', error);
    throw error;
  }
}

// Data Export
export async function requestDataExport(format = 'json') {
  try {
    const response = await api.post('/api/v1/user/data-export', {
      format: format,
      include_analytics: true,
      include_content: true,
      include_settings: true
    });
    return response.data;
  } catch (error) {
    console.error('Failed to request data export:', error);
    throw error;
  }
}

export async function fetchDataExportStatus(exportId) {
  try {
    const response = await api.get(`/api/v1/user/data-export/${exportId}/status`);
    return response.data;
  } catch (error) {
    console.error('Failed to fetch data export status:', error);
    throw error;
  }
}

export async function downloadDataExport(exportId) {
  try {
    const response = await api.get(`/api/v1/user/data-export/${exportId}/download`, {
      responseType: 'blob'
    });
    
    // Create download link
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `autoguru_data_${exportId}.zip`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    
    return { success: true, message: 'Data export downloaded successfully' };
  } catch (error) {
    console.error('Failed to download data export:', error);
    throw error;
  }
}

// Account Deletion
export async function deleteAccount(password, reason) {
  try {
    const response = await api.post('/api/v1/user/delete-account', {
      password: password,
      reason: reason
    });
    return response.data;
  } catch (error) {
    console.error('Failed to delete account:', error);
    throw error;
  }
} 