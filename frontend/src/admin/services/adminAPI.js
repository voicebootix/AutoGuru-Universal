/**
 * Admin API Service for AutoGuru Universal
 * 
 * This service handles all admin-related API calls including platform management,
 * user administration, system configuration, and analytics.
 */

class AdminAPIService {
  constructor() {
    this.baseURL = '/api/admin';
    this.token = localStorage.getItem('admin_token');
  }

  // Authentication methods
  async login(username, password) {
    const response = await this.request('POST', '/auth/login', {
      username,
      password
    });
    
    if (response.access_token) {
      this.token = response.access_token;
      localStorage.setItem('admin_token', this.token);
      localStorage.setItem('admin_user', JSON.stringify(response.admin_user));
    }
    
    return response;
  }

  logout() {
    this.token = null;
    localStorage.removeItem('admin_token');
    localStorage.removeItem('admin_user');
  }

  isAuthenticated() {
    return !!this.token;
  }

  getStoredAdminUser() {
    const user = localStorage.getItem('admin_user');
    return user ? JSON.parse(user) : null;
  }

  // Dashboard methods
  async getDashboardData(timeframe = 'week') {
    return this.request('GET', `/dashboard?timeframe=${timeframe}`);
  }

  async getSystemHealth() {
    return this.request('GET', '/health/detailed');
  }

  async getNotifications() {
    // Placeholder for notifications endpoint
    return [];
  }

  // Platform management methods
  async getPlatformStatuses() {
    return this.request('GET', '/platforms/status');
  }

  async storePlatformCredential(platformType, credentialData) {
    return this.request('POST', `/platforms/${platformType}/credentials`, credentialData);
  }

  async testPlatformConnection(platformType, testRequest) {
    return this.request('POST', `/platforms/${platformType}/test-connection`, testRequest);
  }

  async bulkConfigurePlatforms(configurations) {
    return this.request('POST', '/platforms/bulk-configure', configurations);
  }

  // System configuration methods
  async getSystemConfiguration(configType = null) {
    const params = configType ? `?config_type=${configType}` : '';
    return this.request('GET', `/system/config${params}`);
  }

  async updateSystemConfiguration(configKey, configUpdate) {
    return this.request('POST', `/system/config/${configKey}`, configUpdate);
  }

  // User management methods
  async getUsers(params = {}) {
    const queryString = new URLSearchParams(params).toString();
    return this.request('GET', `/users?${queryString}`);
  }

  async getUserDetails(userId) {
    return this.request('GET', `/users/${userId}`);
  }

  async createUser(userData) {
    return this.request('POST', '/users', userData);
  }

  async updateUser(userId, userData) {
    return this.request('PUT', `/users/${userId}`, userData);
  }

  async deleteUser(userId) {
    return this.request('DELETE', `/users/${userId}`);
  }

  async manageUserPlatformConnection(userId, connectionRequest) {
    return this.request('POST', `/users/${userId}/platform-connections`, connectionRequest);
  }

  // Analytics methods
  async getRevenueAnalytics(timeframe = 'month', breakdown = 'daily') {
    return this.request('GET', `/analytics/revenue?timeframe=${timeframe}&breakdown=${breakdown}`);
  }

  async getPlatformUsageAnalytics(timeframe = 'week') {
    return this.request('GET', `/analytics/platform-usage?timeframe=${timeframe}`);
  }

  async getUserAnalytics(timeframe = 'month') {
    return this.request('GET', `/analytics/users?timeframe=${timeframe}`);
  }

  // Bulk operations
  async executeBulkOperation(operationRequest) {
    return this.request('POST', '/bulk-operations', operationRequest);
  }

  // Maintenance and health
  async runMaintenanceCleanup(cleanupType = 'all') {
    return this.request('POST', `/maintenance/cleanup?cleanup_type=${cleanupType}`);
  }

  async getSystemDiagnostics() {
    return this.request('GET', '/health/detailed');
  }

  // API Keys and AI Services
  async getAIServiceConfigurations() {
    return this.request('GET', '/ai-services/config');
  }

  async updateAIServiceConfiguration(serviceType, config) {
    return this.request('POST', `/ai-services/${serviceType}/config`, config);
  }

  async testAIServiceConnection(serviceType) {
    return this.request('POST', `/ai-services/${serviceType}/test`);
  }

  // Content and automation management
  async getContentAnalytics(timeframe = 'week') {
    return this.request('GET', `/analytics/content?timeframe=${timeframe}`);
  }

  async getAutomationRules() {
    return this.request('GET', '/automation/rules');
  }

  async createAutomationRule(ruleData) {
    return this.request('POST', '/automation/rules', ruleData);
  }

  async updateAutomationRule(ruleId, ruleData) {
    return this.request('PUT', `/automation/rules/${ruleId}`, ruleData);
  }

  async deleteAutomationRule(ruleId) {
    return this.request('DELETE', `/automation/rules/${ruleId}`);
  }

  // System monitoring
  async getPerformanceMetrics(timeframe = 'hour') {
    return this.request('GET', `/monitoring/performance?timeframe=${timeframe}`);
  }

  async getErrorLogs(limit = 100, severity = 'all') {
    return this.request('GET', `/monitoring/errors?limit=${limit}&severity=${severity}`);
  }

  async getAuditLog(limit = 100, userId = null) {
    const params = new URLSearchParams({ limit });
    if (userId) params.append('user_id', userId);
    return this.request('GET', `/monitoring/audit?${params.toString()}`);
  }

  // Backup and restore
  async createSystemBackup(backupType = 'full') {
    return this.request('POST', '/backup/create', { backup_type: backupType });
  }

  async getBackupHistory() {
    return this.request('GET', '/backup/history');
  }

  async restoreFromBackup(backupId) {
    return this.request('POST', `/backup/restore/${backupId}`);
  }

  // Generic request handler
  async request(method, endpoint, data = null) {
    const url = `${this.baseURL}${endpoint}`;
    
    const options = {
      method,
      headers: {
        'Content-Type': 'application/json',
      }
    };

    // Add authorization header if token exists
    if (this.token) {
      options.headers['Authorization'] = `Bearer ${this.token}`;
    }

    // Add data for POST/PUT requests
    if (data && (method === 'POST' || method === 'PUT' || method === 'PATCH')) {
      options.body = JSON.stringify(data);
    }

    try {
      const response = await fetch(url, options);
      
      // Handle authentication errors
      if (response.status === 401) {
        this.logout();
        throw new Error('Authentication expired. Please login again.');
      }

      // Handle other HTTP errors
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      // Return response data
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        return await response.json();
      }
      
      return await response.text();
      
    } catch (error) {
      console.error(`Admin API ${method} ${endpoint} failed:`, error);
      throw error;
    }
  }

  // File upload handler for configuration imports
  async uploadFile(endpoint, file, additionalData = {}) {
    const url = `${this.baseURL}${endpoint}`;
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Add additional form data
    Object.entries(additionalData).forEach(([key, value]) => {
      formData.append(key, value);
    });

    const options = {
      method: 'POST',
      headers: {}
    };

    // Add authorization header if token exists
    if (this.token) {
      options.headers['Authorization'] = `Bearer ${this.token}`;
    }

    options.body = formData;

    try {
      const response = await fetch(url, options);
      
      if (response.status === 401) {
        this.logout();
        throw new Error('Authentication expired. Please login again.');
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Upload failed: ${response.statusText}`);
      }

      return await response.json();
      
    } catch (error) {
      console.error(`File upload to ${endpoint} failed:`, error);
      throw error;
    }
  }

  // Configuration export/import
  async exportConfiguration(configTypes = ['all']) {
    const params = new URLSearchParams();
    configTypes.forEach(type => params.append('types', type));
    
    const response = await fetch(`${this.baseURL}/config/export?${params.toString()}`, {
      headers: {
        'Authorization': `Bearer ${this.token}`
      }
    });

    if (!response.ok) {
      throw new Error('Configuration export failed');
    }

    // Download file
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `autoguru-config-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }

  async importConfiguration(file) {
    return this.uploadFile('/config/import', file);
  }

  // Real-time monitoring setup
  setupRealTimeMonitoring(onUpdate) {
    // This would set up WebSocket connection for real-time updates
    // For now, we'll use polling
    const interval = setInterval(async () => {
      try {
        const health = await this.getSystemHealth();
        const platformStatus = await this.getPlatformStatuses();
        
        onUpdate({
          health,
          platformStatus,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        console.error('Real-time monitoring update failed:', error);
      }
    }, 30000); // Update every 30 seconds

    // Return cleanup function
    return () => clearInterval(interval);
  }

  // Performance optimization: Request caching
  _cache = new Map();
  _cacheExpiry = new Map();

  async cachedRequest(method, endpoint, data = null, cacheDuration = 5 * 60 * 1000) {
    const cacheKey = `${method}:${endpoint}:${JSON.stringify(data)}`;
    const now = Date.now();
    
    // Check if we have a valid cached response
    if (this._cache.has(cacheKey)) {
      const expiry = this._cacheExpiry.get(cacheKey);
      if (now < expiry) {
        return this._cache.get(cacheKey);
      }
    }

    // Make the request
    const response = await this.request(method, endpoint, data);
    
    // Cache the response
    this._cache.set(cacheKey, response);
    this._cacheExpiry.set(cacheKey, now + cacheDuration);
    
    return response;
  }

  // Clear cache
  clearCache() {
    this._cache.clear();
    this._cacheExpiry.clear();
  }

  // Health check with retry logic
  async healthCheck(maxRetries = 3) {
    for (let i = 0; i < maxRetries; i++) {
      try {
        const response = await fetch(`${this.baseURL}/health/ping`, {
          headers: this.token ? { 'Authorization': `Bearer ${this.token}` } : {}
        });
        
        if (response.ok) {
          return { status: 'healthy', attempt: i + 1 };
        }
      } catch (error) {
        if (i === maxRetries - 1) {
          throw new Error(`Health check failed after ${maxRetries} attempts: ${error.message}`);
        }
        
        // Wait before retry
        await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
      }
    }
  }
}

// Create singleton instance
export const adminAPI = new AdminAPIService();

// Export additional utilities
export const AdminAPIError = class extends Error {
  constructor(message, status, details = null) {
    super(message);
    this.name = 'AdminAPIError';
    this.status = status;
    this.details = details;
  }
};

export const formatError = (error) => {
  if (error instanceof AdminAPIError) {
    return {
      message: error.message,
      status: error.status,
      details: error.details
    };
  }
  
  return {
    message: error.message || 'An unexpected error occurred',
    status: null,
    details: null
  };
};

export default adminAPI;