import { useState, useEffect, useCallback } from 'react';
import { adminAPI } from '../services/adminAPI';

/**
 * Custom hook for admin authentication management
 * Handles login, logout, and authentication state
 */
export function useAdminAuth() {
  const [adminUser, setAdminUser] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Initialize authentication state
  useEffect(() => {
    const initializeAuth = () => {
      try {
        const token = localStorage.getItem('admin_token');
        const storedUser = adminAPI.getStoredAdminUser();
        
        if (token && storedUser) {
          setAdminUser(storedUser);
          setIsAuthenticated(true);
        }
      } catch (err) {
        console.error('Failed to initialize admin auth:', err);
        // Clear potentially corrupted data
        adminAPI.logout();
      } finally {
        setLoading(false);
      }
    };

    initializeAuth();
  }, []);

  // Login function
  const login = useCallback(async (username, password) => {
    setLoading(true);
    setError(null);

    try {
      const response = await adminAPI.login(username, password);
      
      setAdminUser(response.admin_user);
      setIsAuthenticated(true);
      
      return response;
    } catch (err) {
      setError(err.message || 'Login failed');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // Logout function
  const logout = useCallback(() => {
    adminAPI.logout();
    setAdminUser(null);
    setIsAuthenticated(false);
    setError(null);
  }, []);

  // Refresh user data
  const refreshUser = useCallback(async () => {
    if (!isAuthenticated || !adminUser?.id) return;

    try {
      // In a real implementation, you'd have an endpoint to get current user
      // For now, we'll just verify the token is still valid
      await adminAPI.getSystemHealth();
    } catch (err) {
      if (err.message.includes('Authentication')) {
        logout();
      }
    }
  }, [isAuthenticated, adminUser?.id, logout]);

  // Check if user has specific permission
  const hasPermission = useCallback((permission) => {
    if (!adminUser || !adminUser.permissions) return false;
    
    // Super admin has all permissions
    if (adminUser.role === 'super_admin' || adminUser.permissions.includes('all')) {
      return true;
    }
    
    return adminUser.permissions.includes(permission);
  }, [adminUser]);

  // Check if user has specific role
  const hasRole = useCallback((role) => {
    if (!adminUser) return false;
    return adminUser.role === role;
  }, [adminUser]);

  return {
    adminUser,
    isAuthenticated,
    loading,
    error,
    login,
    logout,
    refreshUser,
    hasPermission,
    hasRole,
    clearError: () => setError(null)
  };
}