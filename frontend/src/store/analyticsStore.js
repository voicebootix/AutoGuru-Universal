import create from 'zustand';
import * as analyticsApi from '../services/analytics';

const useAnalyticsStore = create((set) => ({
  dashboard: null,
  analytics: null,
  loading: false,
  error: null,
  fetchDashboard: async () => {
    set({ loading: true, error: null });
    try {
      const dashboard = await analyticsApi.fetchDashboardStats();
      set({ dashboard, loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
  fetchAnalytics: async (params) => {
    set({ loading: true, error: null });
    try {
      const analytics = await analyticsApi.fetchAnalytics(params);
      set({ analytics, loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
}));

export default useAnalyticsStore; 