import create from 'zustand';
import * as platformsApi from '../services/platforms';

const usePlatformStore = create((set) => ({
  status: {},
  loading: false,
  error: null,
  fetchStatus: async () => {
    set({ loading: true, error: null });
    try {
      const status = await platformsApi.fetchPlatformStatus();
      set({ status, loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
  connect: async (platform, data) => {
    set({ loading: true, error: null });
    try {
      await platformsApi.connectPlatform(platform, data);
      set({ loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
  refreshToken: async (platform) => {
    set({ loading: true, error: null });
    try {
      await platformsApi.refreshPlatformToken(platform);
      set({ loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
}));

export default usePlatformStore; 