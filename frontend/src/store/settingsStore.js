import create from 'zustand';
import * as settingsApi from '../services/settings';

const useSettingsStore = create((set) => ({
  profile: null,
  apiKey: null,
  oauthTokens: null,
  loading: false,
  error: null,
  fetchProfile: async () => {
    set({ loading: true, error: null });
    try {
      const profile = await settingsApi.fetchUserProfile();
      set({ profile, loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
  updateProfile: async (data) => {
    set({ loading: true, error: null });
    try {
      await settingsApi.updateUserProfile(data);
      set({ loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
  fetchApiKey: async () => {
    set({ loading: true, error: null });
    try {
      const apiKey = await settingsApi.fetchApiKey();
      set({ apiKey, loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
  regenerateApiKey: async () => {
    set({ loading: true, error: null });
    try {
      await settingsApi.regenerateApiKey();
      set({ loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
  fetchOAuthTokens: async () => {
    set({ loading: true, error: null });
    try {
      const oauthTokens = await settingsApi.fetchOAuthTokens();
      set({ oauthTokens, loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
}));

export default useSettingsStore; 