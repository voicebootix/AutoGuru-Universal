import create from 'zustand';
import * as supportApi from '../services/support';

const useSupportStore = create((set) => ({
  info: null,
  feedbackStatus: null,
  loading: false,
  error: null,
  fetchInfo: async () => {
    set({ loading: true, error: null });
    try {
      const info = await supportApi.fetchSupportInfo();
      set({ info, loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
  submitFeedback: async (data) => {
    set({ loading: true, error: null });
    try {
      await supportApi.submitFeedback(data);
      set({ feedbackStatus: 'submitted', loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
}));

export default useSupportStore; 