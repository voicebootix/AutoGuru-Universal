import create from 'zustand';
import * as contentApi from '../services/content';

const useContentStore = create((set) => ({
  contentList: [],
  loading: false,
  error: null,
  createContent: async (data) => {
    set({ loading: true, error: null });
    try {
      await contentApi.createContent(data);
      set({ loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
  scheduleContent: async (data) => {
    set({ loading: true, error: null });
    try {
      await contentApi.scheduleContent(data);
      set({ loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
  fetchContentHistory: async () => {
    set({ loading: true, error: null });
    try {
      const contentList = await contentApi.fetchContentHistory();
      set({ contentList, loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
}));

export default useContentStore; 