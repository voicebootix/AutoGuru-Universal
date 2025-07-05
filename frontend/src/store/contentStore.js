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
      // Refresh content list after creating new content
      const contentList = await contentApi.fetchContentHistory();
      set({ contentList: contentList.content || contentList, loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
  scheduleContent: async (data) => {
    set({ loading: true, error: null });
    try {
      await contentApi.scheduleContent(data);
      // Refresh content list after scheduling
      const contentList = await contentApi.fetchContentHistory();
      set({ contentList: contentList.content || contentList, loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
  fetchContentHistory: async () => {
    set({ loading: true, error: null });
    try {
      const response = await contentApi.fetchContentHistory();
      // Handle both mock data structure and real API structure
      const contentList = response.content || response || [];
      set({ contentList, loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
}));

export default useContentStore; 