import create from 'zustand';
import * as tasksApi from '../services/tasks';

const useTasksStore = create((set) => ({
  tasks: [],
  loading: false,
  error: null,
  fetchTasks: async () => {
    set({ loading: true, error: null });
    try {
      const result = await tasksApi.fetchTasks();
      const tasks = result.tasks || result || [];
      set({ tasks, loading: false });
    } catch (error) {
      set({ error, loading: false });
    }
  },
  fetchTaskStatus: async (taskId) => {
    set({ loading: true, error: null });
    try {
      const status = await tasksApi.fetchTaskStatus(taskId);
      set({ loading: false });
      return status;
    } catch (error) {
      set({ error, loading: false });
    }
  },
  subscribeTasks: (onMessage) => {
    return tasksApi.subscribeToTasks(onMessage);
  },
}));

export default useTasksStore; 