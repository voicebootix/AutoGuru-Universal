import create from 'zustand';
import * as tasksApi from '../services/tasks';

const useTasksStore = create((set) => ({
  tasks: [],
  loading: false,
  error: null,
  fetchTasks: async () => {
    set({ loading: true, error: null });
    try {
      const tasks = await tasksApi.fetchTasks();
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
    return tasksApi.subscribeTasksWS(onMessage);
  },
}));

export default useTasksStore; 