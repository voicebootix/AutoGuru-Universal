import api from './api';

export async function fetchTasks() {
  const res = await api.get('/api/v1/tasks');
  return res.data;
}

export async function fetchTaskStatus(taskId) {
  const res = await api.get(`/api/v1/tasks/${taskId}`);
  return res.data;
}

export function subscribeTasksWS(onMessage) {
  const ws = new WebSocket(`${import.meta.env.VITE_API_URL.replace('http', 'ws')}/ws/tasks`);
  ws.onmessage = (event) => onMessage(JSON.parse(event.data));
  return ws;
} 