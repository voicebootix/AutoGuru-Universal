import api from './api';

export async function createContent(data) {
  const res = await api.post('/api/v1/create-viral-content', data);
  return res.data;
}

export async function scheduleContent(data) {
  const res = await api.post('/api/v1/publish', data);
  return res.data;
}

export async function fetchContentHistory() {
  const res = await api.get('/api/v1/content/history');
  return res.data;
} 