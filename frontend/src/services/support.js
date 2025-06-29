import api from './api';

export async function fetchSupportInfo() {
  const res = await api.get('/support');
  return res.data;
}

export async function submitFeedback(data) {
  const res = await api.post('/api/v1/support/feedback', data);
  return res.data;
} 