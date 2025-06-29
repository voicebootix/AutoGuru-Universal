import api from './api';

export async function fetchPlatformStatus() {
  const res = await api.get('/api/v1/platforms/status');
  return res.data;
}

export async function connectPlatform(platform, data) {
  const res = await api.post(`/api/v1/platforms/connect/${platform}`, data);
  return res.data;
}

export async function refreshPlatformToken(platform) {
  const res = await api.post(`/api/v1/platforms/refresh/${platform}`);
  return res.data;
} 