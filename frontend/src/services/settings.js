import api from './api';

export async function fetchUserProfile() {
  const res = await api.get('/api/v1/user/profile');
  return res.data;
}

export async function updateUserProfile(data) {
  const res = await api.put('/api/v1/user/profile', data);
  return res.data;
}

export async function fetchApiKey() {
  const res = await api.get('/api/v1/user/api-key');
  return res.data;
}

export async function regenerateApiKey() {
  const res = await api.post('/api/v1/user/api-key/regenerate');
  return res.data;
}

export async function fetchOAuthTokens() {
  const res = await api.get('/api/v1/user/oauth-tokens');
  return res.data;
} 