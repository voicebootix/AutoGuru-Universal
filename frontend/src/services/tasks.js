import api from './api';

// Get all tasks
export async function fetchTasks(filters = {}) {
  try {
    const response = await api.get('/api/v1/tasks', {
      params: {
        status: filters.status || 'all',
        limit: filters.limit || 50,
        offset: filters.offset || 0,
        sort: filters.sort || 'created_at',
        order: filters.order || 'desc'
      }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch tasks:', error);
    // Return mock data for now since backend endpoint might not exist
    return {
      tasks: [],
      total: 0,
      page: 1,
      per_page: 50
    };
  }
}

// Get task status by ID
export async function fetchTaskStatus(taskId) {
  try {
    const response = await api.get(`/api/v1/tasks/${taskId}`);
    return response.data;
  } catch (error) {
    console.error('Failed to fetch task status:', error);
    throw error;
  }
}

// Create a new task
export async function createTask(taskData) {
  try {
    const response = await api.post('/api/v1/tasks', {
      name: taskData.name,
      description: taskData.description,
      type: taskData.type,
      priority: taskData.priority || 'medium',
      scheduled_at: taskData.scheduledAt,
      config: taskData.config || {}
    });
    return response.data;
  } catch (error) {
    console.error('Failed to create task:', error);
    throw error;
  }
}

// Update task status
export async function updateTaskStatus(taskId, status) {
  try {
    const response = await api.patch(`/api/v1/tasks/${taskId}/status`, {
      status: status
    });
    return response.data;
  } catch (error) {
    console.error('Failed to update task status:', error);
    throw error;
  }
}

// Cancel a task
export async function cancelTask(taskId) {
  try {
    const response = await api.post(`/api/v1/tasks/${taskId}/cancel`);
    return response.data;
  } catch (error) {
    console.error('Failed to cancel task:', error);
    throw error;
  }
}

// Pause a task
export async function pauseTask(taskId) {
  try {
    const response = await api.post(`/api/v1/tasks/${taskId}/pause`);
    return response.data;
  } catch (error) {
    console.error('Failed to pause task:', error);
    throw error;
  }
}

// Resume a task
export async function resumeTask(taskId) {
  try {
    const response = await api.post(`/api/v1/tasks/${taskId}/resume`);
    return response.data;
  } catch (error) {
    console.error('Failed to resume task:', error);
    throw error;
  }
}

// Delete a task
export async function deleteTask(taskId) {
  try {
    const response = await api.delete(`/api/v1/tasks/${taskId}`);
    return response.data;
  } catch (error) {
    console.error('Failed to delete task:', error);
    throw error;
  }
}

// Get task logs
export async function fetchTaskLogs(taskId, limit = 100) {
  try {
    const response = await api.get(`/api/v1/tasks/${taskId}/logs`, {
      params: { limit }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch task logs:', error);
    throw error;
  }
}

// Get task metrics
export async function fetchTaskMetrics(timeframe = 'week') {
  try {
    const response = await api.get('/api/v1/tasks/metrics', {
      params: { timeframe }
    });
    return response.data;
  } catch (error) {
    console.error('Failed to fetch task metrics:', error);
    throw error;
  }
}

// WebSocket connection for real-time task updates
export function subscribeToTasks(onMessage, onError) {
  const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
  const ws = new WebSocket(`${wsUrl}/ws/tasks`);
  
  ws.onopen = () => {
    console.log('Tasks WebSocket connected');
    // Send initial subscription message
    ws.send(JSON.stringify({
      type: 'subscribe',
      channels: ['task_updates', 'task_status_changes']
    }));
  };
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (onMessage) {
        onMessage(data);
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  };
  
  ws.onerror = (error) => {
    console.error('Tasks WebSocket error:', error);
    if (onError) {
      onError(error);
    }
  };
  
  ws.onclose = () => {
    console.log('Tasks WebSocket disconnected');
  };
  
  return ws;
}

// Common task creation helpers
export async function createContentGenerationTask(contentData) {
  return createTask({
    name: 'Content Generation',
    description: `Generate content for ${contentData.topic}`,
    type: 'content_generation',
    priority: 'high',
    config: {
      topic: contentData.topic,
      business_niche: contentData.businessNiche,
      platforms: contentData.platforms,
      content_type: contentData.contentType
    }
  });
}

export async function createAnalyticsTask(analyticsConfig) {
  return createTask({
    name: 'Analytics Processing',
    description: 'Process and analyze social media data',
    type: 'analytics_processing',
    priority: 'medium',
    config: {
      timeframe: analyticsConfig.timeframe,
      platforms: analyticsConfig.platforms,
      metrics: analyticsConfig.metrics
    }
  });
}

export async function createPublishingTask(publishData) {
  return createTask({
    name: 'Content Publishing',
    description: `Publish content to ${publishData.platforms?.join(', ')}`,
    type: 'content_publishing',
    priority: 'high',
    scheduled_at: publishData.scheduledAt,
    config: {
      content: publishData.content,
      platforms: publishData.platforms,
      cross_post: publishData.crossPost
    }
  });
}

export async function createBackupTask(backupConfig) {
  return createTask({
    name: 'Data Backup',
    description: 'Backup user data and analytics',
    type: 'data_backup',
    priority: 'low',
    config: {
      backup_type: backupConfig.type,
      include_analytics: backupConfig.includeAnalytics,
      include_content: backupConfig.includeContent
    }
  });
}

// Task queue management
export async function getTaskQueue() {
  try {
    const response = await api.get('/api/v1/tasks/queue');
    return response.data;
  } catch (error) {
    console.error('Failed to get task queue:', error);
    throw error;
  }
}

export async function clearTaskQueue() {
  try {
    const response = await api.post('/api/v1/tasks/queue/clear');
    return response.data;
  } catch (error) {
    console.error('Failed to clear task queue:', error);
    throw error;
  }
}

// Task scheduling
export async function scheduleRecurringTask(taskData) {
  try {
    const response = await api.post('/api/v1/tasks/schedule', {
      ...taskData,
      recurring: true,
      schedule_pattern: taskData.schedulePattern // e.g., "daily", "weekly", "monthly"
    });
    return response.data;
  } catch (error) {
    console.error('Failed to schedule recurring task:', error);
    throw error;
  }
}

export async function getScheduledTasks() {
  try {
    const response = await api.get('/api/v1/tasks/scheduled');
    return response.data;
  } catch (error) {
    console.error('Failed to get scheduled tasks:', error);
    throw error;
  }
} 