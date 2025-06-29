import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  Button,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  Stop,
  Refresh,
  Schedule,
  CheckCircle,
  Error,
  Warning,
  Info,
  Delete,
  Visibility,
} from '@mui/icons-material';
import useTasksStore from '../../store/tasksStore';

const Tasks = () => {
  const { tasks, loading, error, fetchTasks, fetchTaskStatus, subscribeTasks } = useTasksStore();
  const [selectedTask, setSelectedTask] = useState(null);
  const [openDetailsDialog, setOpenDetailsDialog] = useState(false);
  const [wsConnection, setWsConnection] = useState(null);

  useEffect(() => {
    fetchTasks();
    
    // Subscribe to real-time updates
    const ws = subscribeTasks((data) => {
      // Handle real-time task updates
      console.log('Task update received:', data);
      // Refresh tasks list when updates are received
      fetchTasks();
    });
    
    setWsConnection(ws);

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [fetchTasks, subscribeTasks]);

  const handleViewDetails = async (taskId) => {
    const taskDetails = await fetchTaskStatus(taskId);
    setSelectedTask(taskDetails);
    setOpenDetailsDialog(true);
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle color="success" />;
      case 'running':
        return <PlayArrow color="primary" />;
      case 'failed':
        return <Error color="error" />;
      case 'pending':
        return <Schedule color="warning" />;
      case 'paused':
        return <Pause color="warning" />;
      default:
        return <Info color="info" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'success';
      case 'running':
        return 'primary';
      case 'failed':
        return 'error';
      case 'pending':
        return 'warning';
      case 'paused':
        return 'warning';
      default:
        return 'default';
    }
  };

  const getProgressValue = (task) => {
    if (task.status === 'completed') return 100;
    if (task.status === 'failed') return 0;
    return task.progress || 0;
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        Failed to load tasks: {error.message}
      </Alert>
    );
  }

  const runningTasks = tasks.filter(t => t.status === 'running').length;
  const completedTasks = tasks.filter(t => t.status === 'completed').length;
  const failedTasks = tasks.filter(t => t.status === 'failed').length;
  const pendingTasks = tasks.filter(t => t.status === 'pending').length;

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Background Tasks
      </Typography>

      {/* Task Stats */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Running Tasks
              </Typography>
              <Typography variant="h4" color="primary">
                {runningTasks}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Completed
              </Typography>
              <Typography variant="h4" color="success.main">
                {completedTasks}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Failed
              </Typography>
              <Typography variant="h4" color="error.main">
                {failedTasks}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Pending
              </Typography>
              <Typography variant="h4" color="warning.main">
                {pendingTasks}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tasks List */}
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">
              Task Queue ({tasks.length} total)
            </Typography>
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={() => fetchTasks()}
            >
              Refresh
            </Button>
          </Box>
          
          {tasks.length > 0 ? (
            <List>
              {tasks.map((task) => (
                <ListItem key={task.id} divider>
                  <ListItemIcon>
                    {getStatusIcon(task.status)}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography variant="subtitle1">
                          {task.name}
                        </Typography>
                        <Chip
                          label={task.status}
                          color={getStatusColor(task.status)}
                          size="small"
                        />
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="body2" color="textSecondary">
                          {task.description}
                        </Typography>
                        <Box sx={{ mt: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={getProgressValue(task)}
                            sx={{ mb: 1 }}
                          />
                          <Typography variant="caption" color="textSecondary">
                            {task.progress || 0}% complete
                            {task.estimatedTime && ` • Est. ${task.estimatedTime}`}
                          </Typography>
                        </Box>
                        {task.createdAt && (
                          <Typography variant="caption" color="textSecondary" display="block">
                            Created: {new Date(task.createdAt).toLocaleString()}
                          </Typography>
                        )}
                      </Box>
                    }
                  />
                  <ListItemSecondaryAction>
                    <IconButton
                      edge="end"
                      onClick={() => handleViewDetails(task.id)}
                    >
                      <Visibility />
                    </IconButton>
                    {task.status === 'running' && (
                      <IconButton edge="end" color="warning">
                        <Pause />
                      </IconButton>
                    )}
                    {task.status === 'paused' && (
                      <IconButton edge="end" color="primary">
                        <PlayArrow />
                      </IconButton>
                    )}
                    {(task.status === 'running' || task.status === 'pending') && (
                      <IconButton edge="end" color="error">
                        <Stop />
                      </IconButton>
                    )}
                    <IconButton edge="end" color="error">
                      <Delete />
                    </IconButton>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          ) : (
            <Typography color="textSecondary" align="center" sx={{ py: 4 }}>
              No tasks found. Background tasks will appear here when they are created.
            </Typography>
          )}
        </CardContent>
      </Card>

      {/* Task Details Dialog */}
      <Dialog open={openDetailsDialog} onClose={() => setOpenDetailsDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          Task Details: {selectedTask?.name}
        </DialogTitle>
        <DialogContent>
          {selectedTask && (
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <Typography variant="subtitle2" color="textSecondary">
                  Status
                </Typography>
                <Box display="flex" alignItems="center" gap={1}>
                  {getStatusIcon(selectedTask.status)}
                  <Chip
                    label={selectedTask.status}
                    color={getStatusColor(selectedTask.status)}
                  />
                </Box>
              </Grid>
              
              <Grid item xs={12}>
                <Typography variant="subtitle2" color="textSecondary">
                  Description
                </Typography>
                <Typography variant="body2">
                  {selectedTask.description}
                </Typography>
              </Grid>

              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" color="textSecondary">
                  Progress
                </Typography>
                <Box sx={{ mt: 1 }}>
                  <LinearProgress
                    variant="determinate"
                    value={getProgressValue(selectedTask)}
                    sx={{ mb: 1 }}
                  />
                  <Typography variant="body2">
                    {selectedTask.progress || 0}% complete
                  </Typography>
                </Box>
              </Grid>

              <Grid item xs={12} sm={6}>
                <Typography variant="subtitle2" color="textSecondary">
                  Estimated Time
                </Typography>
                <Typography variant="body2">
                  {selectedTask.estimatedTime || 'Not available'}
                </Typography>
              </Grid>

              {selectedTask.createdAt && (
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Created
                  </Typography>
                  <Typography variant="body2">
                    {new Date(selectedTask.createdAt).toLocaleString()}
                  </Typography>
                </Grid>
              )}

              {selectedTask.startedAt && (
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Started
                  </Typography>
                  <Typography variant="body2">
                    {new Date(selectedTask.startedAt).toLocaleString()}
                  </Typography>
                </Grid>
              )}

              {selectedTask.completedAt && (
                <Grid item xs={12} sm={6}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Completed
                  </Typography>
                  <Typography variant="body2">
                    {new Date(selectedTask.completedAt).toLocaleString()}
                  </Typography>
                </Grid>
              )}

              {selectedTask.error && (
                <Grid item xs={12}>
                  <Typography variant="subtitle2" color="error">
                    Error Details
                  </Typography>
                  <TextField
                    fullWidth
                    multiline
                    rows={3}
                    value={selectedTask.error}
                    variant="outlined"
                    size="small"
                    InputProps={{ readOnly: true }}
                  />
                </Grid>
              )}

              {selectedTask.result && (
                <Grid item xs={12}>
                  <Typography variant="subtitle2" color="textSecondary">
                    Result
                  </Typography>
                  <TextField
                    fullWidth
                    multiline
                    rows={3}
                    value={JSON.stringify(selectedTask.result, null, 2)}
                    variant="outlined"
                    size="small"
                    InputProps={{ readOnly: true }}
                  />
                </Grid>
              )}
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDetailsDialog(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Tasks; 