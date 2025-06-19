import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Grid,
  Box,
  Chip,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress
} from '@mui/material';
import { 
  Info, 
  Security, 
  TrendingUp, 
  Speed,
  ModelTraining,
  Assessment
} from '@mui/icons-material';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';
import { ApiService, ModelInfo as ModelInfoType } from '../services/ApiService';

interface ModelInfoProps {
  apiStatus: 'loading' | 'connected' | 'error';
}

const ModelInfo: React.FC<ModelInfoProps> = ({ apiStatus }) => {
  const [modelInfo, setModelInfo] = useState<ModelInfoType | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (apiStatus === 'connected') {
      fetchModelInfo();
    }
  }, [apiStatus]);

  const fetchModelInfo = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const info = await ApiService.getModelInfo();
      setModelInfo(info);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch model information');
    } finally {
      setLoading(false);
    }
  };

  const formatPercentage = (value: number | undefined) => {
    return value ? `${(value * 100).toFixed(1)}%` : 'N/A';
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString();
    } catch {
      return dateString;
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        {error}
      </Alert>
    );
  }

  if (!modelInfo) {
    return (
      <Alert severity="info">
        Model information not available. Please ensure the API is connected.
      </Alert>
    );
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Info />
        Model Information & Performance
      </Typography>
      
      <Typography variant="body1" color="text.secondary" paragraph>
        Detailed information about the fraud detection model, its performance metrics, and feature importance.
      </Typography>

      <Grid container spacing={3}>
        {/* Model Overview */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <ModelTraining />
                Model Overview
              </Typography>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box>
                  <Typography variant="body2" color="text.secondary">Model Name</Typography>
                  <Typography variant="h6">{modelInfo.model_name}</Typography>
                </Box>
                
                <Box>
                  <Typography variant="body2" color="text.secondary">Version</Typography>
                  <Chip label={modelInfo.version} color="primary" size="small" />
                </Box>
                
                <Box>
                  <Typography variant="body2" color="text.secondary">Features Count</Typography>
                  <Typography variant="h6">{modelInfo.features_count}</Typography>
                </Box>
                
                <Box>
                  <Typography variant="body2" color="text.secondary">Training Date</Typography>
                  <Typography variant="body1">{formatDate(modelInfo.training_date)}</Typography>
                </Box>
                
                <Box>
                  <Typography variant="body2" color="text.secondary">Model Type</Typography>
                  <Chip label="XGBoost Classifier" color="success" />
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Metrics */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Assessment />
                Performance Metrics
              </Typography>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box>
                  <Typography variant="body2" color="text.secondary">ROC-AUC Score</Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LinearProgress 
                      variant="determinate" 
                      value={(modelInfo.performance_metrics.roc_auc || 0) * 100} 
                      sx={{ flexGrow: 1, height: 8, borderRadius: 1 }}
                    />
                    <Typography variant="h6" color="primary">
                      {formatPercentage(modelInfo.performance_metrics.roc_auc)}
                    </Typography>
                  </Box>
                </Box>
                
                <Box>
                  <Typography variant="body2" color="text.secondary">PR-AUC Score</Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LinearProgress 
                      variant="determinate" 
                      value={(modelInfo.performance_metrics.pr_auc || 0) * 100} 
                      sx={{ flexGrow: 1, height: 8, borderRadius: 1 }}
                      color="secondary"
                    />
                    <Typography variant="h6" color="secondary.main">
                      {formatPercentage(modelInfo.performance_metrics.pr_auc)}
                    </Typography>
                  </Box>
                </Box>
                
                <Box>
                  <Typography variant="body2" color="text.secondary">Fraud Detection Rate</Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LinearProgress 
                      variant="determinate" 
                      value={(modelInfo.performance_metrics.fraud_detection_rate || 0) * 100} 
                      sx={{ flexGrow: 1, height: 8, borderRadius: 1 }}
                      color="success"
                    />
                    <Typography variant="h6" color="success.main">
                      {formatPercentage(modelInfo.performance_metrics.fraud_detection_rate)}
                    </Typography>
                  </Box>
                </Box>
                
                <Box>
                  <Typography variant="body2" color="text.secondary">False Alarm Rate</Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LinearProgress 
                      variant="determinate" 
                      value={(modelInfo.performance_metrics.false_alarm_rate || 0) * 100} 
                      sx={{ flexGrow: 1, height: 8, borderRadius: 1 }}
                      color="warning"
                    />
                    <Typography variant="h6" color="warning.main">
                      {formatPercentage(modelInfo.performance_metrics.false_alarm_rate)}
                    </Typography>
                  </Box>
                </Box>
                
                <Box>
                  <Typography variant="body2" color="text.secondary">Precision</Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LinearProgress 
                      variant="determinate" 
                      value={(modelInfo.performance_metrics.precision || 0) * 100} 
                      sx={{ flexGrow: 1, height: 8, borderRadius: 1 }}
                      color="info"
                    />
                    <Typography variant="h6" color="info.main">
                      {formatPercentage(modelInfo.performance_metrics.precision)}
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Feature Importance Chart */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TrendingUp />
                Top 10 Feature Importance
              </Typography>
              
              <ResponsiveContainer width="100%" height={400}>
                <BarChart
                  data={modelInfo.feature_importance.slice(0, 10)}
                  margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="feature" 
                    angle={-45} 
                    textAnchor="end" 
                    height={100}
                    interval={0}
                  />
                  <YAxis />
                  <Tooltip 
                    formatter={(value: number) => [value.toFixed(4), 'Importance']}
                  />
                  <Bar dataKey="importance" fill="#1976d2" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Feature Importance Table */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Speed />
                Feature Rankings
              </Typography>
              
              <TableContainer sx={{ maxHeight: 400 }}>
                <Table stickyHeader size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Rank</TableCell>
                      <TableCell>Feature</TableCell>
                      <TableCell align="right">Importance</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {modelInfo.feature_importance.map((feature, index) => (
                      <TableRow key={feature.feature}>
                        <TableCell>
                          <Chip 
                            label={index + 1} 
                            size="small" 
                            color={index < 3 ? 'primary' : 'default'}
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" fontWeight={index < 5 ? 'bold' : 'normal'}>
                            {feature.feature}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" color={index < 5 ? 'primary' : 'text.secondary'}>
                            {feature.importance.toFixed(4)}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Technical Details */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Security />
                Technical Implementation Details
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} md={3}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">Algorithm</Typography>
                    <Typography variant="body1">XGBoost (Extreme Gradient Boosting)</Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">Problem Type</Typography>
                    <Typography variant="body1">Binary Classification</Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">Class Imbalance Handling</Typography>
                    <Typography variant="body1">Scale Pos Weight</Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} md={3}>
                  <Box>
                    <Typography variant="body2" color="text.secondary">Evaluation Metric</Typography>
                    <Typography variant="body1">ROC-AUC, PR-AUC</Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12}>
                  <Alert severity="info" sx={{ mt: 2 }}>
                    <Typography variant="body2">
                      <strong>Model Highlights:</strong> This fraud detection model achieves {formatPercentage(modelInfo.performance_metrics.fraud_detection_rate)} fraud detection rate 
                      with only {formatPercentage(modelInfo.performance_metrics.false_alarm_rate)} false alarms. 
                      The top 5 features contribute to over 60% of the model's decision-making process, 
                      indicating strong feature engineering and selection.
                    </Typography>
                  </Alert>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default ModelInfo;