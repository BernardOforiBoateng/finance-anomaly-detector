import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Grid,
  Box,
  Button,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  LinearProgress
} from '@mui/material';
import { 
  Analytics, 
  Security, 
  TrendingUp, 
  Warning,
  PlayArrow,
  Refresh
} from '@mui/icons-material';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  PieChart, 
  Pie, 
  Cell, 
  ResponsiveContainer,
  LineChart,
  Line
} from 'recharts';
import { ApiService, Transaction, BatchFraudDetectionResponse } from '../services/ApiService';

interface DashboardProps {
  apiStatus: 'loading' | 'connected' | 'error';
}

const Dashboard: React.FC<DashboardProps> = ({ apiStatus }) => {
  const [batchResults, setBatchResults] = useState<BatchFraudDetectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [testSize, setTestSize] = useState(100);

  const runBatchTest = async (size: number = testSize) => {
    if (apiStatus !== 'connected') {
      setError('API is not connected. Please check the backend service.');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      // Generate mixed transactions (normal and suspicious)
      const transactions: Transaction[] = [];
      
      for (let i = 0; i < size; i++) {
        // 80% normal, 20% suspicious for demonstration
        if (Math.random() < 0.8) {
          transactions.push(ApiService.generateSampleTransaction());
        } else {
          transactions.push(ApiService.generateSuspiciousTransaction());
        }
      }

      const result = await ApiService.predictFraudBatch(transactions);
      setBatchResults(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to run batch test');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (apiStatus === 'connected') {
      runBatchTest(50); // Run initial test with 50 transactions
    }
  }, [apiStatus]);

  const getRiskDistribution = () => {
    if (!batchResults) return [];
    
    const distribution: { [key: string]: number } = {};
    batchResults.results.forEach(result => {
      distribution[result.risk_level] = (distribution[result.risk_level] || 0) + 1;
    });

    return Object.entries(distribution).map(([risk, count]) => ({
      risk,
      count,
      percentage: (count / batchResults.results.length * 100).toFixed(1)
    }));
  };

  const getProbabilityDistribution = () => {
    if (!batchResults) return [];
    
    const buckets = [
      { range: '0-20%', min: 0, max: 0.2, count: 0 },
      { range: '20-40%', min: 0.2, max: 0.4, count: 0 },
      { range: '40-60%', min: 0.4, max: 0.6, count: 0 },
      { range: '60-80%', min: 0.6, max: 0.8, count: 0 },
      { range: '80-100%', min: 0.8, max: 1.0, count: 0 }
    ];

    batchResults.results.forEach(result => {
      const prob = result.fraud_probability;
      const bucket = buckets.find(b => prob >= b.min && prob < b.max) || buckets[buckets.length - 1];
      bucket.count++;
    });

    return buckets;
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

  const riskColors: { [key: string]: string } = {
    'LOW': '#4caf50',
    'MEDIUM': '#ff9800', 
    'HIGH': '#f44336',
    'CRITICAL': '#d32f2f'
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Analytics />
        Fraud Detection Dashboard
      </Typography>
      
      <Typography variant="body1" color="text.secondary" paragraph>
        Monitor fraud detection performance and analyze transaction patterns in real-time.
      </Typography>

      {/* Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
            <Button
              variant="contained"
              onClick={() => runBatchTest(50)}
              disabled={loading || apiStatus !== 'connected'}
              startIcon={loading ? <LinearProgress sx={{ width: 20 }} /> : <PlayArrow />}
            >
              Test 50 Transactions
            </Button>
            <Button
              variant="outlined"
              onClick={() => runBatchTest(100)}
              disabled={loading || apiStatus !== 'connected'}
            >
              Test 100 Transactions
            </Button>
            <Button
              variant="outlined"
              onClick={() => runBatchTest(200)}
              disabled={loading || apiStatus !== 'connected'}
            >
              Test 200 Transactions
            </Button>
            <Button
              variant="text"
              onClick={() => setBatchResults(null)}
              startIcon={<Refresh />}
            >
              Clear Results
            </Button>
          </Box>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {batchResults && (
        <Grid container spacing={3}>
          {/* Summary Cards */}
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Security color="primary" />
                  <Typography variant="h6">Total Tested</Typography>
                </Box>
                <Typography variant="h3" color="primary">
                  {batchResults.summary.total_transactions}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Transactions analyzed
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Warning color="error" />
                  <Typography variant="h6">Fraud Detected</Typography>
                </Box>
                <Typography variant="h3" color="error">
                  {batchResults.summary.fraud_detected}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {((batchResults.summary.fraud_detected / batchResults.summary.total_transactions) * 100).toFixed(1)}% of total
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <TrendingUp color="success" />
                  <Typography variant="h6">Normal</Typography>
                </Box>
                <Typography variant="h3" color="success.main">
                  {batchResults.summary.normal_transactions}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {((batchResults.summary.normal_transactions / batchResults.summary.total_transactions) * 100).toFixed(1)}% of total
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Analytics color="info" />
                  <Typography variant="h6">Detection Rate</Typography>
                </Box>
                <Typography variant="h3" color="info.main">
                  {batchResults.summary.fraud_detected > 0 ? 
                    ((batchResults.summary.fraud_detected / batchResults.summary.total_transactions) * 100).toFixed(1) : 
                    '0.0'
                  }%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Fraud identification
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Charts */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Risk Level Distribution</Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={getRiskDistribution()}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ risk, percentage }) => `${risk} (${percentage}%)`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="count"
                    >
                      {getRiskDistribution().map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={riskColors[entry.risk] || COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Fraud Probability Distribution</Typography>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={getProbabilityDistribution()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </Grid>

          {/* Recent Results Table */}
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>Recent Detection Results</Typography>
                <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
                  <Table stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>Transaction ID</TableCell>
                        <TableCell>Fraud Status</TableCell>
                        <TableCell>Probability</TableCell>
                        <TableCell>Risk Level</TableCell>
                        <TableCell>Confidence</TableCell>
                        <TableCell>Timestamp</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {batchResults.results.slice(0, 20).map((result, index) => (
                        <TableRow key={index}>
                          <TableCell>{result.transaction_id || `txn_${index + 1}`}</TableCell>
                          <TableCell>
                            <Chip 
                              label={result.is_fraud ? 'FRAUD' : 'NORMAL'}
                              color={result.is_fraud ? 'error' : 'success'}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>{(result.fraud_probability * 100).toFixed(1)}%</TableCell>
                          <TableCell>
                            <Chip 
                              label={result.risk_level}
                              sx={{ bgcolor: riskColors[result.risk_level], color: 'white' }}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>
                            <Chip 
                              label={result.confidence}
                              color={result.confidence === 'HIGH' ? 'success' : 
                                     result.confidence === 'MEDIUM' ? 'warning' : 'default'}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>{new Date(result.timestamp).toLocaleTimeString()}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {!batchResults && !loading && (
        <Card>
          <CardContent sx={{ textAlign: 'center', py: 6 }}>
            <Analytics sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              No Data Available
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Run a batch test to see fraud detection analytics and insights.
            </Typography>
            <Button 
              variant="contained" 
              onClick={() => runBatchTest(50)}
              disabled={apiStatus !== 'connected'}
            >
              Start Analysis
            </Button>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default Dashboard;