import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  Alert,
  Box,
  Chip,
  CircularProgress,
  Divider,
  Paper,
  Tooltip,
  IconButton
} from '@mui/material';
import { 
  Security, 
  Warning, 
  CheckCircle, 
  Error,
  PlayArrow,
  Refresh,
  TrendingUp,
  Info,
  HelpOutline
} from '@mui/icons-material';
import { ApiService, Transaction, FraudDetectionResponse } from '../services/ApiService';

interface FraudDetectorProps {
  apiStatus: 'loading' | 'connected' | 'error';
}

const FraudDetector: React.FC<FraudDetectorProps> = ({ apiStatus }) => {
  const [transaction, setTransaction] = useState<Transaction>(ApiService.generateSampleTransaction());
  const [prediction, setPrediction] = useState<FraudDetectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (field: keyof Transaction) => (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(event.target.value);
    if (!isNaN(value)) {
      setTransaction({
        ...transaction,
        [field]: value
      });
    }
  };

  const handlePredict = async () => {
    if (apiStatus !== 'connected') {
      setError('API is not connected. Please check the backend service.');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const result = await ApiService.predictFraud(transaction);
      setPrediction(result);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to get prediction');
    } finally {
      setLoading(false);
    }
  };

  const generateNewSample = () => {
    setTransaction(ApiService.generateSampleTransaction());
    setPrediction(null);
  };

  const generateSuspiciousTransaction = () => {
    setTransaction(ApiService.generateSuspiciousTransaction());
    setPrediction(null);
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel.toUpperCase()) {
      case 'CRITICAL': return 'error';
      case 'HIGH': return 'warning';
      case 'MEDIUM': return 'info';
      case 'LOW': return 'success';
      default: return 'default';
    }
  };

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel.toUpperCase()) {
      case 'CRITICAL': return <Error />;
      case 'HIGH': return <Warning />;
      case 'MEDIUM': return <TrendingUp />;
      case 'LOW': return <CheckCircle />;
      default: return <Security />;
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Security />
        Real-time Fraud Detection
      </Typography>
      
      <Typography variant="body1" color="text.secondary" paragraph>
        Enter transaction details below to detect potential fraud using our trained XGBoost model.
      </Typography>

      <Alert severity="info" icon={<Info />} sx={{ mb: 3 }}>
        <Typography variant="body2">
          <strong>About the Input Fields:</strong> The V1-V14 fields are anonymized features created using 
          Principal Component Analysis (PCA) to protect sensitive transaction data. These mathematical 
          transformations capture patterns from original features like merchant, location, and time while 
          ensuring privacy. Use the sample buttons to see typical value ranges.
        </Typography>
      </Alert>

      <Grid container spacing={3}>
        {/* Input Form */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">Transaction Details</Typography>
                <Box>
                  <Button 
                    variant="outlined" 
                    size="small" 
                    onClick={generateNewSample}
                    sx={{ mr: 1 }}
                    startIcon={<Refresh />}
                  >
                    Normal Sample
                  </Button>
                  <Button 
                    variant="outlined" 
                    size="small" 
                    onClick={generateSuspiciousTransaction}
                    color="warning"
                    startIcon={<Warning />}
                  >
                    Suspicious Sample
                  </Button>
                </Box>
              </Box>

              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Amount ($)"
                    type="number"
                    value={transaction.Amount.toFixed(2)}
                    onChange={handleInputChange('Amount')}
                    variant="outlined"
                    helperText="Transaction amount in dollars"
                  />
                </Grid>
                
                {Array.from({ length: 14 }, (_, i) => (
                  <Grid item xs={12} sm={6} md={4} key={i}>
                    <Tooltip 
                      title={`PCA Component ${i + 1}: Anonymized feature capturing transaction patterns`}
                      placement="top"
                    >
                      <TextField
                        fullWidth
                        label={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            {`V${i + 1}`}
                            <HelpOutline sx={{ fontSize: 16, color: 'text.secondary' }} />
                          </Box>
                        }
                        type="number"
                        value={transaction[`V${i + 1}` as keyof Transaction].toFixed(4)}
                        onChange={handleInputChange(`V${i + 1}` as keyof Transaction)}
                        variant="outlined"
                        inputProps={{ step: 0.0001 }}
                        helperText="PCA-transformed feature"
                      />
                    </Tooltip>
                  </Grid>
                ))}
              </Grid>

              <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  size="large"
                  onClick={handlePredict}
                  disabled={loading || apiStatus !== 'connected'}
                  startIcon={loading ? <CircularProgress size={20} /> : <PlayArrow />}
                  sx={{ minWidth: 160 }}
                >
                  {loading ? 'Analyzing...' : 'Detect Fraud'}
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Results Panel */}
        <Grid item xs={12} md={4}>
          <Card sx={{ height: 'fit-content' }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Detection Results
              </Typography>

              {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {error}
                </Alert>
              )}

              {prediction ? (
                <Box>
                  <Paper 
                    elevation={2} 
                    sx={{ 
                      p: 2, 
                      mb: 2, 
                      bgcolor: prediction.is_fraud ? 'error.light' : 'success.light',
                      color: prediction.is_fraud ? 'error.contrastText' : 'success.contrastText'
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                      {prediction.is_fraud ? <Error /> : <CheckCircle />}
                      <Typography variant="h6">
                        {prediction.is_fraud ? 'FRAUD DETECTED' : 'TRANSACTION NORMAL'}
                      </Typography>
                    </Box>
                    <Typography variant="body2">
                      Confidence: {(prediction.fraud_probability * 100).toFixed(1)}%
                    </Typography>
                  </Paper>

                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Risk Level
                      </Typography>
                      <Chip 
                        icon={getRiskIcon(prediction.risk_level)}
                        label={prediction.risk_level}
                        color={getRiskColor(prediction.risk_level) as any}
                        variant="filled"
                      />
                    </Box>

                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Fraud Probability
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box
                          sx={{
                            width: '100%',
                            height: 8,
                            bgcolor: 'grey.300',
                            borderRadius: 1,
                            overflow: 'hidden'
                          }}
                        >
                          <Box
                            sx={{
                              width: `${prediction.fraud_probability * 100}%`,
                              height: '100%',
                              bgcolor: prediction.fraud_probability > 0.5 ? 'error.main' : 'success.main',
                              transition: 'width 0.3s ease-in-out'
                            }}
                          />
                        </Box>
                        <Typography variant="body2" fontWeight="bold">
                          {(prediction.fraud_probability * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                    </Box>

                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        Model Confidence
                      </Typography>
                      <Chip 
                        label={prediction.confidence}
                        color={prediction.confidence === 'HIGH' ? 'success' : 
                               prediction.confidence === 'MEDIUM' ? 'warning' : 'default'}
                        size="small"
                      />
                    </Box>

                    <Divider />

                    <Typography variant="caption" color="text.secondary">
                      Analysis completed at: {new Date(prediction.timestamp).toLocaleString()}
                    </Typography>
                  </Box>
                </Box>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Security sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                  <Typography variant="body2" color="text.secondary">
                    Enter transaction details and click "Detect Fraud" to analyze
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default FraudDetector;