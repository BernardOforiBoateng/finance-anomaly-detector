import React, { useState, useEffect } from 'react';
import { 
  Container, 
  Typography, 
  AppBar, 
  Toolbar, 
  Tabs, 
  Tab, 
  Box,
  Alert,
  Snackbar
} from '@mui/material';
import { Security, Analytics, Info } from '@mui/icons-material';
import FraudDetector from './components/FraudDetector';
import Dashboard from './components/Dashboard';
import ModelInfo from './components/ModelInfo';
import { ApiService } from './services/ApiService';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function App() {
  const [currentTab, setCurrentTab] = useState(0);
  const [apiStatus, setApiStatus] = useState<'loading' | 'connected' | 'error'>('loading');
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' | 'info' }>({
    open: false,
    message: '',
    severity: 'info'
  });

  useEffect(() => {
    checkApiConnection();
  }, []);

  const checkApiConnection = async () => {
    try {
      await ApiService.getHealth();
      setApiStatus('connected');
      setSnackbar({
        open: true,
        message: 'Successfully connected to fraud detection API',
        severity: 'success'
      });
    } catch (error) {
      setApiStatus('error');
      setSnackbar({
        open: true,
        message: 'Failed to connect to API. Please ensure the backend is running.',
        severity: 'error'
      });
    }
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  return (
    <div className="App">
      <AppBar position="static" sx={{ bgcolor: '#1976d2' }}>
        <Toolbar>
          <Security sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Personal Finance Anomaly Detector
          </Typography>
          <Box sx={{ 
            width: 12, 
            height: 12, 
            borderRadius: '50%', 
            bgcolor: apiStatus === 'connected' ? 'success.main' : 
                     apiStatus === 'error' ? 'error.main' : 'warning.main',
            mr: 1 
          }} />
          <Typography variant="body2">
            API {apiStatus === 'connected' ? 'Connected' : 
                 apiStatus === 'error' ? 'Disconnected' : 'Connecting...'}
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 2 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={currentTab} onChange={handleTabChange} aria-label="fraud detection tabs">
            <Tab 
              icon={<Security />} 
              label="Fraud Detection" 
              iconPosition="start"
            />
            <Tab 
              icon={<Analytics />} 
              label="Dashboard" 
              iconPosition="start"
            />
            <Tab 
              icon={<Info />} 
              label="Model Info" 
              iconPosition="start"
            />
          </Tabs>
        </Box>

        <TabPanel value={currentTab} index={0}>
          <FraudDetector apiStatus={apiStatus} />
        </TabPanel>
        
        <TabPanel value={currentTab} index={1}>
          <Dashboard apiStatus={apiStatus} />
        </TabPanel>
        
        <TabPanel value={currentTab} index={2}>
          <ModelInfo apiStatus={apiStatus} />
        </TabPanel>
      </Container>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </div>
  );
}

export default App;