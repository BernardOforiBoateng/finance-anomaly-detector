import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 
                     (window.location.origin.includes('localhost') ? 'http://localhost:8000' : window.location.origin);

export interface Transaction {
  Amount: number;
  V1: number;
  V2: number;
  V3: number;
  V4: number;
  V5: number;
  V6: number;
  V7: number;
  V8: number;
  V9: number;
  V10: number;
  V11: number;
  V12: number;
  V13: number;
  V14: number;
}

export interface FraudDetectionResponse {
  is_fraud: boolean;
  fraud_probability: number;
  risk_level: string;
  confidence: string;
  timestamp: string;
  transaction_id?: string;
}

export interface BatchFraudDetectionResponse {
  results: FraudDetectionResponse[];
  summary: {
    total_transactions: number;
    fraud_detected: number;
    normal_transactions: number;
  };
}

export interface ModelInfo {
  model_name: string;
  version: string;
  features_count: number;
  training_date: string;
  performance_metrics: {
    roc_auc?: number;
    pr_auc?: number;
    fraud_detection_rate?: number;
    false_alarm_rate?: number;
    precision?: number;
  };
  feature_importance: Array<{
    feature: string;
    importance: number;
  }>;
}

export interface HealthResponse {
  status: string;
  model_status: string;
  features_count: number;
  timestamp: string;
}

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export class ApiService {
  static async getHealth(): Promise<HealthResponse> {
    const response = await api.get<HealthResponse>('/health');
    return response.data;
  }

  static async predictFraud(transaction: Transaction): Promise<FraudDetectionResponse> {
    const response = await api.post<FraudDetectionResponse>('/predict', transaction);
    return response.data;
  }

  static async predictFraudBatch(transactions: Transaction[]): Promise<BatchFraudDetectionResponse> {
    const response = await api.post<BatchFraudDetectionResponse>('/predict/batch', {
      transactions
    });
    return response.data;
  }

  static async getModelInfo(): Promise<ModelInfo> {
    const response = await api.get<ModelInfo>('/model/info');
    return response.data;
  }

  static async getStats(): Promise<any> {
    const response = await api.get('/stats');
    return response.data;
  }

  // Utility method to generate sample transaction data
  static generateSampleTransaction(): Transaction {
    return {
      Amount: Math.random() * 1000,
      V1: (Math.random() - 0.5) * 4,
      V2: (Math.random() - 0.5) * 4,
      V3: (Math.random() - 0.5) * 4,
      V4: (Math.random() - 0.5) * 4,
      V5: (Math.random() - 0.5) * 4,
      V6: (Math.random() - 0.5) * 4,
      V7: (Math.random() - 0.5) * 4,
      V8: (Math.random() - 0.5) * 4,
      V9: (Math.random() - 0.5) * 4,
      V10: (Math.random() - 0.5) * 4,
      V11: (Math.random() - 0.5) * 4,
      V12: (Math.random() - 0.5) * 4,
      V13: (Math.random() - 0.5) * 4,
      V14: (Math.random() - 0.5) * 4,
    };
  }

  // Generate suspicious transaction (more likely to be fraud)
  static generateSuspiciousTransaction(): Transaction {
    return {
      Amount: Math.random() * 5000 + 1000, // Higher amounts
      V1: (Math.random() - 0.5) * 8,       // More extreme values
      V2: (Math.random() - 0.5) * 8,
      V3: Math.random() * 6 - 3,           // V3 is important feature
      V4: (Math.random() - 0.5) * 6,
      V5: (Math.random() - 0.5) * 6,
      V6: (Math.random() - 0.5) * 6,
      V7: (Math.random() - 0.5) * 6,
      V8: (Math.random() - 0.5) * 6,
      V9: (Math.random() - 0.5) * 6,
      V10: (Math.random() - 0.5) * 6,
      V11: (Math.random() - 0.5) * 6,
      V12: (Math.random() - 0.5) * 6,
      V13: (Math.random() - 0.5) * 6,
      V14: (Math.random() - 0.5) * 6,
    };
  }
}