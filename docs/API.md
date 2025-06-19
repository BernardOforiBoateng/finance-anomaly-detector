# 🚀 API Documentation

## Overview

The Personal Finance Anomaly Detector API is a RESTful web service built with FastAPI that provides real-time fraud detection capabilities for financial transactions.

**Base URL**: `https://fraud-api.railway.app` (Production) | `http://localhost:8000` (Development)

---

## 🔗 **Endpoints**

### **Health Check**

#### `GET /`
Basic health check endpoint.

**Response:**
```json
{
  "message": "Personal Finance Anomaly Detector API",
  "status": "healthy",
  "timestamp": "2024-06-19T17:30:00Z"
}
```

#### `GET /health`
Detailed health check with model status.

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "features_count": 15,
  "timestamp": "2024-06-19T17:30:00Z"
}
```

---

### **Model Information**

#### `GET /model/info`
Get comprehensive model information and performance metrics.

**Response:**
```json
{
  "model_name": "Improved XGBoost (Current_Best)",
  "version": "1.0",
  "features_count": 15,
  "training_date": "2024-06-19T17:21:09.123456",
  "performance_metrics": {
    "roc_auc": 0.922,
    "pr_auc": 0.1302,
    "fraud_detection_rate": 0.732,
    "false_alarm_rate": 0.093,
    "precision": 0.013
  },
  "feature_importance": [
    {
      "feature": "V3",
      "importance": 0.2545
    },
    {
      "feature": "Amount",
      "importance": 0.1306
    }
  ]
}
```

---

### **Fraud Detection**

#### `POST /predict`
Predict fraud probability for a single transaction.

**Request Body:**
```json
{
  "Amount": 149.62,
  "V1": -1.3598,
  "V2": -0.0727,
  "V3": 2.5363,
  "V4": 1.3781,
  "V5": -0.3383,
  "V6": 0.4623,
  "V7": 0.2394,
  "V8": 0.0986,
  "V9": 0.3637,
  "V10": 0.0906,
  "V11": -0.5515,
  "V12": -0.6178,
  "V13": -0.9912,
  "V14": -0.3112
}
```

**Response:**
```json
{
  "is_fraud": false,
  "fraud_probability": 0.0234,
  "risk_level": "LOW",
  "confidence": "HIGH",
  "timestamp": "2024-06-19T17:30:00Z"
}
```

#### `POST /predict/batch`
Predict fraud for multiple transactions in a single request.

**Request Body:**
```json
{
  "transactions": [
    {
      "Amount": 149.62,
      "V1": -1.3598,
      // ... other V features
    },
    {
      "Amount": 2500.00,
      "V1": 2.5432,
      // ... other V features
    }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "is_fraud": false,
      "fraud_probability": 0.0234,
      "risk_level": "LOW",
      "confidence": "HIGH",
      "timestamp": "2024-06-19T17:30:00Z",
      "transaction_id": "txn_1"
    },
    {
      "is_fraud": true,
      "fraud_probability": 0.8542,
      "risk_level": "CRITICAL",
      "confidence": "HIGH",
      "timestamp": "2024-06-19T17:30:00Z",
      "transaction_id": "txn_2"
    }
  ],
  "summary": {
    "total_transactions": 2,
    "fraud_detected": 1,
    "normal_transactions": 1
  }
}
```

---

### **System Statistics**

#### `GET /stats`
Get system statistics and runtime information.

**Response:**
```json
{
  "model_loaded": true,
  "features_available": 15,
  "api_version": "1.0.0",
  "uptime": "Available on startup",
  "last_updated": "2024-06-19T17:30:00Z"
}
```

---

## 📋 **Request/Response Schemas**

### **Transaction Schema**
All transaction features are required and must be numeric values.

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `Amount` | float | Transaction amount (must be positive) | 149.62 |
| `V1` - `V14` | float | PCA-transformed transaction features | -1.3598 |

### **Response Schemas**

#### **Risk Levels**
- `LOW` - Low fraud probability (< 20%)
- `MEDIUM` - Medium fraud probability (20-50%)
- `HIGH` - High fraud probability (50-80%)
- `CRITICAL` - Critical fraud probability (> 80%)

#### **Confidence Levels**
- `HIGH` - Model is very confident in prediction (> 90% or < 10%)
- `MEDIUM` - Model has moderate confidence (70-90% or 10-30%)
- `LOW` - Model has low confidence (30-70%)

---

## 🔒 **Authentication & Security**

### **CORS Policy**
The API allows cross-origin requests from:
- `http://localhost:3000` (development)
- `https://*.vercel.app` (production frontend)

### **Rate Limiting**
- No explicit rate limiting implemented
- Railway platform provides DDoS protection

### **Input Validation**
- All transaction amounts must be positive
- All V features accept any numeric value
- Missing required fields return HTTP 422

---

## 🚨 **Error Handling**

### **HTTP Status Codes**
- `200` - Success
- `422` - Validation Error (invalid input)
- `500` - Internal Server Error
- `503` - Service Unavailable (model not loaded)

### **Error Response Format**
```json
{
  "detail": "Prediction failed: Model not loaded"
}
```

---

## 🧪 **Testing Examples**

### **cURL Examples**

**Health Check:**
```bash
curl https://fraud-api.railway.app/health
```

**Single Prediction:**
```bash
curl -X POST "https://fraud-api.railway.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Amount": 149.62,
    "V1": -1.3598, "V2": -0.0727, "V3": 2.5363,
    "V4": 1.3781, "V5": -0.3383, "V6": 0.4623,
    "V7": 0.2394, "V8": 0.0986, "V9": 0.3637,
    "V10": 0.0906, "V11": -0.5515, "V12": -0.6178,
    "V13": -0.9912, "V14": -0.3112
  }'
```

### **Python Examples**

```python
import requests

# Single prediction
response = requests.post(
    "https://fraud-api.railway.app/predict",
    json={
        "Amount": 149.62,
        "V1": -1.3598, "V2": -0.0727, "V3": 2.5363,
        "V4": 1.3781, "V5": -0.3383, "V6": 0.4623,
        "V7": 0.2394, "V8": 0.0986, "V9": 0.3637,
        "V10": 0.0906, "V11": -0.5515, "V12": -0.6178,
        "V13": -0.9912, "V14": -0.3112
    }
)
result = response.json()
print(f"Fraud probability: {result['fraud_probability']:.1%}")
```

### **JavaScript Examples**

```javascript
// Using fetch API
const response = await fetch('https://fraud-api.railway.app/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    Amount: 149.62,
    V1: -1.3598, V2: -0.0727, V3: 2.5363,
    V4: 1.3781, V5: -0.3383, V6: 0.4623,
    V7: 0.2394, V8: 0.0986, V9: 0.3637,
    V10: 0.0906, V11: -0.5515, V12: -0.6178,
    V13: -0.9912, V14: -0.3112
  })
});

const result = await response.json();
console.log(`Fraud probability: ${(result.fraud_probability * 100).toFixed(1)}%`);
```

---

## 📊 **Performance Characteristics**

### **Response Times**
- Single prediction: < 200ms
- Batch prediction (100 transactions): < 2s
- Model info: < 50ms
- Health check: < 10ms

### **Throughput**
- Concurrent requests: Up to 100 simultaneous
- Transactions per second: ~500 predictions/sec
- Batch size limit: 1000 transactions per request

---

## 🔧 **Integration Guide**

### **Frontend Integration**
The API is designed to work seamlessly with the React frontend. See the `ApiService.ts` file for TypeScript interfaces and example usage.

### **Third-party Integration**
1. Obtain API endpoint URL
2. Implement authentication (if required)
3. Use the `/predict` endpoint for real-time detection
4. Handle error responses appropriately
5. Monitor using `/health` endpoint

### **Webhook Integration**
For real-time fraud alerts, implement a webhook endpoint that receives fraud detection results and triggers appropriate actions (email alerts, transaction blocking, etc.).

---

## 📚 **Additional Resources**

- **Interactive API Docs**: [https://fraud-api.railway.app/docs](https://fraud-api.railway.app/docs)
- **OpenAPI Schema**: [https://fraud-api.railway.app/openapi.json](https://fraud-api.railway.app/openapi.json)
- **Frontend Source**: [Frontend Repository](../frontend/)
- **Model Details**: [Model Documentation](MODEL.md)