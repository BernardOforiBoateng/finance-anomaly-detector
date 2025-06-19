# 🚀 Deployment Guide

## Quick Deploy Links

### Option 1: One-Click Deploy (Recommended)

#### Backend (Railway)
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template)

#### Frontend (Vercel)  
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)

---

## Manual Deployment Steps

### 🔧 **Backend Deployment (Railway)**

1. **Create Railway Account** → [railway.app](https://railway.app)
2. **Connect GitHub** → Link your repository
3. **Create New Project** → "Deploy from GitHub repo"
4. **Select Repository** → `finance-anomaly-detector`
5. **Choose Service** → Select `backend/` folder
6. **Deploy** → Railway auto-detects Dockerfile
7. **Get URL** → Copy the generated Railway URL

**Environment Variables:**
```
PORT=8000
PYTHONPATH=/app
```

### 🎨 **Frontend Deployment (Vercel)**

1. **Create Vercel Account** → [vercel.com](https://vercel.com)
2. **Import Project** → Connect GitHub repo
3. **Select Root Directory** → Choose `frontend/`
4. **Configure Build**:
   - Build Command: `npm run build`
   - Output Directory: `build`
5. **Environment Variables**:
   ```
   REACT_APP_API_URL=https://your-railway-app.railway.app
   ```
6. **Deploy** → Get your Vercel URL

---

## 🧪 **Testing Deployment**

### Backend Test:
```bash
curl https://your-railway-app.railway.app/health
```

### Frontend Test:
Open `https://your-vercel-app.vercel.app` in browser

---

## 🔧 **Local Development**

### Backend:
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend:
```bash
cd frontend  
npm install
npm start
```

---

## 📝 **Configuration Files**

- `backend/Dockerfile` → Railway deployment
- `backend/railway.toml` → Railway configuration  
- `frontend/vercel.json` → Vercel routing
- `frontend/.env.example` → Environment template

---

## 🚨 **Important Notes**

1. **Update CORS**: Add your Vercel URL to backend CORS origins
2. **Environment Variables**: Set `REACT_APP_API_URL` to Railway URL
3. **Model Files**: Ensure ML models are included in backend folder
4. **Free Limits**: 
   - Railway: 500 hours/month
   - Vercel: Unlimited for personal projects

---

## 🎯 **Live URLs** (Update after deployment)

- **API**: `https://your-app.railway.app`
- **Dashboard**: `https://your-app.vercel.app`
- **Health Check**: `https://your-app.railway.app/health`