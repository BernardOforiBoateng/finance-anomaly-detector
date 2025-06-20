# Use Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend and build
COPY frontend/ ./frontend/
WORKDIR /app/frontend
RUN npm install && npm run build

# Copy backend
WORKDIR /app
COPY backend/ ./backend/
COPY ml/ ./ml/

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "backend/app.py"]