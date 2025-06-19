#!/bin/bash
set -e

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Installing Node.js dependencies..."
cd frontend
npm ci --only=production
echo "Building React frontend..."
npm run build
cd ..

echo "Build completed successfully!"