#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js and build frontend
cd frontend
npm install
npm run build
cd ..

echo "Build completed successfully!"