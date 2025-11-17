#!/bin/bash

# V.A.R.G Startup Script
echo "Starting V.A.R.G Food Detection System..."

# Activate virtual environment
source varg_env/bin/activate

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models)" ]; then
    echo "⚠️  No models found. Setting up models..."
    python3 setup_models.py
fi

# Check configuration
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Please create one with your Groq API key:"
    echo "GROQ_API_KEY=your_api_key_here"
    exit 1
fi

# Start V.A.R.G
echo "Launching V.A.R.G..."
python3 v.a.r.g.py
