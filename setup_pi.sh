#!/bin/bash

# V.A.R.G Setup Script for Raspberry Pi Zero W
# This script sets up the environment for food detection system

echo "Setting up V.A.R.G on Raspberry Pi Zero W..."

# Update system packages
echo " Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "🔧 Installing system dependencies (lightweight)..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-numpy \
    python3-pil \
    python3-psutil \
    python3-requests \
    python3-dotenv \
    python3-picamera2 \
    python3-opencv \
    python3-luma.oled \
    libjpeg-dev \
    libatlas-base-dev \
    zlib1g-dev \
    libpng-dev \
    libtiff-dev \
    i2c-tools \
    fonts-dejavu-core

# Enable camera interface
echo "📷 Enabling camera interface..."
sudo raspi-config nonint do_camera 0

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
# Share system packages (numpy/Pillow/etc.) with the venv to avoid slow pip builds
python3 -m venv --system-site-packages varg_env
source varg_env/bin/activate

# Configure pip to use PiWheels and reduce build pressure on low-RAM devices
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=60
export PIP_NO_CACHE_DIR=1
export PIP_INDEX_URL="https://www.piwheels.org/simple"
export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python dependencies (remaining, platform-guarded)
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt || true

# Ensure luma.oled present if apt package unavailable
python3 - << 'PY'
import importlib, subprocess, sys
try:
    importlib.import_module('luma.oled')
except Exception:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '--prefer-binary', 'luma.oled>=3.13.0'], check=False)
PY

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p detections
mkdir -p logs

# Set up environment variables template
echo "⚙️ Creating environment template..."
cat > .env.template << EOF
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# Optional: Camera settings
CAMERA_INDEX=0
DETECTION_INTERVAL=2.0
EOF

# Prompt for API key and write .env (optional)
echo "🔑 Configure Groq API key"
read -r -p " Enter your Groq API Key (leave blank to skip): " GROQ_KEY
if [ -n "$GROQ_KEY" ]; then
    echo "GROQ_API_KEY=$GROQ_KEY" > .env
    # Update config.json with the key for immediate use
    python3 - "$GROQ_KEY" << 'PY'
import json, sys, os
key = sys.argv[1]
cfg = "config.json"
try:
    if os.path.exists(cfg):
        with open(cfg, "r") as f:
            data = json.load(f)
    else:
        data = {}
    data["groq_api_key"] = key
    with open(cfg, "w") as f:
        json.dump(data, f, indent=2)
    print("Updated config.json with Groq API key")
except Exception as e:
    print(f"Warning: could not update config.json: {e}")
PY
fi

# Make the main script executable
chmod +x v.a.r.g.py

# Create systemd service file for auto-start
echo "🔧 Creating systemd service..."

# Resolve working directory and user dynamically
VARG_DIR=$(pwd)
RUN_USER=$(whoami)

sudo tee /etc/systemd/system/varg.service > /dev/null << EOF
[Unit]
Description=V.A.R.G Food Detection System
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$VARG_DIR
Environment=PATH=$VARG_DIR/varg_env/bin
EnvironmentFile=$VARG_DIR/.env
ExecStart=$VARG_DIR/varg_env/bin/python $VARG_DIR/v.a.r.g.py
Restart=always
RestartSec=10

# Resource limits for Pi Zero W
MemoryMax=400M
CPUQuota=80%

[Install]
WantedBy=multi-user.target
EOF

echo " Enabling and starting service..."
sudo systemctl daemon-reload
sudo systemctl enable varg.service
sudo systemctl start varg.service

echo " Setup complete!"
echo ""
echo " Next steps:"
echo "1. Copy .env.template to .env and add your Groq API key"
echo "2. Edit config.json if needed"
echo "3. Run the system: python3 v.a.r.g.py"
echo ""
echo " Service enabled and started. It will auto-start on boot."
echo ""
