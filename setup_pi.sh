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
    python3-numpy \
    python3-pil \
    python3-picamera2 \
    i2c-tools \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev

# Enable camera interface
echo "📷 Enabling camera interface..."
sudo raspi-config nonint do_camera 0

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv varg_env
source varg_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python dependencies (remaining, platform-guarded)
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt || true

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

# Create systemd service file for auto-start (optional)
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/varg.service > /dev/null << EOF
[Unit]
Description=V.A.R.G Food Detection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/V.A.R.G
Environment=PATH=/home/pi/V.A.R.G/varg_env/bin
EnvironmentFile=/home/pi/V.A.R.G/.env
ExecStart=/home/pi/V.A.R.G/varg_env/bin/python /home/pi/V.A.R.G/v.a.r.g.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo " Setup complete!"
echo ""
echo " Next steps:"
echo "1. Copy .env.template to .env and add your Groq API key"
echo "2. Edit config.json if needed"
echo "3. Run the system: python3 v.a.r.g.py"
echo ""
echo " Optional: Enable auto-start on boot:"
echo "   sudo systemctl enable varg.service"
echo "   sudo systemctl start varg.service"
echo ""
