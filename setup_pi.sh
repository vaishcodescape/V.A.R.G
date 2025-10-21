#!/bin/bash

# V.A.R.G Setup Script for Raspberry Pi Zero W
# This script sets up the environment for food detection system

echo "Setting up V.A.R.G on Raspberry Pi Zero W..."

# Update system packages
echo " Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies for OpenCV
echo "🔧 Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libhdf5-dev \
    libhdf5-serial-dev \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    pkg-config \
    cmake \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module

# Enable camera interface
echo "📷 Enabling camera interface..."
sudo raspi-config nonint do_camera 0

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv varg_env
source varg_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

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
echo "3. Test the camera: python3 -c 'import cv2; print(cv2.VideoCapture(0).isOpened())'"
echo "4. Run the system: python3 v.a.r.g.py"
echo ""
echo " Optional: Enable auto-start on boot:"
echo "   sudo systemctl enable varg.service"
echo "   sudo systemctl start varg.service"
echo ""
echo "📊 Monitor logs: tail -f varg.log"
