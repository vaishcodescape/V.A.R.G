#!/bin/bash

# V.A.R.G Deployment Script for Raspberry Pi Zero W
# Optimized for direct deployment without API overhead

echo "V.A.R.G Deployment for Raspberry Pi Zero W"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running on Raspberry Pi
check_raspberry_pi() {
    if [[ ! -f /proc/device-tree/model ]] || ! grep -q "Raspberry Pi" /proc/device-tree/model; then
        print_error "This script should be run on a Raspberry Pi"
        exit 1
    fi
    
    PI_MODEL=$(tr -d '\0' </proc/device-tree/model)
    print_status "Detected: $PI_MODEL"
}

# Optimize Pi Zero W settings
optimize_pi_zero() {
    print_step "Optimizing Raspberry Pi Zero W settings..."
    
    # Determine boot config path (Bookworm uses /boot/firmware)
    CONFIG_FILE=/boot/config.txt
    if [ -f /boot/firmware/config.txt ]; then
        CONFIG_FILE=/boot/firmware/config.txt
    fi
    
    # Increase GPU memory split for camera operations
    if ! grep -q "^gpu_mem=128" "$CONFIG_FILE"; then
        echo "gpu_mem=128" | sudo tee -a "$CONFIG_FILE" >/dev/null
        print_status "Set GPU memory to 128MB"
    fi
    
    # Enable camera interface
    sudo raspi-config nonint do_camera 0
    print_status "Camera interface enabled"
    
    # Enable I2C for OLED display
    sudo raspi-config nonint do_i2c 0
    print_status "I2C interface enabled"
    
    # Optimize for performance (use cautiously on Pi Zero W)
    if ! grep -q "^arm_freq=1000" "$CONFIG_FILE"; then
        echo "arm_freq=1000" | sudo tee -a "$CONFIG_FILE" >/dev/null
        print_status "Set ARM frequency to 1000MHz"
    fi
    
    # Disable unnecessary services to save resources
    print_step "Disabling unnecessary services..."
    sudo systemctl disable bluetooth
    sudo systemctl disable wifi-powersave@wlan0.service 2>/dev/null || true
    
    print_status "Pi Zero W optimization complete"
}

# Install system dependencies
install_system_deps() {
    print_step "Installing system dependencies..."
    
    sudo DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::ForceIPv4=true update -yq

    # Helper: install a package if available; warn and continue if not
    apt_install_safe() {
        local pkg="$1"
		print_status "Installing: $pkg"
		if sudo DEBIAN_FRONTEND=noninteractive apt-get -o Dpkg::Progress-Fancy=1 -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::ForceIPv4=true install -y --no-install-recommends "$pkg"; then
			print_status "✅ Installed: $pkg"
            return 0
        else
            print_warning "Package not available or failed: $pkg, retrying with --fix-missing"
            # Retry once with fix-missing and a fresh update
            sudo DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::ForceIPv4=true update -yq >/dev/null 2>&1 || true
			if sudo DEBIAN_FRONTEND=noninteractive apt-get -o Dpkg::Progress-Fancy=1 -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::ForceIPv4=true install -y --no-install-recommends --fix-missing "$pkg"; then
				print_status "✅ Installed on retry: $pkg"
                return 0
            fi
            print_warning "Package not available or failed: $pkg (continuing)"
            return 1
        fi
    }

	# Helper: install a batch of packages quickly; fallback to per-package on failure
	apt_install_batch() {
		local pkgs=("$@")
		local count=${#pkgs[@]}
		if [ "$count" -eq 0 ]; then
			return 0
		fi
		print_step "Installing $count packages in batch (faster)..."
		printf "%s\n" "${pkgs[@]}" | sed 's/^/- /'
		if sudo DEBIAN_FRONTEND=noninteractive apt-get -o Dpkg::Progress-Fancy=1 -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::ForceIPv4=true install -y --no-install-recommends "${pkgs[@]}"; then
			print_status "Batch install completed"
			return 0
		fi
		print_warning "Batch install failed; retrying with --fix-missing"
		sudo DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::ForceIPv4=true update -yq || true
		if sudo DEBIAN_FRONTEND=noninteractive apt-get -o Dpkg::Progress-Fancy=1 -o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::ForceIPv4=true install -y --no-install-recommends --fix-missing "${pkgs[@]}"; then
			print_status "Batch install completed on retry"
			return 0
		fi
		print_warning "Batch still failing; falling back to per-package installs"
		local p
		for p in "${pkgs[@]}"; do
			apt_install_safe "$p"
		done
		return 0
	}

    # Base packages (lightweight, broadly available)
    BASE_PKGS=(
        python3-pip
        python3-venv
        python3-dev
        libopenblas-dev
        liblapack-dev
        libjpeg-dev
        libpng-dev
        libtiff-dev
        libv4l-dev
        libfontconfig1-dev
        libcairo2-dev
        libgdk-pixbuf-2.0-dev
        libpango1.0-dev
        libgtk-3-dev
        pkg-config
        gfortran
        libhdf5-dev
        libhdf5-serial-dev
        python3-pyqt5
        python3-h5py
        i2c-tools
    )

	# Install base in batch for speed
	apt_install_batch "${BASE_PKGS[@]}"

    # Media codecs (optional)
    OPTIONAL_MEDIA=(libavcodec-dev libavformat-dev libswscale-dev libxvidcore-dev libx264-dev)
	apt_install_batch "${OPTIONAL_MEDIA[@]}"

	# Legacy/renamed packages (skip on modern distros)
    LEGACY_PKGS=(libqtgui4 libqt4-test libgtk2.0-dev libhdf5-103)
	for p in "${LEGACY_PKGS[@]}"; do
		apt_install_safe "$p" || true
	done

	# Occasionally missing on newer releases; treat as optional
	OPTIONAL_MISC=(libjasper-dev)
	for p in "${OPTIONAL_MISC[@]}"; do
		apt_install_safe "$p" || true
	done

    # Picamera2 (fallback to libcamera if unavailable)
	if ! apt_install_safe python3-picamera2; then
        print_warning "Falling back to libcamera tools (USB/OpenCV fallback still supported)"
		apt_install_safe python3-libcamera || true
		apt_install_safe libcamera-apps || true
    fi

    print_status "System dependencies step completed"
}

# Setup Python environment
setup_python_env() {
    print_step "Setting up Python environment..."
    
    # Create virtual environment
    python3 -m venv varg_env
    source varg_env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Use our dependency manager for robust installation
    print_status "Running V.A.R.G dependency manager..."
    python3 install_dependencies.py
    
    # Install from requirements.txt as backup
    print_status "Installing from requirements.txt..."
    pip install -r requirements.txt
    
    print_status "Python environment setup complete"
}

# Setup TensorFlow Lite models
setup_models() {
    print_step "Setting up TensorFlow Lite models..."
    
    # Run model setup script
    python3 setup_models.py
    
    print_status "Models setup complete"
}

# Create systemd service for auto-start
create_service() {
    print_step "Creating systemd service..."
    
    VARG_DIR=$(pwd)
    USER=$(whoami)
    
    sudo tee /etc/systemd/system/varg.service > /dev/null << EOF
[Unit]
Description=V.A.R.G Food Detection System
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$VARG_DIR
Environment=PATH=$VARG_DIR/varg_env/bin
EnvironmentFile=$VARG_DIR/.env
ExecStart=$VARG_DIR/varg_env/bin/python $VARG_DIR/v.a.r.g.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits for Pi Zero W
MemoryMax=400M
CPUQuota=80%

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable varg.service

    # Start only if GROQ_API_KEY is present (user may add it manually later)
    if [ -f "$VARG_DIR/.env" ] && grep -qE '^GROQ_API_KEY=.+$' "$VARG_DIR/.env"; then
        sudo systemctl start varg.service
        print_status "Systemd service created, enabled, and started"
    else
        print_warning "GROQ_API_KEY not set yet; service enabled but not started. Add key to .env and run: sudo systemctl start varg.service"
    fi
}

# Create startup script
create_startup_script() {
    print_step "Creating startup script..."
    
    cat > start_varg.sh << 'EOF'
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
EOF
    
    chmod +x start_varg.sh
    print_status "Startup script created"
}

# Create monitoring script
create_monitoring_script() {
    print_step "Creating monitoring script..."
    
    cat > monitor_varg.sh << 'EOF'
#!/bin/bash

# V.A.R.G Monitoring Script
echo "📊 V.A.R.G System Monitor"
echo "========================"

# Check service status
echo "🔍 Service Status:"
sudo systemctl status varg.service --no-pager -l

echo ""
echo "📈 System Resources:"
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory Usage: $(free | grep Mem | awk '{printf("%.1f%%\n", $3/$2 * 100.0)}')"
echo "Temperature: $(vcgencmd measure_temp | cut -d'=' -f2)"

echo ""
echo "📋 Recent Logs:"
sudo journalctl -u varg.service --no-pager -n 10

echo ""
echo "🔧 Control Commands:"
echo "  Start:   sudo systemctl start varg.service"
echo "  Stop:    sudo systemctl stop varg.service"
echo "  Restart: sudo systemctl restart varg.service"
echo "  Logs:    sudo journalctl -u varg.service -f"
EOF
    
    chmod +x monitor_varg.sh
    print_status "Monitoring script created"
}

# Main deployment function
main() {
    print_step "Starting V.A.R.G deployment..."
    
    # Check if running on Raspberry Pi
    check_raspberry_pi
    
    # Optimize Pi Zero W
    optimize_pi_zero
    
    # Install system dependencies
    install_system_deps
    
    # Setup Python environment
    setup_python_env
    
    # Setup models
    setup_models

    print_step "Creating configuration files..."
    # Create .env template if it doesn't exist (before service creation)
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# Groq API Configuration
GROQ_API_KEY=

# Optional: Override config settings
# CAMERA_INDEX=0
# DETECTION_INTERVAL=3.0
EOF
        print_warning "Created .env template. Add your Groq API key if using LLM."
    fi

    # Create service files
    create_service
    create_startup_script
    create_monitoring_script
    
    echo ""
    print_status "V.A.R.G deployment complete!"
    echo ""
    echo "🔧 Next Steps:"
    echo "1. Edit .env and set GROQ_API_KEY (optional)"
    echo "2. Monitor service: ./monitor_varg.sh or 'sudo journalctl -u varg.service -f'"
    echo "3. Reboot recommended to apply firmware config: sudo reboot"
    echo ""
    echo "📊 Performance Tips for Pi Zero W:"
    echo "- System will automatically optimize based on CPU/memory usage"
    echo "- TensorFlow Lite models provide best accuracy with minimal overhead"
    echo "- OLED display shows real-time status and detection results"
    echo "- Logs are available via: sudo journalctl -u varg.service -f"
    echo ""
    print_warning "Reboot recommended to apply all optimizations: sudo reboot"
}

# Run main function
main "$@"