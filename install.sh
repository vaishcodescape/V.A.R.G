#!/bin/bash

################################################################################
# V.A.R.G Automated Installation Script
# One-command setup for Raspberry Pi Zero W and development environments
################################################################################

# Usage: ./install.sh [--dry-run] [--skip-models] [--skip-service]

set -e  # Exit on error

# Parse command line arguments
DRY_RUN=false
SKIP_MODELS=false
SKIP_SERVICE=false

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --skip-service)
            SKIP_SERVICE=true
            shift
            ;;
        --help|-h)
            echo "V.A.R.G Installation Script"
            echo ""
            echo "Usage: ./install.sh [options]"
            echo ""
            echo "Options:"
            echo "  --dry-run       Show what would be installed without making changes"
            echo "  --skip-models   Skip TensorFlow Lite model downloads"
            echo "  --skip-service  Skip systemd service setup (manual start only)"
            echo "  --help, -h      Show this help message"
            echo ""
            exit 0
            ;;
        *)
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print functions
print_banner() {
    echo -e "${CYAN}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  V.A.R.G - Virtual Augmented Reality Glasses"
    echo "  Installation Script"
    echo "═══════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[→]${NC} $1"
}

print_info() {
    echo -e "${CYAN}[ℹ]${NC} $1"
}

# Detect platform
detect_platform() {
    print_step "Detecting platform..."
    
    # Check if Raspberry Pi
    if [[ -f /proc/cpuinfo ]] && grep -q "BCM\|ARM" /proc/cpuinfo 2>/dev/null; then
        IS_RPI=true
        if [[ -f /proc/device-tree/model ]]; then
            PI_MODEL=$(tr -d '\0' </proc/device-tree/model)
            print_info "Detected: $PI_MODEL"
        else
            print_info "Detected: Raspberry Pi (model unknown)"
        fi
    else
        IS_RPI=false
        print_info "Detected: $(uname -s) on $(uname -m)"
        print_warning "Not running on Raspberry Pi - will set up development environment"
    fi
}

# Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    local errors=0
    
    # Check Python 3
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        ((errors++))
    else
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python $PYTHON_VERSION found"
        
        # Check Python version is 3.7+
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 7 ]); then
            print_error "Python 3.7+ required, found $PYTHON_VERSION"
            ((errors++))
        fi
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
        print_error "pip is not installed"
        ((errors++))
    else
        print_status "pip found"
    fi
    
    # Check git (optional but recommended)
    if command -v git &> /dev/null; then
        print_status "git found"
    else
        print_warning "git not found - some features may be limited"
    fi
    
    # Check disk space
    if command -v df &> /dev/null; then
        AVAILABLE_MB=$(df -m . | tail -1 | awk '{print $4}')
        if [ "$AVAILABLE_MB" -lt 500 ]; then
            print_warning "Low disk space: ${AVAILABLE_MB}MB available (recommend 500MB+)"
        else
            print_status "Disk space: ${AVAILABLE_MB}MB available"
        fi
    fi
    
    # Exit if there were critical errors
    if [ $errors -gt 0 ]; then
        print_error "Prerequisites check failed with $errors error(s)"
        exit 1
    fi
}

# Install system dependencies (Raspberry Pi only)
install_system_deps_pi() {
    print_step "Installing system dependencies for Raspberry Pi..."
    
    # Update package lists
    print_info "Updating package lists..."
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -yq
    
    # Install packages in batch with retry logic
    PACKAGES=(
        python3-pip
        python3-venv
        python3-dev
        python3-numpy
        python3-pil
        python3-psutil
        python3-requests
        python3-dotenv
        python3-rpi.gpio
        python3-spidev
        libatlas-base-dev
        libjpeg-dev
        zlib1g-dev
        libpng-dev
        i2c-tools
        fonts-dejavu-core
    )
    
    # Optional packages (won't fail if unavailable)
    OPTIONAL_PACKAGES=(
        python3-picamera2
        python3-opencv
    )
    
    print_info "Installing core packages..."
    if sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${PACKAGES[@]}"; then
        print_status "Core packages installed successfully"
    else
        print_warning "Some core packages failed to install, retrying with fix-missing..."
        sudo DEBIAN_FRONTEND=noninteractive apt-get update -yq
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --fix-missing "${PACKAGES[@]}" || {
            print_error "Failed to install core packages"
            exit 1
        }
    fi
    
    print_info "Installing optional packages..."
    for pkg in "${OPTIONAL_PACKAGES[@]}"; do
        if sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$pkg"; then
            print_status "Installed $pkg"
        else
            print_warning "Could not install $pkg (optional, continuing)"
        fi
    done
    
    print_status "System dependencies installed"
}

# Enable hardware interfaces (Raspberry Pi only)
enable_hardware_interfaces() {
    print_step "Enabling hardware interfaces..."
    
    # Enable camera
    print_info "Enabling camera interface..."
    sudo raspi-config nonint do_camera 0 2>/dev/null || print_warning "Could not enable camera interface"
    
    # Enable I2C
    print_info "Enabling I2C interface..."
    sudo raspi-config nonint do_i2c 0 2>/dev/null || print_warning "Could not enable I2C interface"
    
    # Enable SPI
    print_info "Enabling SPI interface..."
    sudo raspi-config nonint do_spi 0 2>/dev/null || print_warning "Could not enable SPI interface"
    
    # Optimize GPU memory for camera
    CONFIG_FILE=/boot/config.txt
    if [ -f /boot/firmware/config.txt ]; then
        CONFIG_FILE=/boot/firmware/config.txt
    fi
    
    if [ -f "$CONFIG_FILE" ]; then
        if ! grep -q "^gpu_mem=" "$CONFIG_FILE"; then
            print_info "Setting GPU memory to 128MB..."
            echo "gpu_mem=128" | sudo tee -a "$CONFIG_FILE" >/dev/null
        fi
    fi
    
    print_status "Hardware interfaces configured"
}

# Create Python virtual environment
create_venv() {
    print_step "Creating Python virtual environment..."
    
    if [ "$IS_RPI" = true ]; then
        # On Pi, use system site packages to avoid recompiling numpy, etc.
        python3 -m venv --system-site-packages varg_env
    else
        # On other platforms, use isolated venv
        python3 -m venv varg_env
    fi
    
    print_status "Virtual environment created: varg_env/"
}

# Activate virtual environment and configure pip
setup_pip() {
    print_step "Configuring pip..."
    
    # Activate virtual environment
    source varg_env/bin/activate
    
    # Configure pip for faster installs
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PIP_DEFAULT_TIMEOUT=60
    export PIP_NO_CACHE_DIR=1
    
    if [ "$IS_RPI" = true ]; then
        # Use PiWheels for Raspberry Pi
        export PIP_INDEX_URL="https://www.piwheels.org/simple"
        export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"
        print_info "Configured pip to use PiWheels"
    fi
    
    # Upgrade pip, setuptools, wheel
    print_info "Upgrading pip, setuptools, and wheel..."
    pip install --upgrade pip setuptools wheel --prefer-binary --no-cache-dir || {
        print_warning "Failed to upgrade pip tools, continuing..."
    }
    
    print_status "pip configured"
}

# Install Python dependencies
install_python_deps() {
    print_step "Installing Python dependencies..."
    
    # Ensure venv is activated
    if [ -z "$VIRTUAL_ENV" ]; then
        source varg_env/bin/activate
    fi
    
    # Install from requirements.txt
    if [ -f "requirements.txt" ]; then
        print_info "Installing packages from requirements.txt..."
        
        # Try wheels-only first for speed
        if pip install --prefer-binary --no-cache-dir --only-binary :all: -r requirements.txt 2>/dev/null; then
            print_status "All dependencies installed from wheels"
        else
            print_warning "Some packages not available as wheels, allowing source builds..."
            pip install --prefer-binary --no-cache-dir -r requirements.txt || {
                print_warning "Some packages failed to install, trying line-by-line..."
                
                # Install line by line to isolate failures
                while IFS= read -r line; do
                    # Skip comments and empty lines
                    [[ "$line" =~ ^#.*$ ]] && continue
                    [[ -z "$line" ]] && continue
                    
                    print_info "Installing: $line"
                    pip install --prefer-binary --no-cache-dir "$line" || {
                        print_warning "Failed to install $line (continuing)"
                    }
                done < requirements.txt
            }
        fi
    else
        print_warning "requirements.txt not found, skipping Python package installation"
    fi
    
    # Run dependency manager for additional setup
    if [ -f "install_dependencies.py" ]; then
        print_info "Running dependency manager..."
        python3 install_dependencies.py || print_warning "Dependency manager completed with warnings"
    fi
    
    print_status "Python dependencies installed"
}

# Setup Waveshare OLED library
setup_waveshare_oled() {
    print_step "Setting up Waveshare OLED library..."
    
    # Check if Raspberry directory exists
    if [ -d "Raspberry/python" ]; then
        print_info "Found Waveshare library directory"
        
        # Check if lib directory exists
        if [ -d "Raspberry/python/lib" ]; then
            print_status "Waveshare OLED library is ready"
        else
            print_warning "Waveshare OLED lib directory not found"
            
            # Try to extract from archive if available
            if [ -f "OLED_Module_Code.7z" ]; then
                print_info "Found OLED archive, attempting extraction..."
                
                # Try various extraction tools
                if command -v 7z &> /dev/null; then
                    7z x OLED_Module_Code.7z -y
                    print_status "Archive extracted with 7z"
                elif command -v 7za &> /dev/null; then
                    7za x OLED_Module_Code.7z -y
                    print_status "Archive extracted with 7za"
                elif python3 -c "import py7zr" 2>/dev/null; then
                    python3 -c "import py7zr; py7zr.SevenZipFile('OLED_Module_Code.7z', mode='r').extractall()"
                    print_status "Archive extracted with py7zr"
                else
                    print_warning "No extraction tool found for .7z files"
                    print_info "Install with: sudo apt-get install p7zip-full  (or)  pip install py7zr"
                fi
            else
                print_warning "OLED archive not found, manual setup may be required"
            fi
        fi
    else
        print_warning "Raspberry directory not found, creating structure..."
        mkdir -p Raspberry/python/lib
        print_info "Please place Waveshare OLED library in Raspberry/python/lib/"
    fi
    
    # Run dedicated setup script if available
    if [ -f "setup_waveshare_oled.sh" ]; then
        print_info "Running Waveshare OLED setup script..."
        chmod +x setup_waveshare_oled.sh
        ./setup_waveshare_oled.sh || print_warning "OLED setup script completed with warnings"
    fi
    
    print_status "Waveshare OLED setup complete"
}

# Setup TensorFlow Lite models
setup_models() {
    if [ "$SKIP_MODELS" = true ]; then
        print_warning "Skipping model setup (--skip-models flag)"
        mkdir -p models
        return
    fi
    
    print_step "Setting up TensorFlow Lite models..."
    
    # Ensure venv is activated
    if [ -z "$VIRTUAL_ENV" ]; then
        source varg_env/bin/activate
    fi
    
    # Create models directory
    mkdir -p models
    
    # Run model setup script
    if [ -f "setup_models.py" ]; then
        print_info "Running model setup (this may take a few minutes)..."
        
        # Run in non-interactive mode
        python3 setup_models.py --non-interactive || {
            print_warning "Model setup completed with warnings"
        }
    else
        print_warning "setup_models.py not found, models must be set up manually"
    fi
    
    print_status "Model setup complete"
}

# Create configuration files
create_config_files() {
    print_step "Creating configuration files..."
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        print_info "Creating .env template..."
        cat > .env << 'EOF'
# Groq API Configuration (optional - required for LLM features)
GROQ_API_KEY=

# Optional: Override config.json settings
# CAMERA_INDEX=0
# DETECTION_INTERVAL=3.0
EOF
        print_status "Created .env file (add your Groq API key if needed)"
    else
        print_info ".env file already exists"
    fi
    
    # Create directories
    mkdir -p detections logs
    print_status "Created detections/ and logs/ directories"
}

# Create or update startup script
create_startup_script() {
    print_step "Creating startup script..."
    
    # Make sure start_varg.sh is executable
    if [ -f "start_varg.sh" ]; then
        chmod +x start_varg.sh
        print_status "Updated start_varg.sh"
    else
        print_warning "start_varg.sh not found in repository"
    fi
    
    # Create monitor script if on Raspberry Pi
    if [ "$IS_RPI" = true ] && [ ! -f "monitor_varg.sh" ]; then
        print_info "Creating monitoring script..."
        cat > monitor_varg.sh << 'EOF'
#!/bin/bash
echo "📊 V.A.R.G System Monitor"
echo "========================"
echo ""
echo "🔍 Service Status:"
sudo systemctl status varg.service --no-pager -l || echo "Service not running"
echo ""
echo "📈 System Resources:"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory: $(free | grep Mem | awk '{printf("%.1f%%\n", $3/$2 * 100.0)}')"
if command -v vcgencmd &> /dev/null; then
    echo "Temp: $(vcgencmd measure_temp | cut -d'=' -f2)"
fi
echo ""
echo "📋 Recent Logs:"
sudo journalctl -u varg.service --no-pager -n 20 || echo "No service logs available"
EOF
        chmod +x monitor_varg.sh
        print_status "Created monitor_varg.sh"
    fi
}

# Setup systemd service (Raspberry Pi only)
setup_systemd_service() {
    if [ "$SKIP_SERVICE" = true ]; then
        print_warning "Skipping systemd service setup (--skip-service flag)"
        return
    fi
    
    print_step "Setting up systemd service..."
    
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
Environment=PATH=$VARG_DIR/varg_env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
EnvironmentFile=-$VARG_DIR/.env
ExecStart=$VARG_DIR/varg_env/bin/python3 $VARG_DIR/v.a.r.g.py
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
    
    # Reload systemd
    sudo systemctl daemon-reload
    sudo systemctl enable varg.service
    
    print_status "Systemd service created and enabled"
    print_info "Start with: sudo systemctl start varg.service"
}

# Print completion summary
print_summary() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ Installation Complete!${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    if [ "$IS_RPI" = true ]; then
        echo -e "${BLUE}📋 Next Steps for Raspberry Pi:${NC}"
        echo ""
        echo "1. 🔑 Add your Groq API key (optional, for LLM features):"
        echo "   nano .env"
        echo "   (Set GROQ_API_KEY=your_key_here)"
        echo ""
        echo "2. 🚀 Start V.A.R.G:"
        echo "   - Manual: ./start_varg.sh"
        echo "   - Service: sudo systemctl start varg.service"
        echo ""
        echo "3. 📊 Monitor the system:"
        echo "   - ./monitor_varg.sh"
        echo "   - sudo journalctl -u varg.service -f"
        echo ""
        echo "4. 🔄 Reboot recommended to apply all hardware changes:"
        echo "   sudo reboot"
        echo ""
        print_warning "A reboot is recommended to ensure all hardware interfaces are active"
    else
        echo -e "${BLUE}📋 Next Steps for Development:${NC}"
        echo ""
        echo "1. 🔑 Add your Groq API key (optional):"
        echo "   nano .env"
        echo ""
        echo "2. 🚀 Start V.A.R.G:"
        echo "   source varg_env/bin/activate"
        echo "   python3 v.a.r.g.py"
        echo ""
        echo "3. 📝 Configure settings in config.json as needed"
        echo ""
    fi
    
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    print_info "For detailed documentation, see README.md"
    echo ""
}

# Main installation flow
main() {
    print_banner
    
    # Show dry-run notice if enabled
    if [ "$DRY_RUN" = true ]; then
        print_warning "DRY RUN MODE - No changes will be made"
        echo ""
    fi
    
    # Store start time
    START_TIME=$(date +%s)
    
    # Detect platform
    detect_platform
    echo ""
    
    # Check prerequisites
    check_prerequisites
    echo ""
    
    # Platform-specific setup
    if [ "$IS_RPI" = true ]; then
        print_info "Performing Raspberry Pi installation..."
        echo ""
        
        # Install system dependencies
        install_system_deps_pi
        echo ""
        
        # Enable hardware interfaces
        enable_hardware_interfaces
        echo ""
    else
        print_info "Performing development environment installation..."
        echo ""
    fi
    
    # Create virtual environment
    create_venv
    echo ""
    
    # Setup pip
    setup_pip
    echo ""
    
    # Install Python dependencies
    install_python_deps
    echo ""
    
    # Setup Waveshare OLED
    setup_waveshare_oled
    echo ""
    
    # Setup models
    setup_models
    echo ""
    
    # Create configuration files
    create_config_files
    echo ""
    
    # Create startup scripts
    create_startup_script
    echo ""
    
    # Setup systemd service (Raspberry Pi only)
    if [ "$IS_RPI" = true ]; then
        setup_systemd_service
        echo ""
    fi
    
    # Calculate installation time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    MINUTES=$((DURATION / 60))
    SECONDS=$((DURATION % 60))
    
    print_info "Installation took ${MINUTES}m ${SECONDS}s"
    echo ""
    
    # Run verification
    if [ "$DRY_RUN" = false ] && [ -f "verify_installation.sh" ]; then
        print_step "Running installation verification..."
        echo ""
        chmod +x verify_installation.sh
        ./verify_installation.sh
        echo ""
    fi
    
    # Print summary
    print_summary
}

# Error handler
trap 'print_error "Installation failed at line $LINENO. Check the output above for details."; exit 1' ERR

# Run main installation
main "$@"

