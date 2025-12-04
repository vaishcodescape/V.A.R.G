#!/bin/bash

################################################################################
# V.A.R.G Troubleshooting Script
# Diagnoses common issues and suggests fixes
################################################################################

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  V.A.R.G Troubleshooting Tool${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""

# Detect if running on Raspberry Pi
IS_RPI=false
if [ -f /proc/cpuinfo ] && grep -q "BCM\|ARM" /proc/cpuinfo 2>/dev/null; then
    IS_RPI=true
fi

# Function to print section header
section() {
    echo ""
    echo -e "${BLUE}═══ $1 ═══${NC}"
    echo ""
}

# Function to suggest fix
suggest_fix() {
    echo -e "${YELLOW}💡 Suggested fix:${NC}"
    echo "   $1"
    echo ""
}

# 1. System Information
section "System Information"
echo "Platform: $(uname -s) $(uname -m)"
echo "Kernel: $(uname -r)"
if [ "$IS_RPI" = true ]; then
    echo "Device: $(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo 'Raspberry Pi (unknown model)')"
fi
echo "Python: $(python3 --version 2>&1)"
echo "Disk space: $(df -h . | tail -1 | awk '{print $4}') available"
echo "Memory: $(free -h | grep Mem | awk '{print $4}') available"

if [ "$IS_RPI" = true ] && command -v vcgencmd &>/dev/null; then
    echo "CPU Temp: $(vcgencmd measure_temp | cut -d'=' -f2)"
fi

# 2. Check if installation completed
section "Installation Status"

if [ ! -d "varg_env" ]; then
    echo -e "${RED}✗ Virtual environment not found${NC}"
    suggest_fix "Run: ./install.sh"
elif [ ! -f "varg_env/bin/activate" ]; then
    echo -e "${RED}✗ Virtual environment is corrupted${NC}"
    suggest_fix "Run: rm -rf varg_env && ./install.sh"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
    
    # Check if key packages are installed
    source varg_env/bin/activate
    
    MISSING_PACKAGES=()
    for pkg in numpy PIL psutil requests dotenv; do
        if ! python3 -c "import $pkg" 2>/dev/null; then
            MISSING_PACKAGES+=("$pkg")
        fi
    done
    
    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        echo -e "${RED}✗ Missing packages: ${MISSING_PACKAGES[*]}${NC}"
        suggest_fix "Run: source varg_env/bin/activate && pip install -r requirements.txt"
    else
        echo -e "${GREEN}✓ Core Python packages installed${NC}"
    fi
fi

if [ ! -f "config.json" ]; then
    echo -e "${RED}✗ config.json not found${NC}"
    suggest_fix "Run: ./install.sh or copy config.json from repository"
else
    echo -e "${GREEN}✓ config.json exists${NC}"
fi

# 3. Camera Issues (Raspberry Pi)
if [ "$IS_RPI" = true ]; then
    section "Camera Status"
    
    # Check for camera device
    if [ -e /dev/video0 ]; then
        echo -e "${GREEN}✓ Camera device found (/dev/video0)${NC}"
    else
        echo -e "${RED}✗ Camera device not found${NC}"
        suggest_fix "Enable camera: sudo raspi-config nonint do_camera 0 && sudo reboot"
    fi
    
    # Check for other processes using camera
    if command -v lsof &>/dev/null && [ -e /dev/video0 ]; then
        CAMERA_USERS=$(sudo lsof /dev/video0 2>/dev/null | tail -n +2)
        if [ -n "$CAMERA_USERS" ]; then
            echo -e "${YELLOW}⚠ Camera is being used by:${NC}"
            echo "$CAMERA_USERS"
            suggest_fix "Kill camera processes: sudo pkill -9 libcamera; sudo pkill -9 python3"
        else
            echo -e "${GREEN}✓ Camera is not in use${NC}"
        fi
    fi
    
    # Test camera with libcamera if available
    if command -v libcamera-hello &>/dev/null; then
        echo "Testing camera with libcamera..."
        if timeout 3 libcamera-hello --timeout 1000 &>/dev/null; then
            echo -e "${GREEN}✓ Camera test successful${NC}"
        else
            echo -e "${RED}✗ Camera test failed${NC}"
            suggest_fix "Check camera connection and ribbon cable"
        fi
    fi
fi

# 4. GPIO/SPI/I2C (Raspberry Pi)
if [ "$IS_RPI" = true ]; then
    section "Hardware Interfaces"
    
    # Check SPI
    if [ -e /dev/spidev0.0 ]; then
        echo -e "${GREEN}✓ SPI enabled${NC}"
    else
        echo -e "${RED}✗ SPI not enabled${NC}"
        suggest_fix "Enable SPI: sudo raspi-config nonint do_spi 0 && sudo reboot"
    fi
    
    # Check I2C
    if [ -e /dev/i2c-1 ]; then
        echo -e "${GREEN}✓ I2C enabled${NC}"
    else
        echo -e "${RED}✗ I2C not enabled${NC}"
        suggest_fix "Enable I2C: sudo raspi-config nonint do_i2c 0 && sudo reboot"
    fi
    
    # Check GPIO
    if command -v gpio &>/dev/null; then
        echo -e "${GREEN}✓ GPIO tools available${NC}"
    else
        echo -e "${YELLOW}⚠ GPIO tools not installed${NC}"
        suggest_fix "Install: sudo apt-get install wiringpi"
    fi
fi

# 5. Service Status (Raspberry Pi)
if [ "$IS_RPI" = true ]; then
    section "Service Status"
    
    if systemctl list-unit-files | grep -q "varg.service"; then
        echo -e "${GREEN}✓ Service installed${NC}"
        
        # Check if service is running
        if systemctl is-active varg.service &>/dev/null; then
            echo -e "${GREEN}✓ Service is running${NC}"
            
            # Show recent errors from journal
            ERRORS=$(sudo journalctl -u varg.service --since "5 minutes ago" --no-pager | grep -i "error\|exception\|failed" | tail -5)
            if [ -n "$ERRORS" ]; then
                echo -e "${YELLOW}⚠ Recent errors in service logs:${NC}"
                echo "$ERRORS"
                suggest_fix "View full logs: sudo journalctl -u varg.service -f"
            fi
        else
            echo -e "${RED}✗ Service is not running${NC}"
            
            # Try to get failure reason
            FAILURE=$(sudo systemctl status varg.service --no-pager -l 2>&1 | grep -i "failed\|error" | head -3)
            if [ -n "$FAILURE" ]; then
                echo -e "${YELLOW}Failure reason:${NC}"
                echo "$FAILURE"
            fi
            
            suggest_fix "Start service: sudo systemctl start varg.service
   View logs: sudo journalctl -u varg.service -n 50"
        fi
        
        # Check if enabled
        if systemctl is-enabled varg.service &>/dev/null; then
            echo -e "${GREEN}✓ Service enabled (auto-start on boot)${NC}"
        else
            echo -e "${YELLOW}⚠ Service not enabled${NC}"
            suggest_fix "Enable: sudo systemctl enable varg.service"
        fi
    else
        echo -e "${RED}✗ Service not installed${NC}"
        suggest_fix "Run: ./install.sh"
    fi
fi

# 6. Common File Issues
section "File Integrity"

REQUIRED_FILES=("v.a.r.g.py" "config.json" "requirements.txt" "start_varg.sh")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo -e "${RED}✗ Missing files: ${MISSING_FILES[*]}${NC}"
    suggest_fix "Re-clone repository or restore missing files"
else
    echo -e "${GREEN}✓ All required files present${NC}"
fi

# Check permissions
if [ ! -x "start_varg.sh" ]; then
    echo -e "${YELLOW}⚠ start_varg.sh is not executable${NC}"
    suggest_fix "Run: chmod +x start_varg.sh"
fi

if [ ! -x "v.a.r.g.py" ]; then
    echo -e "${YELLOW}⚠ v.a.r.g.py is not executable${NC}"
    suggest_fix "Run: chmod +x v.a.r.g.py"
fi

# 7. Network/API Issues
section "Network & API"

# Check internet connectivity
if ping -c 1 8.8.8.8 &>/dev/null; then
    echo -e "${GREEN}✓ Internet connectivity${NC}"
else
    echo -e "${RED}✗ No internet connection${NC}"
    suggest_fix "Check WiFi connection: sudo nmcli device wifi list
   Connect to WiFi: sudo nmcli device wifi connect SSID password PASSWORD"
fi

# Check .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠ .env file not found${NC}"
    suggest_fix "Create .env file: echo 'GROQ_API_KEY=' > .env"
elif ! grep -q "^GROQ_API_KEY=.\+$" .env; then
    echo -e "${YELLOW}⚠ GROQ_API_KEY not set in .env${NC}"
    echo "   LLM features will be disabled"
else
    echo -e "${GREEN}✓ GROQ_API_KEY is set${NC}"
fi

# 8. Performance Check
section "Performance"

# Check CPU usage
if command -v top &>/dev/null; then
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    if (( $(echo "$CPU_USAGE > 80" | bc -l 2>/dev/null || echo 0) )); then
        echo -e "${YELLOW}⚠ High CPU usage: ${CPU_USAGE}%${NC}"
        suggest_fix "Check running processes: top
   Optimize config.json for lower CPU usage"
    else
        echo -e "${GREEN}✓ CPU usage: ${CPU_USAGE}%${NC}"
    fi
fi

# Check memory usage
if command -v free &>/dev/null; then
    MEM_USAGE=$(free | grep Mem | awk '{printf("%.0f\n", $3/$2 * 100.0)}')
    if [ "$MEM_USAGE" -gt 80 ]; then
        echo -e "${YELLOW}⚠ High memory usage: ${MEM_USAGE}%${NC}"
        suggest_fix "Restart service: sudo systemctl restart varg.service
   Consider increasing swap size"
    else
        echo -e "${GREEN}✓ Memory usage: ${MEM_USAGE}%${NC}"
    fi
fi

# Check temperature (Raspberry Pi)
if [ "$IS_RPI" = true ] && command -v vcgencmd &>/dev/null; then
    TEMP=$(vcgencmd measure_temp | cut -d'=' -f2 | cut -d"'" -f1)
    if (( $(echo "$TEMP > 70" | bc -l) )); then
        echo -e "${RED}✗ High temperature: ${TEMP}°C${NC}"
        suggest_fix "Improve cooling: add heatsink or fan
   Reduce workload in config.json"
    elif (( $(echo "$TEMP > 60" | bc -l) )); then
        echo -e "${YELLOW}⚠ Elevated temperature: ${TEMP}°C${NC}"
    else
        echo -e "${GREEN}✓ Temperature: ${TEMP}°C${NC}"
    fi
fi

# 9. Quick Fixes
section "Quick Fixes"

echo "Try these common solutions:"
echo ""
echo "1. Restart the service:"
echo "   sudo systemctl restart varg.service"
echo ""
echo "2. View live logs:"
echo "   sudo journalctl -u varg.service -f"
echo ""
echo "3. Reinstall dependencies:"
echo "   source varg_env/bin/activate"
echo "   pip install -r requirements.txt --force-reinstall"
echo ""
echo "4. Full reinstall:"
echo "   ./install.sh"
echo ""
echo "5. Reboot the system:"
echo "   sudo reboot"
echo ""

# Summary
section "Summary"

echo "For more help, check:"
echo "  - QUICKSTART.md for common commands"
echo "  - README.md for detailed documentation"
echo "  - GitHub issues for known problems"
echo ""
echo "Still having issues? Run these for more details:"
echo "  - ./verify_installation.sh (check installation)"
echo "  - ./monitor_varg.sh (system monitor)"
echo "  - sudo journalctl -u varg.service -n 100 (recent logs)"
echo ""

