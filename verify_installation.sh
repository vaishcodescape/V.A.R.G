#!/bin/bash

################################################################################
# V.A.R.G Installation Verification Script
# Checks that all components are properly installed
################################################################################

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  V.A.R.G Installation Verification${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

ERRORS=0
WARNINGS=0

# Check function
check_item() {
    local name="$1"
    local check_cmd="$2"
    local error_level="${3:-error}"  # error or warning
    
    printf "%-50s " "Checking $name..."
    
    if eval "$check_cmd" &>/dev/null; then
        echo -e "${GREEN}✓${NC}"
        return 0
    else
        if [ "$error_level" = "error" ]; then
            echo -e "${RED}✗${NC}"
            ((ERRORS++))
        else
            echo -e "${YELLOW}⚠${NC}"
            ((WARNINGS++))
        fi
        return 1
    fi
}

# 1. Check Python environment
echo -e "${BLUE}[1] Python Environment${NC}"
check_item "Python 3" "command -v python3"
check_item "pip" "command -v pip3 || python3 -m pip --version"
check_item "Virtual environment" "[ -d varg_env ]"
check_item "Virtual environment activated" "[ -n \"\$VIRTUAL_ENV\" ]" "warning"
echo ""

# 2. Check Python packages
echo -e "${BLUE}[2] Python Packages${NC}"

# Activate venv if not already activated
if [ -z "$VIRTUAL_ENV" ] && [ -d "varg_env" ]; then
    echo "Activating virtual environment..."
    source varg_env/bin/activate
fi

check_item "numpy" "python3 -c 'import numpy'"
check_item "PIL (Pillow)" "python3 -c 'import PIL'"
check_item "psutil" "python3 -c 'import psutil'"
check_item "requests" "python3 -c 'import requests'"
check_item "dotenv" "python3 -c 'import dotenv'"
check_item "cv2 (OpenCV)" "python3 -c 'import cv2'" "warning"
echo ""

# 3. Check configuration files
echo -e "${BLUE}[3] Configuration Files${NC}"
check_item "config.json" "[ -f config.json ]"
check_item ".env file" "[ -f .env ]" "warning"

if [ -f ".env" ]; then
    if grep -q "^GROQ_API_KEY=.\+$" .env; then
        echo -e "   ${GREEN}✓${NC} GROQ_API_KEY is set"
    else
        echo -e "   ${YELLOW}⚠${NC} GROQ_API_KEY not set (LLM features disabled)"
        ((WARNINGS++))
    fi
fi
echo ""

# 4. Check directories
echo -e "${BLUE}[4] Required Directories${NC}"
check_item "models/" "[ -d models ]"
check_item "detections/" "[ -d detections ]"
check_item "logs/" "[ -d logs ]"
echo ""

# 5. Check models
echo -e "${BLUE}[5] TensorFlow Lite Models${NC}"
if [ -d "models" ]; then
    MODEL_COUNT=$(find models -name "*.tflite" 2>/dev/null | wc -l)
    if [ "$MODEL_COUNT" -gt 0 ]; then
        echo -e "   ${GREEN}✓${NC} Found $MODEL_COUNT TFLite model(s)"
        find models -name "*.tflite" -exec basename {} \; | sed 's/^/     - /'
    else
        echo -e "   ${YELLOW}⚠${NC} No TFLite models found"
        echo "     Run: python3 setup_models.py"
        ((WARNINGS++))
    fi
else
    echo -e "   ${RED}✗${NC} models/ directory not found"
    ((ERRORS++))
fi
echo ""

# 6. Check Waveshare OLED library
echo -e "${BLUE}[6] Waveshare OLED Library${NC}"
if [ -d "Raspberry/python/lib" ] || [ -d "RaspberryPi/python/lib" ]; then
    echo -e "   ${GREEN}✓${NC} Waveshare library directory found"
    
    # Try to import
    if python3 -c "import sys; sys.path.insert(0, 'Raspberry/python/lib'); from waveshare_OLED import OLED_1in51" 2>/dev/null || \
       python3 -c "import sys; sys.path.insert(0, 'RaspberryPi/python/lib'); from waveshare_OLED import OLED_1in51" 2>/dev/null; then
        echo -e "   ${GREEN}✓${NC} Waveshare OLED library imports successfully"
    else
        echo -e "   ${YELLOW}⚠${NC} Waveshare OLED library found but cannot import"
        ((WARNINGS++))
    fi
else
    echo -e "   ${YELLOW}⚠${NC} Waveshare library not found (OLED display will not work)"
    ((WARNINGS++))
fi
echo ""

# 7. Check scripts
echo -e "${BLUE}[7] Scripts${NC}"
check_item "v.a.r.g.py" "[ -f v.a.r.g.py ]"
check_item "start_varg.sh (executable)" "[ -x start_varg.sh ]" "warning"
check_item "install.sh (executable)" "[ -x install.sh ]" "warning"
echo ""

# 8. Raspberry Pi specific checks
if [ -f /proc/cpuinfo ] && grep -q "BCM\|ARM" /proc/cpuinfo 2>/dev/null; then
    echo -e "${BLUE}[8] Raspberry Pi Configuration${NC}"
    
    # Check camera
    if [ -e /dev/video0 ]; then
        echo -e "   ${GREEN}✓${NC} Camera device found (/dev/video0)"
    else
        echo -e "   ${YELLOW}⚠${NC} Camera device not found"
        echo "     Enable with: sudo raspi-config nonint do_camera 0"
        ((WARNINGS++))
    fi
    
    # Check SPI
    if [ -e /dev/spidev0.0 ]; then
        echo -e "   ${GREEN}✓${NC} SPI enabled"
    else
        echo -e "   ${YELLOW}⚠${NC} SPI not enabled"
        echo "     Enable with: sudo raspi-config nonint do_spi 0"
        ((WARNINGS++))
    fi
    
    # Check I2C
    if [ -e /dev/i2c-1 ]; then
        echo -e "   ${GREEN}✓${NC} I2C enabled"
    else
        echo -e "   ${YELLOW}⚠${NC} I2C not enabled"
        echo "     Enable with: sudo raspi-config nonint do_i2c 0"
        ((WARNINGS++))
    fi
    
    # Check systemd service
    if systemctl list-unit-files | grep -q "varg.service"; then
        echo -e "   ${GREEN}✓${NC} systemd service installed"
        
        if systemctl is-enabled varg.service &>/dev/null; then
            echo -e "   ${GREEN}✓${NC} Service enabled (will start on boot)"
        else
            echo -e "   ${YELLOW}⚠${NC} Service not enabled"
            echo "     Enable with: sudo systemctl enable varg.service"
            ((WARNINGS++))
        fi
        
        if systemctl is-active varg.service &>/dev/null; then
            echo -e "   ${GREEN}✓${NC} Service is running"
        else
            echo -e "   ${BLUE}ℹ${NC} Service not currently running"
            echo "     Start with: sudo systemctl start varg.service"
        fi
    else
        echo -e "   ${YELLOW}⚠${NC} systemd service not installed"
        ((WARNINGS++))
    fi
    echo ""
fi

# Summary
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Summary${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "V.A.R.G is ready to run."
    echo ""
    echo "To start:"
    echo "  ./start_varg.sh"
    echo ""
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Installation complete with $WARNINGS warning(s)${NC}"
    echo ""
    echo "V.A.R.G should work, but some features may be limited."
    echo "Review warnings above for optional improvements."
    echo ""
    exit 0
else
    echo -e "${RED}✗ Installation incomplete: $ERRORS error(s), $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before running V.A.R.G."
    echo ""
    echo "If you just installed, try:"
    echo "  ./install.sh"
    echo ""
    exit 1
fi

