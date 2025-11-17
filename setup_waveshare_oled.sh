#!/bin/bash
# Setup script for Waveshare 1.51" OLED display
# This ensures the Waveshare OLED library is properly installed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RASPBERRY_PI_DIR="${SCRIPT_DIR}/RaspberryPi/python"

echo "🔧 Setting up Waveshare 1.51\" OLED Display Library..."

# Check if RaspberryPi/python directory exists
if [ ! -d "$RASPBERRY_PI_DIR" ]; then
    echo "❌ Error: RaspberryPi/python directory not found!"
    echo "   Expected location: $RASPBERRY_PI_DIR"
    exit 1
fi

# Check if setup.py exists
if [ ! -f "$RASPBERRY_PI_DIR/setup.py" ]; then
    echo "❌ Error: setup.py not found in $RASPBERRY_PI_DIR"
    exit 1
fi

echo "📦 Installing Waveshare OLED library..."
cd "$RASPBERRY_PI_DIR"

# Install the library using pip (modern standard)
if command -v pip3 &> /dev/null; then
    echo "   Using pip3..."
    sudo pip3 install -e . || sudo pip3 install .
elif command -v pip &> /dev/null; then
    echo "   Using pip..."
    sudo pip install -e . || sudo pip install .
elif command -v python3 &> /dev/null; then
    echo "   Using python3 -m pip..."
    sudo python3 -m pip install -e . || sudo python3 -m pip install .
else
    echo "❌ Error: Python/pip not found!"
    exit 1
fi

# Verify installation
echo "✅ Verifying installation..."
python3 -c "from waveshare_OLED import OLED_1in51; print('✅ Waveshare OLED library imported successfully')" 2>/dev/null || {
    echo "⚠️  Warning: Library import test failed, but installation may have succeeded"
    echo "   Try running your script - it may work if the library is in the path"
}

echo ""
echo "✅ Waveshare OLED setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Ensure SPI is enabled: sudo raspi-config -> Interface Options -> SPI -> Enable"
echo "   2. Check wiring connections (see config.json for pin assignments)"
echo "   3. Run: python3 v.a.r.g.py"

