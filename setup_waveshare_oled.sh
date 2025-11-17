#!/bin/bash
# Setup script for Waveshare 1.51" OLED display
# This ensures the Waveshare OLED library is properly installed

# Don't exit on errors - we want to continue even if extraction/installation fails
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RASPBERRY_PI_DIR="${SCRIPT_DIR}/RaspberryPi/python"
ARCHIVE_FILE="${SCRIPT_DIR}/OLED_Module_Code.7z"
EXTRACTED_DIR="${SCRIPT_DIR}/OLED_Module_Code"

echo "🔧 Setting up Waveshare 1.51\" OLED Display Library..."

# Check if RaspberryPi/python directory exists, create if needed
if [ ! -d "$RASPBERRY_PI_DIR" ]; then
    echo "📁 Creating RaspberryPi/python directory..."
    mkdir -p "$RASPBERRY_PI_DIR"
fi

# Check if lib directory exists (required by setup.py)
LIB_DIR="${RASPBERRY_PI_DIR}/lib"
WAVESHARE_LIB="${LIB_DIR}/waveshare_OLED"
SETUP_PY="${RASPBERRY_PI_DIR}/setup.py"

if [ ! -d "$LIB_DIR" ] || [ ! -d "$WAVESHARE_LIB" ]; then
    echo "📦 Waveshare OLED library not found. Checking for archive..."
    
    # Check if archive exists and extract if needed
    if [ -f "$ARCHIVE_FILE" ]; then
        echo "   Found OLED_Module_Code.7z archive"
        
        # Check if extraction tools are available
        EXTRACTED=false
        if command -v 7z &> /dev/null || command -v 7za &> /dev/null; then
            echo "   Extracting archive with 7z..."
            if command -v 7z &> /dev/null; then
                7z x "$ARCHIVE_FILE" -o"$SCRIPT_DIR" -y > /dev/null 2>&1 && EXTRACTED=true
            else
                7za x "$ARCHIVE_FILE" -o"$SCRIPT_DIR" -y > /dev/null 2>&1 && EXTRACTED=true
            fi
        elif command -v unar &> /dev/null; then
            # macOS alternative: unar (install via: brew install unar)
            echo "   Extracting archive with unar..."
            unar "$ARCHIVE_FILE" -o "$SCRIPT_DIR" > /dev/null 2>&1 && EXTRACTED=true
        elif command -v python3 &> /dev/null; then
            # Try using Python's py7zr library
            echo "   Attempting extraction with Python..."
            python3 -c "
import sys
try:
    import py7zr
    with py7zr.SevenZipFile('$ARCHIVE_FILE', mode='r') as archive:
        archive.extractall(path='$SCRIPT_DIR')
    sys.exit(0)
except ImportError:
    print('   py7zr not installed. Install with: pip3 install py7zr')
    sys.exit(1)
except Exception as e:
    print(f'   Extraction failed: {e}')
    sys.exit(1)
" && EXTRACTED=true
        fi
        
        if [ "$EXTRACTED" = true ]; then
            
            # Check if extraction was successful
            if [ -d "$EXTRACTED_DIR/RaspberryPi/python" ]; then
                echo "   ✅ Archive extracted successfully"
                SOURCE_DIR="${EXTRACTED_DIR}/RaspberryPi/python"
                
                # Copy lib directory if it exists
                if [ -d "$SOURCE_DIR/lib" ]; then
                    echo "   Copying lib directory..."
                    cp -r "$SOURCE_DIR/lib" "$RASPBERRY_PI_DIR/" || true
                fi
                
                # Copy setup.py if it exists
                if [ -f "$SOURCE_DIR/setup.py" ]; then
                    echo "   Copying setup.py..."
                    cp "$SOURCE_DIR/setup.py" "$RASPBERRY_PI_DIR/" || true
                fi
                
                # Copy pic directory if it exists
                if [ -d "$SOURCE_DIR/pic" ]; then
                    echo "   Copying pic directory..."
                    cp -r "$SOURCE_DIR/pic" "$RASPBERRY_PI_DIR/" || true
                fi
            fi
        else
            echo "   ⚠️  No extraction tool found!"
            echo ""
            echo "   To extract the archive, install one of:"
            echo "   - macOS: brew install p7zip  (or brew install unar)"
            echo "   - Linux: sudo apt-get install p7zip-full"
            echo "   - Python: pip3 install py7zr"
            echo ""
            echo "   Or manually extract OLED_Module_Code.7z and copy lib/ to RaspberryPi/python/"
        fi
    fi
    
    # Check again if lib directory exists now
    if [ ! -d "$LIB_DIR" ] || [ ! -d "$WAVESHARE_LIB" ]; then
        echo "⚠️  Warning: Waveshare OLED library still not found!"
        echo "   Expected location: $WAVESHARE_LIB"
        echo ""
        echo "   Please ensure the library files are in:"
        echo "   RaspberryPi/python/lib/waveshare_OLED/"
        echo ""
        echo "   The code will use path-based import, so installation is optional."
        echo "   Just ensure the lib directory exists with the library files."
        exit 0  # Don't fail, just inform
    fi
fi

echo "✅ Found Waveshare OLED library at $WAVESHARE_LIB"

# Try to install if lib exists and setup.py is present (optional - path-based import will work too)
if [ -f "$SETUP_PY" ]; then
    echo "📦 setup.py found - attempting installation (optional)..."
    cd "$RASPBERRY_PI_DIR"
    
    # Check if virtual environment exists
    VENV_PATH="${SCRIPT_DIR}/varg_env"
    USE_VENV=false
    
    if [ -d "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
        echo "   Found virtual environment at $VENV_PATH"
        USE_VENV=true
        source "$VENV_PATH/bin/activate"
        echo "   Activated virtual environment"
    fi
    
    # Try to install the library (optional - will work with path-based import too)
    if [ "$USE_VENV" = true ]; then
        echo "   Attempting to install in virtual environment..."
        pip install -e . 2>/dev/null || pip install . 2>/dev/null || echo "   (Installation failed - path-based import will work)"
    elif command -v pip3 &> /dev/null; then
        echo "   Attempting to install for current user..."
        pip3 install --user -e . 2>/dev/null || pip3 install --user . 2>/dev/null || echo "   (Installation failed - path-based import will work)"
    else
        echo "   (Skipping installation - path-based import will work)"
    fi
else
    echo "ℹ️  setup.py not found - skipping pip installation"
    echo "   Path-based import will be used (no installation needed)"
fi

# Verify library can be imported (using path-based import)
echo "✅ Verifying library import..."
if [ "$USE_VENV" = true ]; then
    python -c "import sys; sys.path.insert(0, '$LIB_DIR'); from waveshare_OLED import OLED_1in51; print('✅ Waveshare OLED library imported successfully')" 2>/dev/null || {
        echo "⚠️  Warning: Library import test failed"
        echo "   Make sure the lib/waveshare_OLED directory exists with the library files"
    }
else
    python3 -c "import sys; sys.path.insert(0, '$LIB_DIR'); from waveshare_OLED import OLED_1in51; print('✅ Waveshare OLED library imported successfully')" 2>/dev/null || {
        echo "⚠️  Warning: Library import test failed"
        echo "   Make sure the lib/waveshare_OLED directory exists with the library files"
    }
fi

# Deactivate venv if we activated it
if [ "$USE_VENV" = true ]; then
    deactivate 2>/dev/null || true
fi

echo ""
echo "✅ Waveshare OLED setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Ensure SPI is enabled: sudo raspi-config -> Interface Options -> SPI -> Enable"
echo "   2. Check wiring connections (see config.json for pin assignments)"
echo "   3. Run: python3 v.a.r.g.py"

