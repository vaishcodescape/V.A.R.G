# V.A.R.G Installation Guide

Complete installation guide for V.A.R.G (Virtual Augmented Reality Glasses).

## Table of Contents
- [Quick Installation](#quick-installation)
- [Installation Options](#installation-options)
- [What Gets Installed](#what-gets-installed)
- [Post-Installation](#post-installation)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Quick Installation

### One-Command Install (Recommended)

**Raspberry Pi Zero W:**
```bash
git clone <YOUR_REPO_URL> V.A.R.G
cd V.A.R.G
./install.sh
```

**Development Machine (Mac/Linux):**
```bash
git clone <YOUR_REPO_URL> V.A.R.G
cd V.A.R.G
./install.sh
```

The installer automatically:
- ✅ Detects your platform
- ✅ Installs system dependencies
- ✅ Creates Python virtual environment
- ✅ Installs Python packages
- ✅ Downloads TensorFlow Lite models
- ✅ Sets up Waveshare OLED library
- ✅ Configures hardware interfaces (Raspberry Pi)
- ✅ Creates systemd service (Raspberry Pi)
- ✅ Verifies installation

---

## Installation Options

### Option 1: Automated Installer (Recommended)

```bash
./install.sh [options]
```

**Options:**
- `--dry-run` - Show what would be installed without making changes
- `--skip-models` - Skip TensorFlow Lite model downloads
- `--skip-service` - Skip systemd service setup
- `--help` - Show help message

**Examples:**
```bash
# Standard installation
./install.sh

# Preview what will be installed
./install.sh --dry-run

# Install without models (download later)
./install.sh --skip-models

# Install for manual start only (no service)
./install.sh --skip-service
```

### Option 2: Raspberry Pi Deploy Script

Specifically optimized for Raspberry Pi deployment:
```bash
chmod +x deploy_pi.sh
./deploy_pi.sh
```

This script includes additional Pi-specific optimizations:
- GPU memory allocation
- ARM frequency tuning
- Bluetooth/power management
- Firmware configuration

### Option 3: Manual Installation

For complete control over the installation process:

**Step 1: System Dependencies (Raspberry Pi)**
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-venv python3-dev \
    python3-numpy python3-pil python3-psutil \
    python3-rpi.gpio python3-spidev \
    libjpeg-dev zlib1g-dev i2c-tools
```

**Step 2: Create Virtual Environment**
```bash
python3 -m venv --system-site-packages varg_env
source varg_env/bin/activate
```

**Step 3: Install Python Packages**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 4: Setup Models**
```bash
python3 setup_models.py
```

**Step 5: Setup Waveshare OLED**
```bash
./setup_waveshare_oled.sh
```

**Step 6: Configure Hardware (Raspberry Pi)**
```bash
sudo raspi-config nonint do_camera 0
sudo raspi-config nonint do_spi 0
sudo raspi-config nonint do_i2c 0
```

**Step 7: Create Configuration**
```bash
# Create .env file
echo "GROQ_API_KEY=" > .env

# config.json should exist in repository
# Edit as needed: nano config.json
```

---

## What Gets Installed

### System Packages (Raspberry Pi)
- **Python packages:** python3-pip, python3-venv, python3-dev
- **Scientific:** python3-numpy, python3-pil, python3-psutil
- **Hardware:** python3-rpi.gpio, python3-spidev
- **Libraries:** libjpeg-dev, zlib1g-dev, libatlas-base-dev
- **Tools:** i2c-tools, fonts-dejavu-core
- **Optional:** python3-picamera2, python3-opencv

### Python Packages (Virtual Environment)
```
numpy>=1.19.0,<1.26.0
Pillow>=8.0.0,<11.0.0
psutil>=5.8.0,<6.0.0
python-dotenv>=0.19.0,<2.0.0
requests>=2.25.0,<3.0.0
spidev
RPi.GPIO
```

### TensorFlow Lite Models
- **EfficientNet Lite** (recommended for Pi Zero W)
- **MobileNet Food V2** (high accuracy)
- **Food-101 MobileNet** (comprehensive dataset)

Models are downloaded to `models/` directory.

### Hardware Interfaces (Raspberry Pi)
- **Camera:** /dev/video0
- **SPI:** /dev/spidev0.0
- **I2C:** /dev/i2c-1

### File Structure Created
```
V.A.R.G/
├── varg_env/              # Virtual environment
├── models/                # TensorFlow Lite models
├── detections/            # Saved detections
├── logs/                  # Log files
├── .env                   # API keys
├── config.json            # Configuration
└── Raspberry/             # Waveshare OLED library
```

### Systemd Service (Raspberry Pi)
- **Service file:** /etc/systemd/system/varg.service
- **Auto-start:** Enabled on boot
- **User:** Current user
- **Resource limits:** 400M memory, 80% CPU

---

## Post-Installation

### 1. Configure API Key (Optional)

For LLM-powered food analysis:
```bash
nano .env
# Add: GROQ_API_KEY=your_key_here
```

Get your key from: https://console.groq.com/

### 2. Test Installation

```bash
./verify_installation.sh
```

This checks:
- Python environment
- Required packages
- Configuration files
- Models
- Hardware interfaces (Raspberry Pi)
- Service status (Raspberry Pi)

### 3. Configure Settings

Edit `config.json` to customize:
```bash
nano config.json
```

Key settings:
- `camera_width`, `camera_height` - Resolution (lower = faster)
- `detection_interval` - Seconds between detections
- `detection_confidence` - Detection threshold (0-1)
- `performance.max_fps` - Maximum frame rate
- `oled_display` - Display configuration

### 4. Start V.A.R.G

**Manual start:**
```bash
./start_varg.sh
```

**Service (Raspberry Pi):**
```bash
sudo systemctl start varg.service
```

**Check status:**
```bash
sudo systemctl status varg.service
```

### 5. Monitor System

```bash
./monitor_varg.sh
```

Or view logs:
```bash
sudo journalctl -u varg.service -f
```

---

## Verification

### Quick Check
```bash
./verify_installation.sh
```

### Manual Verification

**1. Check Python environment:**
```bash
source varg_env/bin/activate
python3 -c "import numpy, PIL, psutil, requests; print('✓ All packages OK')"
```

**2. Check models:**
```bash
ls -lh models/*.tflite
```

**3. Check hardware (Raspberry Pi):**
```bash
# Camera
ls -l /dev/video0

# SPI
ls -l /dev/spidev0.0

# I2C
ls -l /dev/i2c-1

# Test camera
libcamera-hello --timeout 2000
```

**4. Check service (Raspberry Pi):**
```bash
systemctl is-enabled varg.service
systemctl is-active varg.service
```

---

## Troubleshooting

### Automated Diagnosis
```bash
./troubleshoot.sh
```

This will check:
- System information
- Installation status
- Camera status
- Hardware interfaces
- Service status
- File integrity
- Network connectivity
- Performance metrics

### Common Issues

#### Installation Failed

**Problem:** Installation script exits with errors

**Solution:**
```bash
# Check prerequisites
python3 --version  # Should be 3.7+
pip3 --version     # Should be available

# Clean and retry
rm -rf varg_env
./install.sh
```

#### Camera Not Working (Raspberry Pi)

**Problem:** Camera device not found

**Solution:**
```bash
# Enable camera interface
sudo raspi-config nonint do_camera 0

# Reboot
sudo reboot

# Test camera
libcamera-hello --timeout 2000
```

#### Service Won't Start (Raspberry Pi)

**Problem:** Service fails to start

**Solution:**
```bash
# Check logs
sudo journalctl -u varg.service -n 50

# Check .env file
cat .env

# Verify installation
./verify_installation.sh

# Restart service
sudo systemctl restart varg.service
```

#### Python Package Errors

**Problem:** ImportError or ModuleNotFoundError

**Solution:**
```bash
# Activate environment
source varg_env/bin/activate

# Reinstall packages
pip install -r requirements.txt --force-reinstall

# Verify imports
python3 -c "import numpy, PIL, psutil, requests"
```

#### OLED Display Not Working

**Problem:** OLED display shows nothing

**Solution:**
```bash
# Enable SPI
sudo raspi-config nonint do_spi 0

# Check wiring
# Verify pin configuration in config.json

# Reboot
sudo reboot
```

#### Models Not Downloading

**Problem:** setup_models.py fails

**Solution:**
```bash
# Check internet connection
ping -c 4 google.com

# Manual model setup
source varg_env/bin/activate
python3 setup_models.py --recommended

# Or skip models
./install.sh --skip-models
```

### Get More Help

1. **Run diagnostics:**
   ```bash
   ./troubleshoot.sh
   ./verify_installation.sh
   ```

2. **Check logs:**
   ```bash
   sudo journalctl -u varg.service -f
   ```

3. **Documentation:**
   - [QUICKSTART.md](QUICKSTART.md) - Quick reference
   - [README.md](README.md) - Complete documentation

4. **Community:**
   - Check GitHub Issues
   - Create new issue with:
     - Output of `./verify_installation.sh`
     - Output of `./troubleshoot.sh`
     - Relevant log excerpts

---

## Advanced Installation

### Custom Virtual Environment Location

```bash
python3 -m venv /path/to/custom/venv
source /path/to/custom/venv/bin/activate
pip install -r requirements.txt
```

### Using Specific Python Version

```bash
python3.9 -m venv varg_env
source varg_env/bin/activate
pip install -r requirements.txt
```

### Offline Installation

1. Download packages on internet-connected machine:
```bash
pip download -r requirements.txt -d packages/
```

2. Transfer `packages/` directory to offline machine

3. Install from local packages:
```bash
pip install --no-index --find-links=packages/ -r requirements.txt
```

### Development Installation

For development with editable installs:
```bash
python3 -m venv varg_env
source varg_env/bin/activate
pip install -e .
pip install -r requirements.txt
```

---

## Platform-Specific Notes

### Raspberry Pi Zero W
- Use `--system-site-packages` for venv (faster)
- Install numpy/PIL via apt (avoid compilation)
- Use PiWheels for pre-built packages
- Enable camera, SPI, I2C interfaces
- Set GPU memory to 128MB
- Consider overclocking for better performance

### Raspberry Pi 4
- Can use higher resolutions
- Better performance with all models
- May not need `--system-site-packages`

### Mac (Development)
- Uses regular venv without system packages
- Camera features may be limited
- Use USB webcam for testing
- OLED features disabled (Raspberry Pi only)

### Linux (Development)
- Similar to Raspberry Pi setup
- May need `python3-dev` for some packages
- USB webcam support available
- GPIO features disabled (Raspberry Pi only)

---

## Uninstallation

To completely remove V.A.R.G:

```bash
# Stop and disable service (Raspberry Pi)
sudo systemctl stop varg.service
sudo systemctl disable varg.service
sudo rm /etc/systemd/system/varg.service
sudo systemctl daemon-reload

# Remove virtual environment
rm -rf varg_env

# Remove models and logs
rm -rf models detections logs

# Remove configuration (optional)
rm .env

# Keep or remove the repository
cd ..
rm -rf V.A.R.G
```

---

## Next Steps

After successful installation:

1. ✅ Run verification: `./verify_installation.sh`
2. ✅ Add API key: `nano .env`
3. ✅ Start V.A.R.G: `./start_varg.sh`
4. ✅ Monitor system: `./monitor_varg.sh`
5. ✅ Read documentation: [README.md](README.md)

Enjoy using V.A.R.G! 🎉

