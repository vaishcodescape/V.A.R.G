# V.A.R.G Quick Start Guide

## 🚀 Installation (One Command!)

### Raspberry Pi Zero W
```bash
git clone <YOUR_REPO_URL> V.A.R.G
cd V.A.R.G
./install.sh
```

That's it! The installer handles everything automatically.

**Verify installation:**
```bash
./verify_installation.sh
```

### Development Machine (Mac/Linux)
```bash
git clone <YOUR_REPO_URL> V.A.R.G
cd V.A.R.G
./install.sh
```

**Verify installation:**
```bash
./verify_installation.sh
```

## ⚡ First Run

### 1. Add API Key (Optional)
```bash
nano .env
# Add: GROQ_API_KEY=your_key_here
```

### 2. Start V.A.R.G

**Easy way:**
```bash
./start_varg.sh
```

**Manual way:**
```bash
source varg_env/bin/activate
python3 v.a.r.g.py
```

**As a service (Raspberry Pi):**
```bash
sudo systemctl start varg.service
```

## 📊 Monitoring

**Check service status:**
```bash
sudo systemctl status varg.service
```

**View logs:**
```bash
sudo journalctl -u varg.service -f
```

**System monitor:**
```bash
./monitor_varg.sh
```

## 🛠️ Common Commands

### Start/Stop Service
```bash
sudo systemctl start varg.service    # Start
sudo systemctl stop varg.service     # Stop
sudo systemctl restart varg.service  # Restart
sudo systemctl status varg.service   # Check status
```

### View Logs
```bash
# Real-time logs
sudo journalctl -u varg.service -f

# Last 50 lines
sudo journalctl -u varg.service -n 50

# Logs from today
sudo journalctl -u varg.service --since today
```

### Enable/Disable Auto-Start
```bash
sudo systemctl enable varg.service   # Auto-start on boot
sudo systemctl disable varg.service  # Disable auto-start
```

## 🔧 Troubleshooting

**Quick Diagnosis:**
```bash
./troubleshoot.sh
```

This script will automatically detect and suggest fixes for common issues.

### Installation Issues

**Problem:** Missing system packages
```bash
# On Raspberry Pi, run:
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-dev
```

**Problem:** Virtual environment not working
```bash
# Recreate it:
rm -rf varg_env
python3 -m venv varg_env
source varg_env/bin/activate
pip install -r requirements.txt
```

### Camera Issues

**Problem:** Camera not detected
```bash
# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options -> Camera -> Enable

# Or use command line:
sudo raspi-config nonint do_camera 0

# Test camera
libcamera-hello --timeout 2000
```

**Problem:** Camera busy error
```bash
# Kill other camera processes
sudo pkill -9 libcamera
sudo systemctl restart varg.service
```

### OLED Display Issues

**Problem:** OLED not working
```bash
# Check SPI is enabled
sudo raspi-config nonint do_spi 0

# Check I2C is enabled (if using I2C OLED)
sudo raspi-config nonint do_i2c 0

# Verify wiring and pin configuration in config.json
```

**Problem:** GPIO busy error
```bash
# Restart the service
sudo systemctl restart varg.service

# Or reboot
sudo reboot
```

### Performance Issues

**Problem:** System running slow
```bash
# Check CPU and memory usage
./monitor_varg.sh

# Or use system tools
htop
free -h
```

**Optimize config.json:**
```json
{
  "camera_width": 224,
  "camera_height": 224,
  "detection_interval": 4.0,
  "performance": {
    "max_fps": 8,
    "frame_skip": 3
  }
}
```

### Service Won't Start

**Check logs:**
```bash
sudo journalctl -u varg.service -n 50
```

**Common fixes:**
```bash
# Check .env file exists and has proper format
cat .env

# Verify Python environment
source varg_env/bin/activate
python3 -c "import numpy, PIL; print('OK')"

# Reinstall dependencies
source varg_env/bin/activate
pip install -r requirements.txt --force-reinstall

# Check file permissions
chmod +x v.a.r.g.py
chmod +x start_varg.sh
```

## 📁 File Structure

```
V.A.R.G/
├── install.sh              # 🌟 Main installer (use this!)
├── v.a.r.g.py             # Main application
├── start_varg.sh          # Quick start script
├── monitor_varg.sh        # System monitor
├── config.json            # Configuration
├── .env                   # API keys (create this)
├── requirements.txt       # Python dependencies
├── varg_env/              # Virtual environment (auto-created)
├── models/                # TensorFlow Lite models (auto-downloaded)
├── detections/            # Saved detections
├── logs/                  # Log files
└── Raspberry/             # Waveshare OLED library
    └── python/
        └── lib/
```

## 🔑 Environment Variables (.env)

```bash
# Required for LLM features (optional otherwise)
GROQ_API_KEY=your_groq_api_key_here

# Optional overrides
CAMERA_INDEX=0
DETECTION_INTERVAL=3.0
```

## ⚙️ Configuration (config.json)

Key settings you might want to adjust:

```json
{
  "camera_width": 320,           // Lower = faster
  "camera_height": 240,          // Lower = faster
  "detection_interval": 3.0,     // Seconds between detections
  "detection_confidence": 0.5,   // Detection threshold
  "save_images": false,          // Save detected images?
  
  "performance": {
    "max_fps": 10,               // Max frames per second
    "frame_skip": 2,             // Skip frames for performance
    "memory_cleanup_interval": 30 // Cleanup interval (seconds)
  },
  
  "oled_display": {
    "width": 128,                // OLED width
    "height": 64,                // OLED height
    "update_interval": 1.0       // Display update rate
  }
}
```

## 💡 Tips

### For Best Performance on Raspberry Pi Zero W

1. **Lower resolution:** Set camera to 224x224 or 320x240
2. **Increase interval:** Set detection_interval to 4-5 seconds
3. **Disable image saving:** Set save_images to false
4. **Use power mode:** Ensure good 5V 2A power supply
5. **Optimize memory:** Limit background processes

### Battery Life (Portable Use)

```json
{
  "camera_width": 224,
  "detection_interval": 5.0,
  "performance": {
    "max_fps": 8,
    "frame_skip": 4
  }
}
```

### High Accuracy (With good cooling)

```json
{
  "camera_width": 640,
  "detection_interval": 2.0,
  "performance": {
    "max_fps": 15,
    "frame_skip": 1
  }
}
```

## 🆘 Getting Help

1. Check the logs: `sudo journalctl -u varg.service -f`
2. Run system check: `python3 validate_system.py`
3. Check monitor: `./monitor_varg.sh`
4. Review README.md for detailed documentation
5. Check GitHub issues or create a new one

## 📚 Additional Resources

- **Full Documentation:** See [README.md](README.md)
- **Groq API Keys:** https://console.groq.com/
- **Raspberry Pi Setup:** https://www.raspberrypi.com/documentation/
- **Waveshare OLED:** https://www.waveshare.com/wiki/1.51inch_OLED_Module

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Install | `./install.sh` |
| Start | `./start_varg.sh` |
| Start service | `sudo systemctl start varg.service` |
| Stop service | `sudo systemctl stop varg.service` |
| View logs | `sudo journalctl -u varg.service -f` |
| Monitor | `./monitor_varg.sh` |
| Edit config | `nano config.json` |
| Edit API key | `nano .env` |
| Reboot | `sudo reboot` |
| Update code | `git pull` |

---

**Need more help?** Check [README.md](README.md) for comprehensive documentation.

