# 🚀 V.A.R.G - START HERE

Welcome to V.A.R.G (Virtual Augmented Reality Glasses)!

## ⚡ Quick Start (2 Steps!)

### 1. Run the Installer
```bash
./install.sh
```

### 2. Start V.A.R.G
```bash
./start_varg.sh
```

**That's it!** 🎉

---

## 📚 What to Read Next

### First Time Users
1. **You are here!** (START_HERE.md)
2. Read [QUICKSTART.md](QUICKSTART.md) - Common commands and tips
3. Browse [README.md](README.md) - Full feature documentation

### Detailed Installation
- See [INSTALL.md](INSTALL.md) for comprehensive installation guide
- Use [SETUP_CHECKLIST.md](SETUP_CHECKLIST.md) to verify everything

### Having Issues?
- Run `./troubleshoot.sh` for automatic diagnosis
- Run `./verify_installation.sh` to check installation
- Check [QUICKSTART.md](QUICKSTART.md) troubleshooting section

---

## 🎯 Common Tasks

### Install
```bash
./install.sh
```

### Start (Manual)
```bash
./start_varg.sh
```

### Start as Service (Raspberry Pi)
```bash
sudo systemctl start varg.service
```

### Check Status
```bash
sudo systemctl status varg.service
```

### View Logs
```bash
sudo journalctl -u varg.service -f
```

### Monitor System
```bash
./monitor_varg.sh
```

### Verify Installation
```bash
./verify_installation.sh
```

### Troubleshoot Issues
```bash
./troubleshoot.sh
```

---

## 📖 Documentation Index

| Document | Purpose |
|----------|---------|
| **START_HERE.md** | You are here - Quick orientation |
| **QUICKSTART.md** | Quick reference for daily use |
| **README.md** | Complete feature documentation |
| **INSTALL.md** | Detailed installation guide |
| **SETUP_CHECKLIST.md** | Step-by-step setup verification |
| **INSTALLATION_SUMMARY.md** | What changed in this version |

---

## 🛠️ Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `install.sh` | ⭐ Main installer | `./install.sh` |
| `start_varg.sh` | Start V.A.R.G manually | `./start_varg.sh` |
| `verify_installation.sh` | Check installation | `./verify_installation.sh` |
| `troubleshoot.sh` | Diagnose issues | `./troubleshoot.sh` |
| `monitor_varg.sh` | System monitor | `./monitor_varg.sh` |
| `setup_models.py` | Download models | `python3 setup_models.py` |

---

## 🎁 What the Installer Does

The `install.sh` script automatically:
- ✅ Detects your platform (Raspberry Pi or development)
- ✅ Installs system dependencies
- ✅ Creates Python virtual environment
- ✅ Installs Python packages
- ✅ Downloads TensorFlow Lite models
- ✅ Sets up Waveshare OLED library
- ✅ Enables hardware interfaces (camera, SPI, I2C)
- ✅ Creates systemd service
- ✅ Verifies everything works

**One command. Everything installed.** ✨

---

## ⚙️ Configuration (Optional)

### Add Groq API Key for LLM Features
```bash
nano .env
# Add: GROQ_API_KEY=your_key_here
```

Get your key: https://console.groq.com/

### Adjust Settings
```bash
nano config.json
```

Common tweaks:
- `camera_width`, `camera_height` - Lower for better performance
- `detection_interval` - Seconds between detections
- `performance.max_fps` - Frame rate limit

---

## 🆘 Need Help?

### Automatic Tools
```bash
./troubleshoot.sh        # Diagnose problems
./verify_installation.sh  # Check everything
```

### Manual Checks
```bash
# Check service
sudo systemctl status varg.service

# View logs
sudo journalctl -u varg.service -f

# Test camera
libcamera-hello --timeout 2000

# Check Python packages
source varg_env/bin/activate
python3 -c "import numpy, PIL; print('OK')"
```

### Documentation
- [QUICKSTART.md](QUICKSTART.md) - Common issues and solutions
- [INSTALL.md](INSTALL.md) - Installation troubleshooting
- [README.md](README.md) - Full documentation

---

## 🎯 Next Steps

After installation:

1. ✅ **Verify** - Run `./verify_installation.sh`
2. ✅ **Configure** - Edit `.env` with your API key (optional)
3. ✅ **Test** - Run `./start_varg.sh` manually first
4. ✅ **Service** - Enable with `sudo systemctl start varg.service` (Pi)
5. ✅ **Monitor** - Use `./monitor_varg.sh` or view logs
6. ✅ **Learn** - Read [QUICKSTART.md](QUICKSTART.md)

---

## 💡 Pro Tips

### For Best Performance (Raspberry Pi Zero W)
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

### For Maximum Accuracy
```json
{
  "camera_width": 640,
  "camera_height": 480,
  "detection_interval": 2.0,
  "performance": {
    "max_fps": 15,
    "frame_skip": 1
  }
}
```

### For Battery Saving
```json
{
  "camera_width": 224,
  "detection_interval": 5.0,
  "performance": {
    "max_fps": 6,
    "frame_skip": 4
  }
}
```

---

## 📊 System Requirements

### Minimum
- Raspberry Pi Zero W or development machine
- Python 3.7+
- 512MB RAM
- 500MB disk space
- Internet for initial setup

### Recommended
- Raspberry Pi Zero W or Pi 4
- Python 3.9+
- 1GB RAM
- 2GB disk space
- Camera module
- OLED display (optional)

---

## 🚀 Ready to Go!

Your installation command:
```bash
./install.sh
```

**Questions?**
- Check [QUICKSTART.md](QUICKSTART.md)
- Run `./troubleshoot.sh`
- Read [README.md](README.md)

**Enjoy V.A.R.G!** 🎉

---

*For detailed information about changes in this version, see [INSTALLATION_SUMMARY.md](INSTALLATION_SUMMARY.md)*

