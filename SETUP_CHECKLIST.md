# V.A.R.G Setup Checklist

Use this checklist to ensure proper installation and configuration of V.A.R.G.

## 📋 Pre-Installation

### Hardware Requirements
- [ ] Raspberry Pi Zero W (or newer) **OR** Development machine (Mac/Linux)
- [ ] MicroSD card (16GB+ recommended)
- [ ] Power supply (5V, 2A minimum for Pi)
- [ ] USB Camera or Raspberry Pi Camera Module
- [ ] (Optional) Waveshare 1.51" OLED Display
- [ ] Internet connection for initial setup

### Software Requirements  
- [ ] Python 3.7 or higher installed
- [ ] pip installed
- [ ] git installed (recommended)
- [ ] 500MB+ free disk space

### Verify Prerequisites
```bash
python3 --version    # Should show 3.7+
pip3 --version       # Should show version
git --version        # Should show version
df -h .              # Should show 500MB+ available
```

## 🚀 Installation Steps

### Step 1: Clone Repository
- [ ] Clone the repository
  ```bash
  git clone <YOUR_REPO_URL> V.A.R.G
  cd V.A.R.G
  ```

### Step 2: Run Installer
- [ ] Make installer executable
  ```bash
  chmod +x install.sh
  ```

- [ ] Run installation
  ```bash
  ./install.sh
  ```

- [ ] Wait for completion (10-20 minutes on Pi Zero W)

### Step 3: Verify Installation
- [ ] Run verification script
  ```bash
  ./verify_installation.sh
  ```

- [ ] Check all items show ✓ (green checkmarks)

## ⚙️ Configuration

### Step 1: API Key (Optional)
- [ ] Get Groq API key from https://console.groq.com/
- [ ] Add to .env file
  ```bash
  nano .env
  # Add: GROQ_API_KEY=your_actual_key_here
  ```

### Step 2: Review Configuration
- [ ] Open config.json
  ```bash
  nano config.json
  ```

- [ ] Adjust camera settings (if needed)
  - `camera_width`: Default 320 (lower = faster)
  - `camera_height`: Default 240 (lower = faster)
  - `detection_interval`: Default 3.0 seconds

- [ ] Adjust performance settings (if needed)
  - `max_fps`: Default 10
  - `frame_skip`: Default 2
  - `memory_cleanup_interval`: Default 30

- [ ] Configure OLED display (if using)
  - Verify pin assignments match your wiring
  - Default: DC=GPIO25, RST=GPIO24

### Step 3: Hardware Setup (Raspberry Pi Only)

#### Camera
- [ ] Camera module connected properly
- [ ] Camera ribbon cable seated correctly
- [ ] Camera interface enabled
  ```bash
  sudo raspi-config nonint do_camera 0
  ```
- [ ] Test camera
  ```bash
  libcamera-hello --timeout 2000
  ```

#### SPI (for OLED)
- [ ] SPI interface enabled
  ```bash
  sudo raspi-config nonint do_spi 0
  ```
- [ ] Verify SPI device exists
  ```bash
  ls -l /dev/spidev0.0
  ```

#### I2C (alternative for OLED)
- [ ] I2C interface enabled (if using I2C OLED)
  ```bash
  sudo raspi-config nonint do_i2c 0
  ```
- [ ] Verify I2C device exists
  ```bash
  ls -l /dev/i2c-1
  ```

#### OLED Wiring (if applicable)
- [ ] VCC → 3.3V (Pin 1 or 17)
- [ ] GND → Ground (Pin 6, 9, 14, 20, 25, 30, 34, or 39)
- [ ] DIN → MOSI (GPIO10, Pin 19)
- [ ] CLK → SCLK (GPIO11, Pin 23)
- [ ] CS → CE0 (GPIO8, Pin 24)
- [ ] DC → GPIO25 (Pin 22) - configurable in config.json
- [ ] RST → GPIO24 (Pin 18) - configurable in config.json

## 🧪 Testing

### Test 1: Python Environment
- [ ] Activate virtual environment
  ```bash
  source varg_env/bin/activate
  ```

- [ ] Test imports
  ```bash
  python3 -c "import numpy, PIL, psutil, requests; print('✓ OK')"
  ```

### Test 2: Models
- [ ] Check models downloaded
  ```bash
  ls -lh models/*.tflite
  ```
  
- [ ] Should see at least one .tflite file

### Test 3: Hardware (Raspberry Pi)
- [ ] Camera test passed
  ```bash
  libcamera-hello --timeout 2000
  ```

- [ ] SPI enabled
  ```bash
  ls -l /dev/spidev0.0  # Should exist
  ```

### Test 4: Service (Raspberry Pi)
- [ ] Service installed
  ```bash
  systemctl list-unit-files | grep varg
  ```

- [ ] Service enabled
  ```bash
  systemctl is-enabled varg.service
  ```

## 🎯 First Run

### Manual Start (Recommended for first test)
- [ ] Run startup script
  ```bash
  ./start_varg.sh
  ```

- [ ] Watch for errors in output
- [ ] Press Ctrl+C to stop
- [ ] Check logs if any errors
  ```bash
  cat logs/varg.log
  ```

### Service Start (Raspberry Pi)
- [ ] Start service
  ```bash
  sudo systemctl start varg.service
  ```

- [ ] Check status
  ```bash
  sudo systemctl status varg.service
  ```

- [ ] Should show "active (running)"

- [ ] View logs
  ```bash
  sudo journalctl -u varg.service -f
  ```

### Monitoring
- [ ] Run monitor script
  ```bash
  ./monitor_varg.sh
  ```

- [ ] Check CPU usage (should be reasonable)
- [ ] Check memory usage (should be < 80%)
- [ ] Check temperature (should be < 70°C for Pi)

## 🔄 Reboot (Raspberry Pi)

### Before Rebooting
- [ ] All configuration complete
- [ ] Service tested and working
- [ ] No critical errors in logs

### Reboot
- [ ] Reboot to apply all hardware changes
  ```bash
  sudo reboot
  ```

### After Reboot
- [ ] Check service auto-started
  ```bash
  sudo systemctl status varg.service
  ```

- [ ] Check recent logs
  ```bash
  sudo journalctl -u varg.service --since "5 minutes ago"
  ```

- [ ] Verify system running smoothly
  ```bash
  ./monitor_varg.sh
  ```

## ✅ Post-Installation Verification

### System Health
- [ ] CPU usage normal (< 80%)
- [ ] Memory usage normal (< 80%)
- [ ] Temperature normal (< 70°C for Pi)
- [ ] Disk space adequate (> 100MB free)

### Functionality
- [ ] Camera captures frames
- [ ] OLED display shows status (if connected)
- [ ] Detection runs without errors
- [ ] Logs are being written
- [ ] No crash/restart loops

### Performance
- [ ] Frame rate acceptable (5-10 fps on Pi Zero W)
- [ ] Detection interval working (default 3 seconds)
- [ ] System responsive
- [ ] No thermal throttling (Pi)

## 🐛 Troubleshooting

### If Something Doesn't Work

1. [ ] Run troubleshooter
   ```bash
   ./troubleshoot.sh
   ```

2. [ ] Check verification again
   ```bash
   ./verify_installation.sh
   ```

3. [ ] Review logs
   ```bash
   # Application logs
   cat logs/varg.log
   
   # Service logs (Raspberry Pi)
   sudo journalctl -u varg.service -n 100
   ```

4. [ ] Check common issues in [QUICKSTART.md](QUICKSTART.md)

5. [ ] Review [INSTALL.md](INSTALL.md) troubleshooting section

### Common Quick Fixes
- [ ] Restart service: `sudo systemctl restart varg.service`
- [ ] Reactivate venv: `source varg_env/bin/activate`
- [ ] Reinstall packages: `pip install -r requirements.txt --force-reinstall`
- [ ] Reboot system: `sudo reboot`

## 📚 Documentation Reference

- [ ] Read [QUICKSTART.md](QUICKSTART.md) for daily commands
- [ ] Bookmark [README.md](README.md) for full documentation
- [ ] Keep [INSTALL.md](INSTALL.md) for reference
- [ ] Save this checklist for future setups

## 🎉 Setup Complete!

### Final Checklist
- [ ] Installation verified with no errors
- [ ] Configuration complete
- [ ] First run successful
- [ ] Service running (Raspberry Pi)
- [ ] System health good
- [ ] Documentation reviewed
- [ ] Troubleshooting tools tested

### You're Ready!

Your V.A.R.G system is now fully operational. Here's what you can do:

**Monitor:**
```bash
./monitor_varg.sh
sudo journalctl -u varg.service -f
```

**Control:**
```bash
sudo systemctl start varg.service    # Start
sudo systemctl stop varg.service     # Stop
sudo systemctl restart varg.service  # Restart
```

**Update:**
```bash
git pull
./install.sh  # Re-run to update
```

**Get Help:**
```bash
./troubleshoot.sh
./verify_installation.sh
```

---

## 📝 Notes

Use this space to record any customizations or notes:

```
Camera settings used: _______________
Detection interval: _______________
OLED pins: DC=_____ RST=_____
Performance notes: _______________
Issues encountered: _______________
Solutions applied: _______________
```

---

**Date Completed:** _______________
**System:** Raspberry Pi / Development Machine
**Version:** _______________

**Status:** ✅ Ready to use!

