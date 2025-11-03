# V.A.R.G -  Virtual Augmented Reality Glasses

A food detection and calorie estimation system designed for Raspberry Pi Zero W, using lightweight computer vision (PIL/scikit-image), TensorFlow Lite, and Groq LLM integration.

## 🎯 Features

- **TensorFlow Lite Food Detection**: Uses pre-trained TFLite models for accurate food classification
- **Real-time Food Detection**: Lightweight computer vision with PIL and scikit-image
- **AI-Powered Analysis**: Integrates with Groq LLM for intelligent food identification and calorie estimation
- **Raspberry Pi Optimized**: Specifically optimized for Raspberry Pi Zero W performance
- **Smart Detection Pipeline**: Only triggers LLM analysis when food is actually detected
- **Transparent OLED Display**: Shows real-time detection status and calorie information
- **Asynchronous Processing**: Non-blocking LLM analysis with background threading
- **Multiple Detection Methods**: TFLite models with lightweight PIL/scikit-image fallback

## 🛠️ Hardware Requirements

- Raspberry Pi Zero W
- USB Camera or Raspberry Pi Camera Module
- MicroSD card (16GB+ recommended)
- Power supply (5V, 2A recommended)

## 📦 Setup & Execution

### Quick Start — Clone and Auto-Start on Raspberry Pi

```bash
# On your Raspberry Pi (Zero W or newer)
sudo apt update && sudo apt install -y git

# Clone the repo (replace <YOUR_REPO_URL> if needed)
cd /home/pi
git clone <YOUR_REPO_URL> V.A.R.G
cd V.A.R.G

# Deploy (creates venv, installs deps, sets up models/service, enables auto-start)
chmod +x deploy_pi.sh
./deploy_pi.sh

# Add your Groq API key (optional; LLM features disabled if omitted)
echo "GROQ_API_KEY=your_actual_groq_api_key_here" > .env

# Reboot to apply firmware config and start the service
sudo reboot
```

Run the system:
```bash
# Start immediately (manual run)
source varg_env/bin/activate
python3 v.a.r.g.py

# Or use the service (auto-start already enabled by deploy_pi.sh)
sudo systemctl status varg.service
sudo journalctl -u varg.service -f
```

Monitor:
```bash
./monitor_varg.sh
sudo journalctl -u varg.service -f
```

### Lightweight Detection Stack

- Default pipeline is optimized for Raspberry Pi Zero W:
  - TensorFlow Lite runtime (int8 EfficientNet‑Lite) for food classification
  - PIL for preprocessing (brightness/contrast/blur), fast and low‑memory
  - scikit‑image (optional) for connected components; can be omitted if needed
### Smart Glasses Profile (Performance-Friendly)

Recommended `config.json` changes for longer battery life and lower heat:
```json
{
  "camera_width": 224,
  "camera_height": 224,
  "detection_interval": 4.0,
  "performance": { "max_fps": 8, "frame_skip": 3, "low_quality_threshold": 70 }
}
```

Tips:
- Keep `save_images` set to `false`.
- Use `tflite-runtime` (already preferred) instead of full TensorFlow.
- Short I2C wiring for the OLED; confirm with `sudo i2cdetect -y 1`.

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### Configuration File (`config.json`)

```json
{
  "groq_api_key": "",
  "camera_index": 0,
  "camera_width": 320,
  "camera_height": 240,
  "detection_confidence": 0.5,
  "calorie_estimation_model": "llama-3.2-11b-vision-preview",
  "detection_interval": 3.0,
  "save_images": false,
  "output_dir": "detections",
  "preprocessing": {
    "blur_kernel": 3,
    "brightness_adjustment": 1.1,
    "contrast_adjustment": 1.05
  },
  "performance": {
    "max_fps": 10,
    "frame_skip": 2,
    "low_quality_threshold": 75,
    "memory_cleanup_interval": 30
  },
  "oled_display": {
    "width": 128,
    "height": 64,
    "update_interval": 1.0,
    "show_system_info": true
  },
  "tflite": {
    "enabled": true,
    "model_path": "kagglehub:google/aiy/tfLite/vision-classifier-food-v1",
    "confidence_threshold": 0.35,
    "max_detections": 3,
    "use_fallback_cv": true
  }
}
```

## 🚀 Usage

### Basic Usage

```bash
# Activate virtual environment
source varg_env/bin/activate

# Run the food detection system
python3 v.a.r.g.py
```

### Running as a Service

To run V.A.R.G automatically on boot:

```bash
sudo systemctl enable varg.service
sudo systemctl start varg.service
```

Monitor the service:
```bash
sudo systemctl status varg.service
journalctl -u varg.service -f
```

## 📊 How It Works

1. **Camera Capture**: Continuously captures frames from the camera
2. **Preprocessing**: Applies brightness, contrast, and blur adjustments
3. **Computer Vision Detection**: Uses color-based detection and contour analysis to identify potential food objects
4. **AI Analysis**: Sends detected regions to Groq LLM for food identification and calorie estimation
5. **Results Display**: Shows detected foods, portion sizes, and calorie estimates
6. **Data Logging**: Saves results and images for later analysis

## 🎛️ Detection Methods

### Computer Vision Techniques
- **Color-based detection**: Identifies foods by color ranges (red, green, yellow, brown)
- **Contour analysis**: Finds object boundaries and filters by size and aspect ratio
- **Background subtraction**: Detects changes in the scene
- **Morphological operations**: Cleans up detection masks

### AI Integration
- **Groq LLM**: Uses advanced language models for food identification
- **Image analysis**: Processes captured images to identify specific foods
- **Calorie estimation**: Provides calorie estimates based on visual portion analysis
- **Confidence scoring**: Returns confidence levels for identifications

## 📁 Output Structure

```
detections/
├── detection_20231021_143022.jpg    # Captured images
├── result_20231021_143022.json      # Analysis results
└── ...

logs/
└── varg.log                         # System logs
```

## 🔧 Troubleshooting

### Camera Issues
```bash
# Enable camera interface
sudo raspi-config nonint do_camera 0

# Quick test (Picamera2)
python3 - << 'PY'
from picamera2 import Picamera2
from time import sleep
cam = Picamera2()
cam.start()
sleep(1)
arr = cam.capture_array()
cam.stop()
print('Camera OK' if arr is not None else 'Camera Error')
PY

# Or use libcamera tool
libcamera-hello --timeout 2000
```

### Performance Optimization
- Reduce `camera_width` and `camera_height` in config.json
- Increase `detection_interval` to reduce processing frequency
- Disable `save_images` if storage is limited

### Memory Issues
```bash
# Increase GPU memory split
sudo raspi-config nonint do_memory_split 128

# Monitor memory usage
free -h
```

## 🔑 API Keys

Get your Groq API key from [Groq Console](https://console.groq.com/):
1. Sign up for a Groq account
2. Navigate to API Keys section
3. Create a new API key
4. Add it to your `.env` file

## 📈 Performance Tips

### For Raspberry Pi Zero W:
- Use lower resolution (320x240 or 640x480)
- Increase detection interval (3-5 seconds)
- Disable display output in headless mode
- Use efficient image preprocessing settings

### Memory Management:
- The system automatically manages detection history (keeps last 10 results)
- Images are compressed before sending to API
- Configurable output directory for storage management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on Raspberry Pi hardware
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.