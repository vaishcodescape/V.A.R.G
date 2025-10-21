<<<<<<< HEAD
# V.A.R.G {Virtual Augmented Reality Glasses}
This is the Python Script for our Virtually Augmented Reality Glasses which integrates Python Libraries like OpenCV with RESTful API's and modern LLM's. It was a part of our B.Tech Course in the Exploratory Project an academic requirement in the B.Tech ICT Branch.
=======
# V.A.R.G -  Virtual Augmented Reality Glasses

A sophisticated food detection and calorie estimation system designed for Raspberry Pi Zero W, using OpenCV computer vision and Groq LLM integration.

## 🎯 Features

- **TensorFlow Lite Food Detection**: Uses pre-trained TFLite models for accurate food classification
- **Real-time Food Detection**: Lightweight computer vision with PIL and scikit-image
- **AI-Powered Analysis**: Integrates with Groq LLM for intelligent food identification and calorie estimation
- **Raspberry Pi Optimized**: Specifically optimized for Raspberry Pi Zero W performance
- **Smart Detection Pipeline**: Only triggers LLM analysis when food is actually detected
- **Transparent OLED Display**: Shows real-time detection status and calorie information
- **Asynchronous Processing**: Non-blocking LLM analysis with background threading
- **Multiple Detection Methods**: TFLite models with traditional CV fallback

## 🛠️ Hardware Requirements

- Raspberry Pi Zero W
- USB Camera or Raspberry Pi Camera Module
- MicroSD card (16GB+ recommended)
- Power supply (5V, 2A recommended)

## 📦 Setup & Execution

### Option A — Automated Setup (Recommended)

```bash
# On your Raspberry Pi (Zero W or newer)
cd /home/pi
cd V.A.R.G  # or your chosen directory name

# Make deployment script executable and run it
chmod +x deploy_pi.sh
./deploy_pi.sh

# Add your Groq API key
nano .env
# GROQ_API_KEY=your_actual_groq_api_key_here

# Optional: tweak configuration
nano config.json
```

Run the system:
```bash
# Start immediately (manual run)
source varg_env/bin/activate
python3 v.a.r.g.py

# Or run as a service (auto-start on boot)
sudo systemctl start varg.service
sudo systemctl enable varg.service
```

Monitor:
```bash
./monitor_varg.sh
sudo journalctl -u varg.service -f
```

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
  "camera_width": 640,
  "camera_height": 480,
  "detection_confidence": 0.5,
  "calorie_estimation_model": "llama3-70b-8192",
  "detection_interval": 2.0,
  "save_images": true,
  "output_dir": "detections",
  "preprocessing": {
    "blur_kernel": 5,
    "brightness_adjustment": 1.2,
    "contrast_adjustment": 1.1
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
# Test camera
python3 -c "import cv2; print('Camera OK' if cv2.VideoCapture(0).isOpened() else 'Camera Error')"

# Enable camera interface
sudo raspi-config nonint do_camera 0
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

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- Groq for LLM API access
- Raspberry Pi Foundation for affordable computing hardware

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review system logs: `tail -f varg.log`
3. Test individual components (camera, API connection)
4. Open an issue with detailed error information

---

**V.A.R.G** - Making food tracking intelligent and automated! 🍽️🤖
>>>>>>> ec9c8a2 (python script init commit)
