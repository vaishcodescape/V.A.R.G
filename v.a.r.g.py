#!/usr/bin/env python3
"""
V.A.R.G - Visual Automated Recipe & Grocery
Food Detection and Calorie Estimation System for Raspberry Pi Zero W
Using OpenCV (capture), lightweight PIL/numpy processing, optional remote inference via HTTP,
and Transparent OLED Display. Optimized for Pi Camera Module and minimal resource usage.
"""

import numpy as np
import json
import time
import logging
from datetime import datetime
from typing import Dict, List
import os
import requests
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import queue
import gc
from collections import deque
import psutil

# Pillow resampling compatibility (older Pillow on Pi may lack Image.Resampling)
try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS  # Pillow >=9.1
except Exception:
    RESAMPLE_LANCZOS = getattr(Image, "LANCZOS", Image.BICUBIC)

# Load environment variables from .env if present (optional on Pi)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env", override=False)
except Exception:
    pass

# Heavy scientific CV libs like scikit-image are not used on Pi Zero W

# TensorFlow/TFLite are disabled on Pi Zero W
TFLITE_AVAILABLE = False

# Fallback to minimal OpenCV only for camera operations
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logging.warning("OpenCV not available")

# Try to import Pi-specific libraries
try:
    from picamera2 import Picamera2
    PI_CAMERA_AVAILABLE = True
except ImportError:
    PI_CAMERA_AVAILABLE = False
    logging.warning("Picamera2 not available, falling back to OpenCV")

# Try to import OLED display libraries
try:
    import board
    import busio
    import digitalio
    import displayio
    try:
        # displayio SSD1306 path (fallback)
        import adafruit_displayio_ssd1306
        DISPLAYIO_SSD1306_AVAILABLE = True
    except Exception:
        DISPLAYIO_SSD1306_AVAILABLE = False
    OLED_AVAILABLE = True
except ImportError:
    OLED_AVAILABLE = False
    logging.warning("OLED display libraries not available")

# Optional: Waveshare SPI OLED driver (SSD1309/SSD1306 variants)
try:
    from waveshare_OLED import OLED_1in51 as WS_OLED_1in51
    WAVESHARE_AVAILABLE = True
except Exception:
    WAVESHARE_AVAILABLE = False

# Prefer lightweight luma.oled SPI driver for Pi Zero W
try:
    from luma.core.interface.serial import spi as luma_spi
    from luma.core.render import canvas as luma_canvas
    from luma.oled.device import ssd1306 as luma_ssd1306
    try:
        from luma.oled.device import ssd1309 as luma_ssd1309
        LUMA_SSD1309_AVAILABLE = True
    except Exception:
        LUMA_SSD1309_AVAILABLE = False
    LUMA_AVAILABLE = True
except Exception:
    LUMA_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('varg.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LightweightCV:
    """Lightweight computer vision operations using PIL and numpy"""
    
    @staticmethod
    def pil_to_numpy(pil_image):
        """Convert PIL image to numpy array"""
        return np.array(pil_image)
    
    @staticmethod
    def numpy_to_pil(np_array):
        """Convert numpy array to PIL image"""
        if np_array.dtype != np.uint8:
            if 'float' in str(np_array.dtype):
                np_array = (np.clip(np_array, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                np_array = np.clip(np_array, 0, 255).astype(np.uint8)
        return Image.fromarray(np_array)
    
    @staticmethod
    def resize_image(image, size, method=None):
        """Resize image efficiently"""
        if isinstance(image, np.ndarray):
            image = LightweightCV.numpy_to_pil(image)
        if method is None:
            method = RESAMPLE_LANCZOS
        return image.resize(size, method)
    
    @staticmethod
    def gaussian_blur(image, radius=2):
        """Apply Gaussian blur using PIL (much faster than OpenCV on Pi Zero W)"""
        if isinstance(image, np.ndarray):
            image = LightweightCV.numpy_to_pil(image)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    @staticmethod
    def enhance_image(image, brightness=1.0, contrast=1.0, saturation=1.0):
        """Enhance image using PIL (very efficient)"""
        if isinstance(image, np.ndarray):
            image = LightweightCV.numpy_to_pil(image)
        
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
        
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)
        
        return image
    
    @staticmethod
    def convert_color_space(image, mode='HSV'):
        """Convert color space using PIL"""
        if isinstance(image, np.ndarray):
            image = LightweightCV.numpy_to_pil(image)
        
        if mode == 'HSV':
            return image.convert('HSV')
        elif mode == 'LAB':
            return image.convert('LAB')
        elif mode == 'L':
            return image.convert('L')
        else:
            return image
    
    @staticmethod
    def threshold_image(image, threshold=128):
        """Simple thresholding using PIL"""
        if isinstance(image, np.ndarray):
            image = LightweightCV.numpy_to_pil(image)
        
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply threshold
        return image.point(lambda x: 255 if x > threshold else 0, mode='1')
    
    @staticmethod
    def find_edges_pil(image, threshold1=50, threshold2=150):
        """Edge detection using PIL filters (lighter than OpenCV Canny)"""
        if isinstance(image, np.ndarray):
            image = LightweightCV.numpy_to_pil(image)
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply edge detection filter
        edges = image.filter(ImageFilter.FIND_EDGES)
        
        # Enhance edges
        enhancer = ImageEnhance.Contrast(edges)
        edges = enhancer.enhance(2.0)
        
        return edges
    
    @staticmethod
    def color_mask_pil(image, color_ranges):
        """Create color mask using PIL (much lighter than OpenCV)"""
        if isinstance(image, np.ndarray):
            image = LightweightCV.numpy_to_pil(image)
        
        # Convert to HSV for better color detection
        hsv_image = image.convert('HSV')
        hsv_array = np.array(hsv_image)
        
        # Create combined mask
        mask = np.zeros(hsv_array.shape[:2], dtype=np.uint8)
        
        for color_name, color_range in color_ranges.items():
            # Create mask for this color range
            lower = np.array(color_range['lower'])
            upper = np.array(color_range['upper'])
            
            color_mask = np.all((hsv_array >= lower) & (hsv_array <= upper), axis=2)
            mask = np.logical_or(mask, color_mask)
        
        return Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    
    @staticmethod
    def find_contours_pil(mask_image, min_area=100):
        """Find contours using PIL and basic image processing"""
        if isinstance(mask_image, Image.Image):
            mask_array = np.array(mask_image)
        else:
            mask_array = mask_image
        
        # Create a single bounding box over non-zero area (fast, low memory)
        try:
            pil_mask = mask_image if isinstance(mask_image, Image.Image) else Image.fromarray(mask_array)
            bbox = pil_mask.getbbox()
            if not bbox:
                return []
            x0, y0, x1, y1 = bbox
            w, h = max(0, x1 - x0), max(0, y1 - y0)
            area = w * h
            if area < min_area:
                return []
            return [{
                'bbox': (x0, y0, w, h),
                'area': area,
                'centroid': (y0 + h / 2.0, x0 + w / 2.0)
            }]
        except Exception:
            return []
    
    @staticmethod
    def morphological_operations(image, operation='close', kernel_size=5):
        """Morphological operations using simple PIL filters"""
        if isinstance(image, np.ndarray):
            image = LightweightCV.numpy_to_pil(image)
        
        if operation == 'close':
            for _ in range(max(1, kernel_size // 2)):
                image = image.filter(ImageFilter.MaxFilter(3))
            for _ in range(max(1, kernel_size // 2)):
                image = image.filter(ImageFilter.MinFilter(3))
        
        return image

class RemoteInferenceClient:
    """Remote inference via HTTP API to keep Pi Zero W lightweight"""
    
    def __init__(self, url: str = "", timeout: float = 8.0, max_upload_kb: int = 512):
        self.url = url or ""
        self.timeout = float(timeout or 8.0)
        self.max_upload_kb = int(max_upload_kb or 512)
    
    def encode_image_jpeg_base64(self, image: Image.Image, max_side: int = 320) -> str:
        """Resize and JPEG-encode image to stay within max_upload_kb"""
        try:
            if not isinstance(image, Image.Image):
                image = LightweightCV.numpy_to_pil(image)
            
            img = image.copy()
            img.thumbnail((max_side, max_side), resample=RESAMPLE_LANCZOS)
            
            quality = 80
            for _ in range(4):
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=quality, optimize=True)
                data = buf.getvalue()
                kb = len(data) // 1024
                if kb <= self.max_upload_kb or quality <= 55:
                    return "data:image/jpeg;base64," + base64.b64encode(data).decode()
                quality -= 8
            
            return "data:image/jpeg;base64," + base64.b64encode(data).decode()
        except Exception:
            return ""
    
    def analyze(self, image: Image.Image, detected_objects: List[Dict]) -> Dict:
        """Call remote inference endpoint and return analysis JSON"""
        if not self.url:
            return {"foods_detected": [], "analysis_notes": "Remote inference disabled"}
        
        try:
            img_b64 = self.encode_image_jpeg_base64(image)
            if not img_b64:
                return {"foods_detected": [], "analysis_notes": "Image encode failed"}
            
            hints = []
            for obj in (detected_objects or [])[:3]:
                hints.append({
                    "area": int(obj.get("area", 0)),
                    "method": obj.get("detection_method", "cv")
                })
            
            payload = {"image": img_b64, "hints": hints}
            resp = requests.post(self.url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                return {"foods_detected": [], "analysis_notes": "Invalid response"}
            data.setdefault("foods_detected", [])
            return data
        except Exception as e:
            return {"foods_detected": [], "analysis_notes": f"Remote inference error: {e}"}

class OLEDDisplay:
    """Handles transparent OLED display for showing detection results"""
    
    def __init__(self, config: Dict):
        """Initialize OLED with a config dictionary."""
        self.config = config
        self.width = self.config.get("width", 128)
        self.height = self.config.get("height", 64)
        self.display = None
        self.luma_device = None
        self.font = None
        self.small_font = None
        self.rotate = int(self.config.get("rotate", 0) or 0)
        self.invert = bool(self.config.get("invert", False))
        # Track last frame hash to skip redundant SPI updates
        self._last_frame_hash = None
        
        self.init_display()
    
    def show_message(self, lines):
        """Show a simple multi-line centered message on the OLED."""
        if not self.display and not self.luma_device:
            return
        try:
            img = Image.new('1', (self.width, self.height), 0)
            draw = ImageDraw.Draw(img)
            # Choose fonts
            title_font = self.font or ImageFont.load_default()
            text_font = self.small_font or ImageFont.load_default()
            # Compute total height
            heights = []
            for i, line in enumerate(lines):
                f = title_font if i == 0 else text_font
                _, h = f.getsize(line)
                heights.append(h)
            total_h = sum(heights) + max(0, (len(lines) - 1)) * 2
            y = max(0, (self.height - total_h) // 2)
            # Draw lines
            for i, line in enumerate(lines):
                f = title_font if i == 0 else text_font
                w, h = f.getsize(line)
                x = max(0, (self.width - w) // 2)
                draw.text((x, y), line, font=f, fill=1)
                y += h + 2
            self.update_display(img)
        except Exception as e:
            logger.error(f"Error showing OLED message: {e}")
    
    def show_startup(self):
        """Show startup splash to indicate process start."""
        self.show_message(["V.A.R.G", "Food Calorie Detector"])
    
    def init_display(self):
        """Initialize the OLED display using SPI (prefer luma.oled)."""
        logger.info("Initializing OLED display (SPI)...")
        
        # Load fonts
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
            self.small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except (OSError, IOError):
            self.font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()

        # Attempt luma.oled SPI first (fastest, reliable on Pi)
        if LUMA_AVAILABLE:
            try:
                spi_cfg = self.config.get("spi", {}) if isinstance(self.config.get("spi", {}), dict) else {}
                bus = int(spi_cfg.get("bus", 0))
                device = int(spi_cfg.get("device", 0))
                baudrate = int(spi_cfg.get("baudrate", 8000000))
                gpio_dc = int(spi_cfg.get("dc_pin_bcm", 25))
                gpio_rst = int(spi_cfg.get("rst_pin_bcm", 24))
                driver = str(spi_cfg.get("driver", "ssd1309")).lower()
                
                serial = luma_spi(port=bus, device=device, gpio_DC=gpio_dc, gpio_RST=gpio_rst, bus_speed_hz=baudrate)
                if driver == "ssd1306":
                    device_obj = luma_ssd1306(serial, width=self.width, height=self.height)
                elif driver == "ssd1309" and LUMA_SSD1309_AVAILABLE:
                    device_obj = luma_ssd1309(serial, width=self.width, height=self.height)
                else:
                    # Fallback order: SSD1309 -> SSD1306
                    device_obj = luma_ssd1309(serial, width=self.width, height=self.height) if LUMA_SSD1309_AVAILABLE else luma_ssd1306(serial, width=self.width, height=self.height)
                
                # Apply rotation/inversion if requested
                try:
                    if self.rotate:
                        device_obj.rotate = self.rotate
                except Exception:
                    pass
                self.luma_device = device_obj
                logger.info(f"OLED initialized via luma.oled ({driver}, SPI {bus}.{device}, DC={gpio_dc}, RST={gpio_rst})")
                return
            except Exception as e:
                logger.warning(f"luma.oled SPI init failed: {e}")
        
        # Fallback to Waveshare SPI driver
        try:
            self.init_spi_display_waveshare_or_displayio()
            return
        except Exception as e:
            logger.error(f"Failed to initialize OLED display (fallbacks): {e}")
            self.display = None

    def init_spi_display_waveshare_or_displayio(self):
        """Initialize SPI-based OLED display."""
        # Prefer Waveshare driver for 1.51" SPI OLEDs (SSD1309/SSD1306)
        if WAVESHARE_AVAILABLE:
            self.ws = WS_OLED_1in51.OLED_1in51()
            self.ws.Init()
            self.ws.clear()
            # Use panel-reported dimensions
            try:
                self.width = getattr(self.ws, "width", self.width)
                self.height = getattr(self.ws, "height", self.height)
            except Exception:
                pass
            self.display = self.ws  # mark as initialized
            logger.info("OLED display initialized via Waveshare driver (SPI)")
        else:
            # Fallback to displayio SSD1306 over SPI
            if not OLED_AVAILABLE or not DISPLAYIO_SSD1306_AVAILABLE:
                raise RuntimeError("SPI display fallback unavailable (displayio/adfr SSD1306 not installed)")
            displayio.release_displays()
            spi_config = self.config.get("spi", {})
            cs_pin_name = spi_config.get("cs_pin", "CE0")
            dc_pin_name = spi_config.get("dc_pin", "D25")
            rst_pin_name = spi_config.get("rst_pin", "D24")
            baudrate = spi_config.get("baudrate", 2000000)

            spi = busio.SPI(board.SCLK, board.MOSI)
            
            try:
                cs_pin = getattr(board, cs_pin_name)
            except AttributeError:
                logger.error(f"Invalid CS pin name '{cs_pin_name}', defaulting to CE0.")
                cs_pin = board.CE0

            tft_cs = digitalio.DigitalInOut(cs_pin)
            tft_dc = digitalio.DigitalInOut(getattr(board, dc_pin_name))
            tft_rst = digitalio.DigitalInOut(getattr(board, rst_pin_name))

            display_bus = displayio.FourWire(
                spi, command=tft_dc, chip_select=tft_cs, reset=tft_rst, baudrate=baudrate
            )
            self.display = adafruit_displayio_ssd1306.SSD1306(display_bus, width=self.width, height=self.height)
            logger.info("OLED display initialized via displayio SSD1306 (SPI)")

    def create_food_display(self, foods_data: Dict, food_detected: bool = False, llm_processing: bool = False) -> Image.Image:
        """Create display image showing only food name and calories."""
        img = Image.new('1', (self.width, self.height), 0)  # 1-bit image for OLED
        draw = ImageDraw.Draw(img)
        
        # Only render when we have at least one food item; otherwise keep screen blank
        try:
            foods = (foods_data or {}).get("foods_detected", [])
            if not foods:
                return img
            
            first = foods[0]
            name = str(first.get("name", "")).strip()
            cal_val = first.get("calories")
            # Fallback to calories_per_100g if provided by some endpoints
            if cal_val is None:
                cal_val = first.get("calories_per_100g", 0)
            calories = int(cal_val) if isinstance(cal_val, (int, float, str)) and str(cal_val).isdigit() else 0
            
            # Format strings
            name = name[:16] if name else "Food"
            calories_text = f"{calories} kcal"
            
            # Center text horizontally
            try:
                name_w, name_h = draw.textsize(name, font=self.font)
                cal_w, cal_h = draw.textsize(calories_text, font=self.small_font)
            except Exception:
                # Fallback sizes if PIL lacks textsize
                name_w, name_h = (len(name) * 6, 12)
                cal_w, cal_h = (len(calories_text) * 6, 10)
            
            name_x = max(0, (self.width - name_w) // 2)
            cal_x = max(0, (self.width - cal_w) // 2)
            
            # Vertical positioning
            name_y = 14
            cal_y = name_y + name_h + 8
            
            draw.text((name_x, name_y), name, font=self.font, fill=1)
            draw.text((cal_x, cal_y), calories_text, font=self.small_font, fill=1)
            
            return img
        except Exception as e:
            logger.debug(f"OLED render fallback: {e}")
            return img
    
    def show_system_info(self, cpu_usage: float, memory_usage: float, fps: float):
        """Display system performance information"""
        if not self.display:
            return
        
        img = Image.new('1', (self.width, self.height), 0)
        draw = ImageDraw.Draw(img)
        
        draw.text((2, 2), "SYSTEM INFO", font=self.font, fill=1)
        draw.text((2, 18), f"CPU: {cpu_usage:.1f}%", font=self.small_font, fill=1)
        draw.text((2, 30), f"MEM: {memory_usage:.1f}%", font=self.small_font, fill=1)
        draw.text((2, 42), f"FPS: {fps:.1f}", font=self.small_font, fill=1)
        
        self.update_display(img)
    
    def update_display(self, image: Image.Image):
        """Update the OLED display with new image"""
        # No device initialized
        if not self.display and not self.luma_device:
            return
        
        try:
            frame = image.convert('1')
            if self.rotate:
                try:
                    frame = frame.rotate(self.rotate, expand=False)
                except Exception:
                    pass
            if self.invert:
                try:
                    frame = Image.eval(frame, lambda p: 255 - p)
                except Exception:
                    pass

            # Skip update if frame is identical to last (saves SPI bandwidth/CPU)
            try:
                import hashlib as _hashlib
                current_hash = _hashlib.md5(frame.tobytes()).hexdigest()
                if self._last_frame_hash == current_hash:
                    return
                self._last_frame_hash = current_hash
            except Exception:
                pass

            if self.luma_device is not None:
                # luma device rendering
                try:
                    self.luma_device.display(frame)
                except Exception:
                    # Use canvas fallback
                    with luma_canvas(self.luma_device) as draw:
                        draw.bitmap((0, 0), frame, fill=1)
                return

            if WAVESHARE_AVAILABLE and isinstance(self.display, WS_OLED_1in51.OLED_1in51):
                # Waveshare driver expects rotated buffer for 1.51" panel
                frame_ws = frame.rotate(180)
                self.display.ShowImage(self.display.getbuffer(frame_ws))
            else:
                # displayio path
                bitmap = displayio.Bitmap(self.width, self.height, 2)
                for y in range(self.height):
                    for x in range(self.width):
                        pixel = frame.getpixel((x, y))
                        bitmap[x, y] = 1 if pixel else 0
                palette = displayio.Palette(2)
                palette[0] = 0x000000
                palette[1] = 0xFFFFFF
                tile_grid = displayio.TileGrid(bitmap, pixel_shader=palette)
                group = displayio.Group()
                group.append(tile_grid)
                self.display.show(group)
            
        except Exception as e:
            logger.error(f"Error updating OLED display: {e}")

class PerformanceMonitor:
    """Monitor system performance and manage resources"""
    
    def __init__(self, max_history=10):
        self.cpu_history = deque(maxlen=max_history)
        self.memory_history = deque(maxlen=max_history)
        self.fps_history = deque(maxlen=max_history)
        self.last_frame_time = time.time()
        # Reduce sampling overhead on Pi Zero W
        self.last_metrics_time = 0.0
        self.min_update_interval = 0.5  # seconds
        self._last_cpu = 0.0
        self._last_mem = 0.0
        try:
            # Prime psutil CPU to enable non-blocking subsequent calls
            _ = psutil.cpu_percent(interval=None)
        except Exception:
            pass
        
    def update_metrics(self):
        """Update performance metrics"""
        now = time.time()
        # CPU and memory sampling throttled
        if (now - self.last_metrics_time) >= self.min_update_interval:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                self._last_cpu = cpu_percent
            except Exception:
                cpu_percent = self._last_cpu
            try:
                memory = psutil.virtual_memory()
                self._last_mem = memory.percent
            except Exception:
                pass
            self.last_metrics_time = now
        cpu_percent = self._last_cpu
        memory_percent = self._last_mem
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory_percent)
        
        # FPS calculation
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if self.last_frame_time else 0
        self.fps_history.append(fps)
        self.last_frame_time = current_time
        
        return cpu_percent, memory_percent, fps
    
    def get_average_metrics(self):
        """Get average performance metrics"""
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history) if self.cpu_history else 0
        avg_memory = sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        return avg_cpu, avg_memory, avg_fps
    
    def should_reduce_quality(self):
        """Determine if processing quality should be reduced"""
        if len(self.cpu_history) < 3:
            return False
        
        avg_cpu, avg_memory, avg_fps = self.get_average_metrics()
        
        # Reduce quality if system is overloaded
        return avg_cpu > 80 or avg_memory > 85 or avg_fps < 5
    
    def cleanup_memory(self):
        """Force garbage collection to free memory"""
        gc.collect()

class FoodDetector:
    """Main class for food detection and calorie estimation"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the food detector with configuration"""
        self.config = self.load_config(config_path)
        self.groq_client = None
        self.camera = None
        self.picamera2 = None
        self.detection_queue = queue.Queue(maxsize=5)  # Reduced queue size
        self.results_history = deque(maxlen=5)  # Use deque with max length
        
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize OLED display
        self.oled_display = OLEDDisplay(config=self.config.get("oled_display", {}))
        # Show startup splash
        try:
            self.oled_display.show_startup()
            # Brief pause so the splash is visible even if we fail early
            time.sleep(1.5)
        except Exception:
            pass
        
        # Remote inference client (optional)
        self.remote_client = RemoteInferenceClient(
            url=self.config.get("remote_inference_url", ""),
            timeout=self.config.get("remote_timeout", 8.0),
            max_upload_kb=self.config.get("max_upload_size_kb", 512)
        )
        
        # No on-device ML on Pi Zero W
        self.tflite_detector = None
        
        # Performance optimization flags
        self.low_quality_mode = False
        self.frame_skip_counter = 0
        self.last_detection_result = None
        
        # Food detection state management
        self.food_detected = False
        self.last_food_detection_time = 0
        self.food_detection_cooldown = 1.0  # Minimum time between food detections
        self.llm_processing = False
        self.camera_unavailable = False
        
        # Initialize camera (Pi Camera preferred)
        self.init_camera()
        
        # Load food detection models/classifiers
        self.init_detection_models()
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            "ml_enabled": False,
            "remote_inference_url": os.getenv("REMOTE_INFERENCE_URL", ""),
            "remote_timeout": 8.0,
            "max_upload_size_kb": 512,
            "camera_index": 0,
            "camera_width": 320,  # Reduced for Pi Zero W
            "camera_height": 240,  # Reduced for Pi Zero W
            "detection_confidence": 0.5,
            "detection_interval": 3.0,  # Increased interval
            "save_images": False,  # Disabled by default to save storage
            "output_dir": "detections",
            "preprocessing": {
                "blur_kernel": 3,  # Reduced kernel size
                "brightness_adjustment": 1.1,  # Reduced adjustment
                "contrast_adjustment": 1.05   # Reduced adjustment
            },
            "performance": {
                "max_fps": 10,  # Limit FPS for Pi Zero W
                "frame_skip": 2,  # Skip frames when overloaded
                "low_quality_threshold": 75,  # CPU threshold for quality reduction
                "memory_cleanup_interval": 30  # Seconds between memory cleanup
            },
            "oled_display": {
                "width": 128,
                "height": 64,
                "update_interval": 1.0,  # Update display every second
                "show_system_info": True,
                # SPI-centric defaults for Raspberry Pi (transparent OLEDs are commonly SSD1309/SSD1306 over SPI)
                "rotate": 0,
                "invert": False,
                "spi": {
                    "bus": 0,            # SPI bus (0 for /dev/spidev0.*)
                    "device": 0,         # Chip select (0 => CE0, 1 => CE1)
                    "baudrate": 8000000, # 8MHz (lower to 4000000 if unstable)
                    "dc_pin_bcm": 25,    # Data/Command pin (BCM numbering)
                    "rst_pin_bcm": 24,   # Reset pin (BCM numbering)
                    "driver": "ssd1309"  # "ssd1309" (Waveshare 1.51") or "ssd1306"
                }
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
        except Exception as e:
            logger.warning(f"Could not load config file: {e}. Using defaults.")
            
        return default_config
    
	# Groq client initialization removed; use remote HTTP if configured
    
    def init_camera(self):
        """Initialize camera for Raspberry Pi (prefer Pi Camera module)"""
        try:
            # Try Pi Camera first (more efficient on Pi Zero W)
            if PI_CAMERA_AVAILABLE:
                try:
                    self.picamera2 = Picamera2()
                    
                    # Configure for low-resource usage
                    config = self.picamera2.create_preview_configuration(
                        main={"size": (self.config["camera_width"], self.config["camera_height"])},
                        buffer_count=2  # Minimal buffer
                    )
                    self.picamera2.configure(config)
                    self.picamera2.start()
                    
                    # Allow camera to warm up
                    time.sleep(2)
                    
                    logger.info("Pi Camera module initialized successfully")
                    return
                    
                except Exception as e:
                    logger.warning(f"Pi Camera initialization failed: {e}")
            # Fallback: try USB camera via OpenCV if available
            if OPENCV_AVAILABLE:
                try:
                    cam_index = int(self.config.get("camera_index", 0))
                    cam = cv2.VideoCapture(cam_index)
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera_width"])
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera_height"])
                    ok, _ = cam.read()
                    if ok:
                        self.camera = cam
                        logger.info("USB camera initialized via OpenCV fallback")
                        return
                    cam.release()
                except Exception as e:
                    logger.warning(f"OpenCV USB camera fallback failed: {e}")
            # Show on OLED and mark unavailable
            try:
                if self.oled_display and self.oled_display.display:
                    self.oled_display.show_message(["Camera not found", "Check cable &", "raspi-config"])
                    # Give user a moment to read the message
                    time.sleep(2)
            except Exception:
                pass
            self.camera_unavailable = True
            return
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            self.camera_unavailable = True
            return
    
    def init_detection_models(self):
        """Initialize computer vision models for food detection"""
        try:
            # Initialize background subtractor for motion detection (if OpenCV available)
            self.bg_subtractor = None
            if OPENCV_AVAILABLE:
                try:
                    self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                        detectShadows=True, varThreshold=50
                    )
                except Exception as e:
                    logger.debug(f"Background subtractor unavailable: {e}")
            
            # Initialize contour detection parameters
            self.contour_params = {
                'min_area': 1000,
                'max_area': 50000,
                'min_aspect_ratio': 0.2,
                'max_aspect_ratio': 5.0
            }
            
            # Initialize color detection ranges for common foods
            self.color_ranges = {
                'red_foods': {
                    'lower': np.array([0, 50, 50]),
                    'upper': np.array([10, 255, 255])
                },
                'green_foods': {
                    'lower': np.array([40, 50, 50]),
                    'upper': np.array([80, 255, 255])
                },
                'yellow_foods': {
                    'lower': np.array([20, 50, 50]),
                    'upper': np.array([30, 255, 255])
                },
                'brown_foods': {
                    'lower': np.array([10, 50, 20]),
                    'upper': np.array([20, 255, 200])
                }
            }
            
            logger.info("Detection models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize detection models: {e}")
            raise
    
    def capture_frame(self):
        """Capture frame from camera (Pi Camera or USB) and return as PIL Image"""
        try:
            if self.picamera2:
                # Use Pi Camera - returns RGB array
                frame_array = self.picamera2.capture_array()
                # Convert to PIL Image (RGB format)
                frame = Image.fromarray(frame_array)
                return True, frame
            elif self.camera is not None and OPENCV_AVAILABLE:
                ret, frame_bgr = self.camera.read()
                if not ret or frame_bgr is None:
                    return False, None
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                return True, Image.fromarray(frame_rgb)
            else:
                return False, None
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return False, None
    
	# LLM worker removed; remote inference is handled inline
    
    def is_food_detected(self, detected_objects: List[Dict]) -> bool:
        """Enhanced food detection logic to determine if actual food is present"""
        if not detected_objects:
            return False
        
        # Check if we have significant food-like objects
        food_indicators = 0
        total_confidence = 0
        
        for obj in detected_objects:
            confidence = obj.get('confidence', 0)
            area = obj.get('area', 0)
            aspect_ratio = obj.get('aspect_ratio', 1)
            
            # Food-like characteristics scoring
            score = 0
            
            # Size scoring (food items are usually medium-sized)
            if 2000 <= area <= 25000:
                score += 0.3
            elif 1000 <= area <= 35000:
                score += 0.2
            
            # Aspect ratio scoring (food items are usually not too elongated)
            if 0.3 <= aspect_ratio <= 3.0:
                score += 0.2
            elif 0.2 <= aspect_ratio <= 5.0:
                score += 0.1
            
            # Confidence scoring
            score += confidence * 0.5
            
            if score >= 0.4:  # Threshold for food-like object
                food_indicators += 1
                total_confidence += confidence
        
        # Determine if food is detected based on multiple factors
        avg_confidence = total_confidence / len(detected_objects) if detected_objects else 0
        
        # Food detected if:
        # - At least one strong food indicator, OR
        # - Multiple weaker indicators with good average confidence
        return (food_indicators >= 1 and avg_confidence >= 0.3) or \
               (food_indicators >= 2 and avg_confidence >= 0.2)
    
    def preprocess_image(self, frame):
        """Preprocess image for better food detection using lightweight methods"""
        try:
            # Ensure we have a PIL Image
            if isinstance(frame, np.ndarray):
                frame = LightweightCV.numpy_to_pil(frame)
            
            # Apply brightness and contrast adjustments using PIL (much faster)
            frame = LightweightCV.enhance_image(
                frame,
                brightness=self.config["preprocessing"]["brightness_adjustment"],
                contrast=self.config["preprocessing"]["contrast_adjustment"]
            )
            
            # Apply Gaussian blur to reduce noise using PIL
            blur_radius = self.config["preprocessing"]["blur_kernel"] / 2
            frame = LightweightCV.gaussian_blur(frame, radius=blur_radius)
            
            return frame
        except Exception as e:
            logger.error(f"Error in image preprocessing: {e}")
            return frame
    
    def detect_food_objects(self, frame) -> List[Dict]:
        """Detect potential food objects using lightweight computer vision"""
        detected_objects = []
        
        try:
            # Ensure we have a PIL Image
            if isinstance(frame, np.ndarray):
                frame = LightweightCV.numpy_to_pil(frame)
            
            # Traditional computer vision (color mask + simple bbox heuristic)
            logger.debug("Using traditional CV for food detection")
            
            # Create color mask using lightweight method
            color_mask = LightweightCV.color_mask_pil(frame, self.color_ranges)
            
            # Apply morphological operations to clean up the mask
            cleaned_mask = LightweightCV.morphological_operations(
                color_mask, operation='close', kernel_size=5
            )
            
            # Find contours using lightweight method
            contours = LightweightCV.find_contours_pil(
                cleaned_mask, min_area=self.contour_params['min_area']
            )
            
            for contour in contours:
                area = contour['area']
                x, y, w, h = contour['bbox']
                aspect_ratio = w / h
                
                if (area <= self.contour_params['max_area'] and
                    self.contour_params['min_aspect_ratio'] <= aspect_ratio <= 
                    self.contour_params['max_aspect_ratio']):
                    
                    # Extract region of interest from original frame
                    roi = frame.crop((x, y, x + w, y + h))
                    
                    detected_objects.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'roi': roi,
                        'confidence': min(area / 10000, 1.0),  # Simple confidence based on size
                        'detection_method': 'traditional_cv'
                    })
            
            # Sort by confidence (area-based)
            detected_objects.sort(key=lambda x: x['confidence'], reverse=True)
            
            return detected_objects[:5]  # Return top 5 detections
            
        except Exception as e:
            logger.error(f"Error in food object detection: {e}")
            return []
    
	# Groq SDK removed; remote inference is handled by RemoteInferenceClient
    
    def save_detection_result(self, frame: np.ndarray, result: Dict):
        """Save detection result and image if configured"""
        try:
            if not self.config["save_images"]:
                return
            
            # Create output directory if it doesn't exist
            output_dir = self.config["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save image
            image_path = os.path.join(output_dir, f"detection_{timestamp}.jpg")
            try:
                # Ensure PIL Image
                if isinstance(frame, Image.Image):
                    frame.save(image_path, format="JPEG", quality=85)
                else:
                    # If numpy array, convert safely
                    img = Image.fromarray(frame)
                    img.save(image_path, format="JPEG", quality=85)
            except Exception as e:
                logger.warning(f"Failed to save image via PIL: {e}")
            
            # Save result JSON
            result_path = os.path.join(output_dir, f"result_{timestamp}.json")
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Saved detection result: {result_path}")
            
        except Exception as e:
            logger.error(f"Error saving detection result: {e}")
    
    def draw_detections(self, frame: np.ndarray, detected_objects: List[Dict], groq_result: Dict) -> np.ndarray:
        """Draw detection results on frame"""
        try:
            if not OPENCV_AVAILABLE:
                return frame
            output_frame = frame.copy()
            
            # Draw bounding boxes for detected objects
            for i, obj in enumerate(detected_objects):
                x, y, w, h = obj['bbox']
                confidence = obj['confidence']
                
                # Draw bounding box
                color = (0, 255, 0) if confidence > 0.5 else (0, 255, 255)
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw confidence
                cv2.putText(
                    output_frame, f"Obj {i+1}: {confidence:.2f}",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )
            
            # Draw Groq analysis results
            if "foods_detected" in groq_result:
                y_offset = 30
                for food in groq_result["foods_detected"]:
                    text = f"{food['name']}: {food['calories']} cal"
                    cv2.putText(
                        output_frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                    )
                    y_offset += 25
                
                # Draw total calories
                if "total_calories" in groq_result:
                    total_text = f"Total: {groq_result['total_calories']} calories"
                    cv2.putText(
                        output_frame, total_text, (10, y_offset + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                    )
            
            return output_frame
            
        except Exception as e:
            logger.error(f"Error drawing detections: {e}")
            return frame
    
    def run_detection_loop(self):
        """Main detection loop optimized for Pi Zero W"""
        logger.info("Starting optimized food detection loop...")
        
        try:
            from hashlib import md5
            last_display_update = 0
            last_memory_cleanup = 0
            frame_count = 0
            last_scene_sig = None
            last_camera_retry = 0
            # Cap FPS per config
            target_fps = max(1, int(self.config.get("performance", {}).get("max_fps", 10)))
            frame_period = 1.0 / float(target_fps)
            
            while True:
                loop_start = time.time()
                current_time = loop_start
                
                # Update performance metrics
                cpu_usage, memory_usage, fps = self.performance_monitor.update_metrics()
                
                # If camera is unavailable, retry periodically and keep OLED message visible
                if (self.picamera2 is None and (not OPENCV_AVAILABLE or self.camera is None)):
                    if current_time - last_camera_retry > 5:
                        logger.warning("Camera unavailable; retrying initialization...")
                        self.init_camera()
                        last_camera_retry = current_time
                    try:
                        if self.oled_display and self.oled_display.display:
                            self.oled_display.show_message(["Camera not found", "Check cable &", "raspi-config"])
                    except Exception:
                        pass
                    time.sleep(1.0)
                    continue
                
                # Capture frame
                ret, frame = self.capture_frame()
                if not ret or frame is None:
                    logger.error("Failed to capture frame")
                    time.sleep(0.5)
                    continue
                
                frame_count += 1
                
                # Check if we should reduce quality due to performance
                if self.performance_monitor.should_reduce_quality():
                    if not self.low_quality_mode:
                        logger.info("Enabling low quality mode due to performance")
                        self.low_quality_mode = True
                else:
                    if self.low_quality_mode:
                        logger.info("Disabling low quality mode")
                        self.low_quality_mode = False
                
                # Skip frames if in low quality mode
                if self.low_quality_mode:
                    self.frame_skip_counter += 1
                    if self.frame_skip_counter < self.config["performance"]["frame_skip"]:
                        continue
                    self.frame_skip_counter = 0
                
                # Preprocess frame (reduced processing in low quality mode)
                if self.low_quality_mode:
                    # Minimal preprocessing using PIL
                    processed_frame = LightweightCV.gaussian_blur(frame, radius=1)
                else:
                    processed_frame = self.preprocess_image(frame)
                
                # Always run object detection (lightweight)
                detected_objects = self.detect_food_objects(processed_frame)
                
                # Check if actual food is detected
                current_food_detected = self.is_food_detected(detected_objects)
                
                # Update food detection state
                if current_food_detected:
                    if not self.food_detected:
                        # Food just appeared
                        logger.info(f"🍽️ FOOD DETECTED! Found {len(detected_objects)} potential food objects")
                        self.food_detected = True
                        self.last_food_detection_time = current_time
                        
                        # Inline remote analysis on scene change (debounce)
                        try:
                            top_roi = detected_objects[0]['roi'].resize((64, 64)).convert('L')
                            sig = md5(top_roi.tobytes()).hexdigest()
                            if last_scene_sig != sig:
                                last_scene_sig = sig
                                self.llm_processing = True
                                analysis = self.remote_client.analyze(frame, detected_objects)
                                analysis["detection_timestamp"] = datetime.now().isoformat()
                                self.last_detection_result = analysis
                                self.results_history.append({
                                    "timestamp": analysis.get("detection_timestamp"),
                                    "detected_objects": len(detected_objects),
                                    "result": analysis
                                })
                                self.print_results(analysis)
                                self.llm_processing = False
                        except Exception as e:
                            logger.debug(f"Remote analysis skipped: {e}")
                    
                    # Update detection time for continuous food presence
                    self.last_food_detection_time = current_time
                
                else:
                    # Check if food disappeared (with cooldown to avoid flickering)
                    if self.food_detected and (current_time - self.last_food_detection_time) > self.food_detection_cooldown:
                        logger.info("🚫 Food no longer detected")
                        self.food_detected = False
                        # Clear display after food disappears
                        self.last_detection_result = {"foods_detected": [], "message": "No food detected"}
                
                # Update OLED display
                display_update_interval = self.config["oled_display"]["update_interval"]
                if current_time - last_display_update >= display_update_interval:
                    
                    # Always show only food name and calories; blank if none
                    display_img = self.oled_display.create_food_display(
                        self.last_detection_result or {},
                        food_detected=self.food_detected,
                        llm_processing=self.llm_processing
                    )
                    self.oled_display.update_display(display_img)
                    
                    last_display_update = current_time
                
                # Periodic memory cleanup
                cleanup_interval = self.config["performance"]["memory_cleanup_interval"]
                if current_time - last_memory_cleanup >= cleanup_interval:
                    self.performance_monitor.cleanup_memory()
                    last_memory_cleanup = current_time
                    logger.debug("Performed memory cleanup")
                
                # Display frame with detections (only if DISPLAY is available and not in low quality mode)
                if os.getenv("DISPLAY") and not self.low_quality_mode and OPENCV_AVAILABLE:
                    # Convert PIL image to OpenCV format for display
                    if isinstance(frame, Image.Image):
                        frame_array = np.array(frame)
                        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                    
                    cv2.imshow("V.A.R.G Food Detection", frame_bgr)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Adaptive delay based on performance
                elapsed = time.time() - loop_start
                # In low quality mode, we can afford slightly lower FPS target
                effective_period = frame_period * (1.5 if self.low_quality_mode else 1.0)
                sleep_left = max(0.0, effective_period - elapsed)
                if sleep_left > 0:
                    time.sleep(sleep_left)
                
        except KeyboardInterrupt:
            logger.info("Detection loop interrupted by user")
        except Exception as e:
            logger.error(f"Error in detection loop: {e}")
        finally:
            self.cleanup()
    
    def print_results(self, result: Dict):
        """Print detection results to console"""
        print("\n" + "="*60)
        print("🍽️  V.A.R.G FOOD DETECTION & CALORIE ANALYSIS")
        print("="*60)
        
        # Show detection timestamp
        if "detection_timestamp" in result:
            print(f"⏰ Detection Time: {result['detection_timestamp']}")
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        if "foods_detected" in result and result["foods_detected"]:
            print("🔍 Identified Foods:")
            print("-" * 40)
            
            for i, food in enumerate(result["foods_detected"], 1):
                print(f"  {i}. 🥘 {food['name']}")
                print(f"     📏 Portion: {food['portion_size']}")
                print(f"     🔥 Calories: {food['calories']} kcal")
                print(f"     ✅ Confidence: {food['confidence']}/10")
                print(f"     📝 Notes: {food['description']}")
                print()
            
            if "total_calories" in result:
                print("=" * 40)
                print(f"🔥 TOTAL ESTIMATED CALORIES: {result['total_calories']} kcal")
                print("=" * 40)
        else:
            print("🤷 No food items clearly identified in this analysis")
        
        if "analysis_notes" in result:
            print(f"📋 Additional Notes: {result['analysis_notes']}")
        
        print("="*60)
    
    def get_detection_history(self) -> List[Dict]:
        """Get detection history"""
        return self.results_history
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # No background threads to stop
            
            # Clean up camera resources
            if self.picamera2:
                self.picamera2.stop()
                self.picamera2.close()
            if self.camera:
                self.camera.release()
            
            # Clean up display
            if OPENCV_AVAILABLE:
                cv2.destroyAllWindows()
            
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def create_default_config():
    """Create default configuration file"""
    config = {
        "ml_enabled": False,
        "remote_inference_url": "",
        "remote_timeout": 8.0,
        "max_upload_size_kb": 512,
        "camera_index": 0,
        "camera_width": 320,
        "camera_height": 240,
        "detection_confidence": 0.5,
        "detection_interval": 3.0,
        "save_images": False,
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
            "show_system_info": True,
            "rotate": 0,
            "invert": False,
            "spi": {
                "bus": 0,
                "device": 0,
                "baudrate": 8000000,
                "dc_pin_bcm": 25,
                "rst_pin_bcm": 24,
                "driver": "ssd1309"
            }
        }
    }
    
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Created default config.json file")
    print("Optionally set remote_inference_url in config.json or .env (REMOTE_INFERENCE_URL)")

def main():
    """Main function"""
    print("🤖 V.A.R.G - Visual Automated Recipe & Grocery")
    print("Food Detection and Calorie Estimation System")
    print("Optimized for Raspberry Pi Zero W")
    print("-" * 50)
    
    # Check if config exists
    if not os.path.exists("config.json"):
        print("Config file not found. Creating default configuration...")
        create_default_config()
        return
    
    try:
        # Initialize detector
        detector = FoodDetector()
        
        # Run detection loop
        detector.run_detection_loop()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    main()
