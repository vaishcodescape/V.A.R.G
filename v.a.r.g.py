#!/usr/bin/env python3
"""
V.A.R.G - Visual Automated Recipe & Grocery
Food Detection and Calorie Estimation System for Raspberry Pi Zero W
Using OpenCV, Computer Vision, Groq LLM Integration, and Transparent OLED Display
Optimized for Pi Camera Module and minimal resource usage
"""

import numpy as np
import json
import time
import logging
from datetime import datetime
from typing import Dict, List
import os
from groq import Groq
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import queue
import gc
import threading
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

# Try to import lightweight computer vision libraries
try:
    from skimage import measure, morphology, color
    from skimage.feature import canny
    from skimage.util import img_as_ubyte
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logging.warning("scikit-image not available, using PIL-based processing")

# Try to import TensorFlow Lite for food detection models
try:
    # Try TensorFlow Lite runtime first (smaller package)
    import tflite_runtime.interpreter as tflite
    tf = None  # We'll use tflite directly
    TFLITE_AVAILABLE = True
    TFLITE_RUNTIME_ONLY = True
    logging.info("TensorFlow Lite runtime available for food detection models")
except ImportError:
    try:
        # Fallback to full TensorFlow
        import tensorflow as tf
        tflite = None
        TFLITE_AVAILABLE = True
        TFLITE_RUNTIME_ONLY = False
        logging.info("TensorFlow (full) available for food detection models")
    except ImportError:
        tf = None
        tflite = None
        TFLITE_AVAILABLE = False
        TFLITE_RUNTIME_ONLY = False
        logging.warning("TensorFlow Lite not available, using traditional CV methods")

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
    import adafruit_displayio_ssd1306
    import displayio
    OLED_AVAILABLE = True
except ImportError:
    OLED_AVAILABLE = False
    logging.warning("OLED display libraries not available")

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
    """Lightweight computer vision operations using PIL and scikit-image"""
    
    @staticmethod
    def pil_to_numpy(pil_image):
        """Convert PIL image to numpy array"""
        return np.array(pil_image)
    
    @staticmethod
    def numpy_to_pil(np_array):
        """Convert numpy array to PIL image"""
        if np_array.dtype != np.uint8:
            # Safely convert to 8-bit without requiring scikit-image
            try:
                if SKIMAGE_AVAILABLE:
                    from skimage.util import img_as_ubyte as _img_as_ubyte
                    np_array = _img_as_ubyte(np_array)
                else:
                    if 'float' in str(np_array.dtype):
                        np_array = (np.clip(np_array, 0.0, 1.0) * 255.0).astype(np.uint8)
                    else:
                        np_array = np.clip(np_array, 0, 255).astype(np.uint8)
            except Exception:
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
    def find_edges_skimage(image, sigma=1.0, low_threshold=0.1, high_threshold=0.2):
        """Edge detection using scikit-image (if available)"""
        if not SKIMAGE_AVAILABLE:
            return LightweightCV.find_edges_pil(image)
        
        if isinstance(image, Image.Image):
            image = LightweightCV.pil_to_numpy(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = color.rgb2gray(image)
        
        # Apply Canny edge detection
        edges = canny(image, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
        
        return (edges * 255).astype(np.uint8)
    
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
        
        # Simple contour detection using connected components
        if SKIMAGE_AVAILABLE:
            # Use scikit-image for better contour detection
            labeled = measure.label(mask_array > 128)
            regions = measure.regionprops(labeled)
            
            contours = []
            for region in regions:
                if region.area >= min_area:
                    # Get bounding box
                    minr, minc, maxr, maxc = region.bbox
                    contours.append({
                        'bbox': (minc, minr, maxc - minc, maxr - minr),
                        'area': region.area,
                        'centroid': region.centroid
                    })
            
            return contours
        else:
            # Fallback: create a single bounding box over non-zero area
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
        """Morphological operations using scikit-image or PIL"""
        if not SKIMAGE_AVAILABLE:
            # Simple dilation/erosion using PIL
            if isinstance(image, np.ndarray):
                image = LightweightCV.numpy_to_pil(image)
            
            if operation == 'close':
                # Dilation followed by erosion
                for _ in range(kernel_size // 2):
                    image = image.filter(ImageFilter.MaxFilter(3))
                for _ in range(kernel_size // 2):
                    image = image.filter(ImageFilter.MinFilter(3))
            
            return image
        
        # Use scikit-image for better morphological operations
        if isinstance(image, Image.Image):
            image = LightweightCV.pil_to_numpy(image)
        
        kernel = morphology.disk(kernel_size // 2)
        
        if operation == 'close':
            result = morphology.closing(image > 128, kernel)
        elif operation == 'open':
            result = morphology.opening(image > 128, kernel)
        else:
            result = image
        
        return (result * 255).astype(np.uint8)

class TFLiteFoodDetector:
    """TensorFlow Lite-based food detection and classification"""
    
    def __init__(self, model_path=None):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_shape = None
        self.food_labels = []
        self.model_loaded = False
        
        # Default model paths
        self.default_models = {
            'food_classification': 'models/food_classifier.tflite',
            'food_detection': 'models/food_detector.tflite',
            'mobilenet_food': 'models/mobilenet_food_v2.tflite'
        }
        
        # Food categories with calorie estimates (per 100g)
        self.food_categories = {
            'fruits': {
                'apple': 52, 'banana': 89, 'orange': 47, 'grapes': 62, 'strawberry': 32,
                'pineapple': 50, 'mango': 60, 'watermelon': 30, 'peach': 39, 'pear': 57
            },
            'vegetables': {
                'carrot': 41, 'broccoli': 34, 'spinach': 23, 'tomato': 18, 'cucumber': 16,
                'lettuce': 15, 'onion': 40, 'potato': 77, 'bell_pepper': 31, 'corn': 86
            },
            'grains': {
                'rice': 130, 'bread': 265, 'pasta': 131, 'oats': 389, 'quinoa': 120,
                'wheat': 327, 'barley': 354, 'noodles': 138
            },
            'proteins': {
                'chicken': 165, 'beef': 250, 'fish': 206, 'eggs': 155, 'tofu': 76,
                'beans': 127, 'lentils': 116, 'nuts': 607, 'cheese': 402
            },
            'dairy': {
                'milk': 42, 'yogurt': 59, 'cheese': 402, 'butter': 717, 'cream': 345
            }
        }
        
        if TFLITE_AVAILABLE:
            self.load_model(model_path)
    
    def load_model(self, model_path=None):
        """Load TensorFlow Lite model for food detection"""
        try:
            # Try to load specified model or default models
            models_to_try = []

            # Support KaggleHub model path: "kagglehub:<model_id>"
            if model_path and isinstance(model_path, str) and model_path.startswith("kagglehub"):
                try:
                    parts = model_path.split(":", 1)
                    kh_id = parts[1] if len(parts) > 1 else "google/aiy/tfLite/vision-classifier-food-v1"
                    import importlib
                    kagglehub = importlib.import_module("kagglehub")
                    download_dir = kagglehub.model_download(kh_id)
                    # Find first .tflite file in download dir
                    chosen = None
                    for root, _, files in os.walk(download_dir):
                        for f in files:
                            if f.lower().endswith(".tflite"):
                                chosen = os.path.join(root, f)
                                break
                        if chosen:
                            break
                    if chosen and os.path.exists(chosen):
                        os.makedirs('models', exist_ok=True)
                        dest_name = os.path.basename(chosen)
                        dest_path = os.path.join('models', dest_name)
                        try:
                            if chosen != dest_path:
                                import shutil
                                shutil.copy2(chosen, dest_path)
                            models_to_try.append(dest_path)
                            logger.info(f"Using KaggleHub model: {dest_path}")
                        except Exception:
                            models_to_try.append(chosen)
                            logger.info(f"Using KaggleHub model in-place: {chosen}")
                    else:
                        logger.warning("KaggleHub download succeeded but no .tflite file found")
                except Exception as e:
                    logger.warning(f"KaggleHub model resolution failed: {e}")

            if model_path and os.path.exists(model_path):
                models_to_try.append(model_path)
            
            # Add default models
            for model_name, path in self.default_models.items():
                if os.path.exists(path):
                    models_to_try.append(path)
            
            if not models_to_try:
                logger.warning("No TFLite models found. Creating directory structure...")
                os.makedirs('models', exist_ok=True)
                self._download_default_models()
                return
            
            # Load the first available model
            model_path = models_to_try[0]
            logger.info(f"Loading TFLite model: {model_path}")
            
            # Load TFLite model and allocate tensors
            if TFLITE_RUNTIME_ONLY:
                self.interpreter = tflite.Interpreter(model_path=model_path)
            else:
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output tensors
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get input shape
            self.input_shape = self.input_details[0]['shape']
            logger.info(f"Model input shape: {self.input_shape}")
            
            # Load food labels if available
            self._load_food_labels(model_path)
            
            self.model_loaded = True
            logger.info("TFLite food detection model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load TFLite model: {e}")
            self.model_loaded = False
    
    def _load_food_labels(self, model_path):
        """Load food labels for the model"""
        try:
            # Try to load labels file
            labels_path = model_path.replace('.tflite', '_labels.txt')
            if os.path.exists(labels_path):
                with open(labels_path, 'r') as f:
                    self.food_labels = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {len(self.food_labels)} food labels")
            else:
                # Use default food categories
                self.food_labels = []
                for category, foods in self.food_categories.items():
                    self.food_labels.extend(foods.keys())
                logger.info(f"Using default food labels: {len(self.food_labels)} items")
                
        except Exception as e:
            logger.error(f"Error loading food labels: {e}")
            self.food_labels = list(self.food_categories.keys())
    
    def _download_default_models(self):
        """Download default food detection models"""
        logger.info("Setting up default food detection models...")
        
        # Create a simple food classification model info file
        model_info = {
            "models_needed": [
                {
                    "name": "MobileNet Food Classifier",
                    "url": "https://tfhub.dev/google/lite-model/aiy/vision/classifier/food_V1/1",
                    "description": "Pre-trained food classification model"
                }
            ],
            "setup_instructions": [
                "1. Download pre-trained food models from TensorFlow Hub",
                "2. Place .tflite files in the 'models' directory",
                "3. Ensure corresponding _labels.txt files are present",
                "4. Restart the application"
            ]
        }
        
        with open('models/README.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info("Created models directory and setup instructions")
    
    def preprocess_image(self, image):
        """Preprocess image for TFLite model input"""
        try:
            if not self.model_loaded:
                return None
            
            # Ensure we have a PIL Image
            if isinstance(image, np.ndarray):
                image = LightweightCV.numpy_to_pil(image)
            
            # Get target size from model input shape
            if len(self.input_shape) == 4:  # Batch, Height, Width, Channels
                target_height = self.input_shape[1]
                target_width = self.input_shape[2]
            else:
                target_height, target_width = 224, 224  # Default size
            
            # Resize image
            image = image.resize((target_width, target_height), RESAMPLE_LANCZOS)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32)
            
            # Normalize to [0, 1] or [-1, 1] depending on model
            image_array = image_array / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image for TFLite: {e}")
            return None
    
    def detect_food(self, image):
        """Detect and classify food using TFLite model"""
        try:
            if not self.model_loaded:
                return []
            
            # Preprocess image
            input_data = self.preprocess_image(image)
            if input_data is None:
                return []
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process results
            results = self._process_detection_results(output_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in TFLite food detection: {e}")
            return []
    
    def _process_detection_results(self, output_data):
        """Process TFLite model output into food detection results"""
        try:
            results = []
            
            # Handle different output formats
            if len(output_data.shape) == 2:  # Classification output
                predictions = output_data[0]
                
                # Get top predictions
                top_indices = np.argsort(predictions)[-5:][::-1]  # Top 5
                
                for idx in top_indices:
                    confidence = float(predictions[idx])
                    
                    if confidence > 0.1:  # Minimum confidence threshold
                        # Get food name
                        if idx < len(self.food_labels):
                            food_name = self.food_labels[idx]
                        else:
                            food_name = f"food_class_{idx}"
                        
                        # Estimate calories
                        calories = self._estimate_calories(food_name, confidence)
                        
                        results.append({
                            'name': food_name,
                            'confidence': confidence,
                            'calories_per_100g': calories,
                            'detection_method': 'tflite_classification'
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing TFLite results: {e}")
            return []
    
    def _estimate_calories(self, food_name, confidence):
        """Estimate calories for detected food"""
        try:
            food_name_lower = food_name.lower()
            
            # Search in food categories
            for category, foods in self.food_categories.items():
                for food, calories in foods.items():
                    if food in food_name_lower or food_name_lower in food:
                        return calories
            
            # Default calorie estimates based on food type keywords
            if any(keyword in food_name_lower for keyword in ['fruit', 'apple', 'banana', 'orange']):
                return 50  # Average fruit calories
            elif any(keyword in food_name_lower for keyword in ['vegetable', 'carrot', 'broccoli']):
                return 35  # Average vegetable calories
            elif any(keyword in food_name_lower for keyword in ['meat', 'chicken', 'beef']):
                return 200  # Average meat calories
            elif any(keyword in food_name_lower for keyword in ['bread', 'rice', 'pasta']):
                return 150  # Average carb calories
            else:
                return 100  # Default estimate
                
        except Exception as e:
            logger.error(f"Error estimating calories: {e}")
            return 100

class OLEDDisplay:
    """Handles transparent OLED display for showing detection results"""
    
    def __init__(self, width=128, height=64):
        self.width = width
        self.height = height
        self.display = None
        self.font = None
        self.small_font = None
        
        if OLED_AVAILABLE:
            self.init_display()
        else:
            logger.warning("OLED display not available")
    
    def init_display(self):
        """Initialize the OLED display"""
        try:
            displayio.release_displays()
            
            # Initialize I2C
            i2c = board.I2C()
            # Try common SSD1306 I2C addresses (0x3C then 0x3D)
            try:
                display_bus = displayio.I2CDisplay(i2c, device_address=0x3C)
            except Exception:
                display_bus = displayio.I2CDisplay(i2c, device_address=0x3D)
            
            # Initialize display
            self.display = adafruit_displayio_ssd1306.SSD1306(
                display_bus, width=self.width, height=self.height
            )
            
            # Load fonts
            try:
                self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
                self.small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
            except (OSError, IOError):
                self.font = ImageFont.load_default()
                self.small_font = ImageFont.load_default()
            
            logger.info("OLED display initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OLED display: {e}")
            self.display = None
    
    def create_food_display(self, foods_data: Dict, food_detected: bool = False, llm_processing: bool = False) -> Image.Image:
        """Create display image for food detection results"""
        img = Image.new('1', (self.width, self.height), 0)  # 1-bit image for OLED
        draw = ImageDraw.Draw(img)
        
        # Show different states based on detection status
        if not food_detected:
            # Show "Scanning" when no food detected
            draw.text((25, 20), "Scanning", font=self.font, fill=1)
            draw.text((15, 35), "for Food...", font=self.small_font, fill=1)
            return img
        
        if llm_processing:
            # Show "Analyzing" when LLM is processing
            draw.text((15, 15), "Analyzing", font=self.font, fill=1)
            draw.text((25, 30), "Food...", font=self.small_font, fill=1)
            # Add a simple progress indicator
            for i in range(3):
                x = 30 + i * 20
                draw.ellipse([(x, 45), (x + 5, 50)], fill=1)
            return img
        
        if not foods_data or "foods_detected" not in foods_data:
            # Show "Processing" message
            draw.text((20, 20), "Processing", font=self.font, fill=1)
            draw.text((25, 35), "Food...", font=self.small_font, fill=1)
            return img
        
        foods = foods_data["foods_detected"]
        if not foods:
            if "message" in foods_data:
                draw.text((10, 20), "No Food", font=self.font, fill=1)
                draw.text((10, 35), "Detected", font=self.font, fill=1)
            else:
                draw.text((20, 20), "Processing", font=self.font, fill=1)
                draw.text((25, 35), "Food...", font=self.small_font, fill=1)
            return img
        
        y_pos = 4
        
        # Show first food item (most confident)
        food = foods[0]
        name = str(food.get("name", "Food"))[:14]  # Truncate long names to fit
        calories = int(food.get("calories", 0))
        
        # Determine harm level from calories (simple heuristic)
        # LOW: <=150 kcal, MED: 151-300 kcal, HIGH: >300 kcal
        if calories <= 150:
            harm = "LOW"
        elif calories <= 300:
            harm = "MED"
        else:
            harm = "HIGH"
        
        # Vertical layout: Name, Calories, Harm level
        draw.text((4, y_pos), name, font=self.font, fill=1)
        y_pos += 18
        draw.text((4, y_pos), f"Cal: {calories} kcal", font=self.small_font, fill=1)
        y_pos += 12
        draw.text((4, y_pos), f"Harm: {harm}", font=self.small_font, fill=1)
        
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
        if not self.display:
            return
        
        try:
            # Convert PIL image to displayio bitmap
            bitmap = displayio.Bitmap(self.width, self.height, 2)
            
            for y in range(self.height):
                for x in range(self.width):
                    pixel = image.getpixel((x, y))
                    bitmap[x, y] = 1 if pixel else 0
            
            # Create palette and tile grid
            palette = displayio.Palette(2)
            palette[0] = 0x000000  # Black
            palette[1] = 0xADD8E6  # Light Blue for active pixels (monochrome displays will still show white)
            
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
        
    def update_metrics(self):
        """Update performance metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_history.append(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_history.append(memory.percent)
        
        # FPS calculation
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if self.last_frame_time else 0
        self.fps_history.append(fps)
        self.last_frame_time = current_time
        
        return cpu_percent, memory.percent, fps
    
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
        self.oled_display = OLEDDisplay()
        
        # Initialize TensorFlow Lite food detector
        tflite_cfg = self.config.get("tflite", {}) if isinstance(self.config.get("tflite", {}), dict) else {}
        model_path_cfg = tflite_cfg.get("model_path")
        self.tflite_detector = TFLiteFoodDetector(model_path=model_path_cfg) if TFLITE_AVAILABLE else None
        
        # Performance optimization flags
        self.low_quality_mode = False
        self.frame_skip_counter = 0
        self.last_detection_result = None
        
        # Food detection state management
        self.food_detected = False
        self.last_food_detection_time = 0
        self.food_detection_cooldown = 1.0  # Minimum time between food detections
        self.llm_processing = False
        self.llm_queue = queue.Queue(maxsize=3)  # Queue for LLM processing
        
        # Start LLM processing thread
        self.llm_thread = threading.Thread(target=self._llm_processing_worker, daemon=True)
        self.llm_thread.start()
        
        # Initialize Groq client
        self.init_groq_client()
        
        # Initialize camera (Pi Camera preferred)
        self.init_camera()
        
        # Load food detection models/classifiers
        self.init_detection_models()
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            "groq_api_key": os.getenv("GROQ_API_KEY", ""),
            "camera_index": 0,
            "camera_width": 320,  # Reduced for Pi Zero W
            "camera_height": 240,  # Reduced for Pi Zero W
            "detection_confidence": 0.5,
            # Use a Groq vision-capable model by default
            "calorie_estimation_model": "llama-3.2-11b-vision-preview",
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
                "show_system_info": True
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                # Environment variable should take precedence if provided
                env_key = os.getenv("GROQ_API_KEY")
                if env_key:
                    default_config["groq_api_key"] = env_key
        except Exception as e:
            logger.warning(f"Could not load config file: {e}. Using defaults.")
            
        return default_config
    
    def init_groq_client(self):
        """Initialize Groq client for LLM integration"""
        try:
            # Prefer env var if present
            env_key = os.getenv("GROQ_API_KEY", "")
            if env_key:
                self.config["groq_api_key"] = env_key
            if not self.config["groq_api_key"]:
                logger.warning("Groq API key not set; continuing without LLM integration.")
                self.groq_client = None
                return
            
            self.groq_client = Groq(api_key=self.config["groq_api_key"])
            logger.info("Groq client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            self.groq_client = None
    
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
            raise RuntimeError("No available camera. Enable Picamera2 or attach USB camera with OpenCV available.")
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            raise
    
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
    
    def _llm_processing_worker(self):
        """Background worker for LLM processing"""
        while True:
            try:
                # Get task from queue (blocks until available)
                task = self.llm_queue.get(timeout=1.0)
                
                if task is None:  # Shutdown signal
                    break
                
                frame, detected_objects, timestamp = task
                
                # Set processing flag
                self.llm_processing = True
                
                # Analyze with Groq LLM
                groq_result = self.analyze_food_with_groq(frame, detected_objects)
                groq_result["detection_timestamp"] = timestamp
                
                # Update last detection result
                self.last_detection_result = groq_result
                
                # Save results if enabled
                if self.config["save_images"]:
                    self.save_detection_result(frame, {
                        "detected_objects": len(detected_objects),
                        "groq_analysis": groq_result
                    })
                
                # Add to history
                self.results_history.append({
                    "timestamp": timestamp,
                    "detected_objects": len(detected_objects),
                    "groq_result": groq_result
                })
                
                # Print results
                self.print_results(groq_result)
                
                # Clear processing flag
                self.llm_processing = False
                
                # Mark task as done
                self.llm_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in LLM processing worker: {e}")
                self.llm_processing = False
    
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
        """Detect potential food objects using TFLite or lightweight computer vision"""
        detected_objects = []
        
        try:
            # Ensure we have a PIL Image
            if isinstance(frame, np.ndarray):
                frame = LightweightCV.numpy_to_pil(frame)
            
            # Try TensorFlow Lite detection first (more accurate)
            if self.tflite_detector and self.tflite_detector.model_loaded:
                tflite_results = self.tflite_detector.detect_food(frame)

                # Filter by confidence threshold and limit to max detections
                thr = float(self.config.get("tflite", {}).get("confidence_threshold", 0.35))
                max_det = int(self.config.get("tflite", {}).get("max_detections", 3))
                tflite_results = [r for r in (tflite_results or []) if r.get('confidence', 0.0) >= thr]
                tflite_results.sort(key=lambda r: r.get('confidence', 0.0), reverse=True)
                tflite_results = tflite_results[:max_det]

                if tflite_results:
                    logger.debug(f"TFLite kept {len(tflite_results)} candidates after thresholding")
                    
                    # Convert TFLite results to our format
                    for i, result in enumerate(tflite_results):
                        # Create a bounding box (since TFLite classification doesn't provide one)
                        # Use the whole image or estimate based on center
                        width, height = frame.size
                        bbox_size = min(width, height) // 2
                        x = (width - bbox_size) // 2
                        y = (height - bbox_size) // 2
                        
                        detected_objects.append({
                            'bbox': (x, y, bbox_size, bbox_size),
                            'area': bbox_size * bbox_size,
                            'aspect_ratio': 1.0,
                            'roi': frame.crop((x, y, x + bbox_size, y + bbox_size)),
                            'confidence': result.get('confidence', 0.0),
                            'food_name': result.get('name'),
                            'calories_per_100g': result.get('calories_per_100g'),
                            'detection_method': 'tflite'
                        })
                    
                    return detected_objects[:max_det]
            
            # Fallback to traditional computer vision if TFLite not available or no results
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
    
    def encode_image_for_groq(self, image) -> str:
        """Encode image to base64 for Groq API"""
        try:
            # Ensure we have a PIL Image
            if isinstance(image, np.ndarray):
                pil_image = LightweightCV.numpy_to_pil(image)
            else:
                pil_image = image
            
            # Resize for efficiency (Groq has size limits) - PIL is much faster for this
            pil_image.thumbnail((512, 512), resample=RESAMPLE_LANCZOS)
            
            # Convert to base64
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG", quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return ""
    
    def analyze_food_with_groq(self, image: np.ndarray, detected_objects: List[Dict]) -> Dict:
        """Use Groq LLM to identify food and estimate calories"""
        try:
            if not self.groq_client:
                return {"error": "Groq client not initialized"}
            
            # Encode image for API
            image_data = self.encode_image_for_groq(image)
            if not image_data:
                return {"error": "Failed to encode image"}
            
            # Candidate hints from on-device model
            candidate_hints = []
            for obj in (detected_objects or []):
                if obj.get('detection_method') == 'tflite':
                    name = obj.get('food_name')
                    if name and name not in candidate_hints:
                        candidate_hints.append(name)
            hint_text = ""
            if candidate_hints:
                hint_text = f"Candidate foods (from on-device model): {', '.join(candidate_hints[:3])}. Prefer these if reasonable.\n"

            # Create prompt for food identification and calorie estimation (request strict JSON only)
            prompt = f"""
            {hint_text}
            Analyze this image and identify any food items present. For each food item you identify:
            
            1. Name of the food item
            2. Estimated portion size (small, medium, large, or specific measurements if possible)
            3. Estimated calories for that portion
            4. Confidence level (1-10) in your identification
            5. Brief description of what you see
            
            I detected {len(detected_objects)} potential food objects in the image using computer vision.
            
            Return STRICT JSON ONLY with this schema (no extra commentary):
            {{
                "foods_detected": [
                    {{
                        "name": "food_name",
                        "portion_size": "estimated_size",
                        "calories": estimated_calories_number,
                        "confidence": confidence_score,
                        "description": "what_you_see"
                    }}
                ],
                "total_calories": total_estimated_calories,
                "analysis_notes": "any_additional_observations"
            }}
            
            If no food is clearly visible, return an empty foods_detected array.
            """
            
            # Make API call to Groq
            response = self.groq_client.chat.completions.create(
                model=self.config["calorie_estimation_model"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_data}}
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            # Parse response with robust JSON extraction
            response_text = response.choices[0].message.content
            
            try:
                result = json.loads(response_text)
                result["timestamp"] = datetime.now().isoformat()
                result["detection_method"] = "groq_llm"
                return result
            except json.JSONDecodeError:
                import re
                match = re.search(r"\{[\s\S]*\}$", response_text.strip())
                if match:
                    try:
                        result = json.loads(match.group(0))
                        result["timestamp"] = datetime.now().isoformat()
                        result["detection_method"] = "groq_llm"
                        return result
                    except Exception:
                        pass
                return {
                    "error": "Failed to parse JSON response",
                    "raw_response": response_text,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in Groq analysis: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
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
            last_display_update = 0
            last_memory_cleanup = 0
            frame_count = 0
            
            while True:
                current_time = time.time()
                
                # Update performance metrics
                cpu_usage, memory_usage, fps = self.performance_monitor.update_metrics()
                
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
                        
                        # Queue LLM analysis only when food is detected and scene changed (debounce)
                        if not self.llm_processing and not self.llm_queue.full():
                            try:
                                # Create a simple scene signature from top ROI to avoid redundant calls
                                top_roi = detected_objects[0]['roi'].resize((64, 64)).convert('L')
                                from hashlib import md5
                                sig = md5(top_roi.tobytes()).hexdigest()
                                if getattr(self, '_last_scene_sig', None) != sig:
                                    self._last_scene_sig = sig
                                    self.llm_queue.put_nowait((frame, detected_objects, datetime.now().isoformat()))
                                    logger.info("🤖 Queued for LLM analysis...")
                            except queue.Full:
                                logger.warning("LLM queue is full, skipping analysis")
                    
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
                    
                    if self.config["oled_display"]["show_system_info"] and frame_count % 20 == 0:
                        # Show system info occasionally (less frequent)
                        avg_cpu, avg_memory, avg_fps = self.performance_monitor.get_average_metrics()
                        self.oled_display.show_system_info(avg_cpu, avg_memory, avg_fps)
                    else:
                        # Show food detection status and results
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
                if self.low_quality_mode:
                    time.sleep(0.2)  # Longer delay in low quality mode
                else:
                    time.sleep(0.05)  # Shorter delay for better responsiveness
                
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
            # Shutdown LLM processing thread
            if hasattr(self, 'llm_queue'):
                self.llm_queue.put(None)  # Shutdown signal
                if hasattr(self, 'llm_thread') and self.llm_thread.is_alive():
                    self.llm_thread.join(timeout=2.0)
            
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
        "groq_api_key": "",
        "camera_index": 0,
        "camera_width": 320,
        "camera_height": 240,
        "detection_confidence": 0.5,
        "calorie_estimation_model": "llama-3.2-11b-vision-preview",
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
            "show_system_info": True
        },
        "tflite": {
            "enabled": True,
            "model_path": "kagglehub:google/aiy/tfLite/vision-classifier-food-v1",
            "confidence_threshold": 0.35,
            "max_detections": 3,
            "use_fallback_cv": True
        }
    }
    
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Created default config.json file")
    print("Please edit config.json and add your Groq API key")

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
