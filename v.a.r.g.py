import sys
import os
import json
from pathlib import Path
import gc
import threading
import logging
import time
import traceback
import base64
import io
import re
import concurrent.futures
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Configure GPIOZero to use RPi.GPIO backend by default (avoids some lgpio 'GPIO busy' issues)
os.environ.setdefault("GPIOZERO_PIN_FACTORY", "rpigpio")

# Add library paths for Waveshare OLED
current_dir = os.path.dirname(os.path.realpath(__file__))

# Support both 'RaspberryPi' and 'Raspberry' layouts for the Waveshare lib
lib_candidates = [
    os.path.join(current_dir, 'RaspberryPi', 'python', 'lib'),
    os.path.join(current_dir, 'Raspberry', 'python', 'lib'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'),
]
for libdir in lib_candidates:
    if os.path.exists(libdir) and libdir not in sys.path:
        sys.path.append(libdir)

# Picture/font directory candidates (for Font.ttc, bitmaps, etc.)
pic_candidates = [
    os.path.join(current_dir, 'RaspberryPi', 'python', 'pic'),
    os.path.join(current_dir, 'Raspberry', 'python', 'pic'),
    os.path.join(current_dir, 'pic'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'pic'),
]
picdir = None
for candidate in pic_candidates:
    if os.path.exists(candidate):
        picdir = candidate
        break
if picdir is None:
    # Fallback to current directory to avoid crashing; fonts may not be found
    picdir = current_dir


def try_release_gpio_pins(pins):
    """
    Best-effort attempt to free GPIO pins that might be left busy by previous runs.
    This is mainly to reduce 'GPIO busy' errors from gpiozero/lgpio when re-running the script.
    """
    # Legacy sysfs unexport (no-op on newer kernels, but harmless)
    for pin in pins:
        try:
            with open("/sys/class/gpio/unexport", "w") as f:
                f.write(str(pin))
        except Exception:
            # Ignore if sysfs interface is not available or pin not exported
            pass

# Waveshare OLED import
try:
    from waveshare_OLED import OLED_1in51
except ImportError:
    logging.warning("waveshare_OLED library not found. Display will not work.")
    OLED_1in51 = None

# Try to import requests for Groq API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("Requests library not available. Groq API will not work.")

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Optimize numpy for Pi Zero W (single core)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# TensorFlow Lite support has been removed in this build to simplify deployment
TFLITE_AVAILABLE = False
TFLITE_RUNTIME = False

# Try to import camera and CV libraries
# For Raspberry Pi Zero W we now use OpenCV directly instead of Picamera2
cv2 = None  # Initialize cv2 as None, will be set if available
try:
    import cv2
    CAMERA_AVAILABLE = True
    logging.warning("Using OpenCV camera backend (Pi Zero W).")
except ImportError:
    cv2 = None
    CAMERA_AVAILABLE = False
    logging.warning("OpenCV not available. Camera will be disabled and mock detection used.")

# Reduce logging for performance on Pi Zero W
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# Food calorie database (simplified lookup)
FOOD_CALORIES = {
    'apple': 95, 'banana': 105, 'orange': 62, 'strawberry': 4,
    'bread': 79, 'pizza': 266, 'burger': 354, 'sandwich': 250,
    'rice': 130, 'pasta': 131, 'chicken': 231, 'beef': 250,
    'salad': 20, 'soup': 100, 'cake': 235, 'cookie': 50,
    'coffee': 2, 'tea': 2, 'milk': 103, 'yogurt': 100
}

class FoodDetector:
    def __init__(self, config_path='config.json'):
        self.config = self.load_config(config_path)
        self.camera = None
        self.detected_food = None
        self.calories = 0
        self.detection_active = True
        self.tflite_interpreter = None
        self.tflite_labels = []
        self.model_input_size = (224, 224)  # Default input size for most food models
        
        # Synchronization primitives
        self.lock = threading.Lock()
        self.new_data_event = threading.Event()
        
        # Performance optimization: cache tensor details
        self.input_details = None
        self.output_details = None
        self.input_index = None
        self.output_index = None
        
        # Frame skipping for Pi Zero W (more aggressive)
        self.frame_skip = self.config.get('performance', {}).get('frame_skip', 5)
        self.frame_count = 0
        
        # Memory optimization: reuse buffers
        self.last_frame = None
        
        # Groq API configuration
        self.groq_api_key = None
        self.groq_model = self.config.get('calorie_estimation_model', 'llama-3.2-11b-vision-preview')
        # Validate model is a vision model
        if 'vision' not in self.groq_model.lower() and 'llama3-70b' not in self.groq_model.lower():
            # Check if it's a known non-vision model
            if 'llama3-70b-8192' in self.groq_model.lower():
                logging.warning(f"Model '{self.groq_model}' may not support vision. Consider using 'llama-3.2-11b-vision-preview' for image analysis.")
        self.groq_enabled = False
        self.init_groq()
        
        # Performance optimization: result caching and smoothing
        self.last_result = None
        self.last_result_time = 0
        self.result_cache_duration = 5.0  # Cache results for 5 seconds
        self.groq_call_interval = 10.0  # Only call Groq every 10 seconds max
        self.last_groq_call = 0
        self.pending_groq_result = None  # For async processing
        
        # Frame difference detection (only process if scene changed)
        self.last_processed_frame_hash = None
        
        # Thread pool for async Groq calls
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning(f"Config file {config_path} not found. Using defaults.")
            return {
                'camera_index': 0,
                'camera_width': 224,  # Smaller for Pi Zero W
                'camera_height': 224,
                'detection_interval': 5.0,  # Longer interval for Pi Zero W
                'groq_api_key': '',
                'calorie_estimation_model': 'llama-3.2-11b-vision-preview',
                'tflite': {
                    'enabled': True,
                    'confidence_threshold': 0.35,
                    'model_priority': ['efficientnet_food', 'mobilenet_food_v2', 'food101_mobilenet']
                },
                'performance': {
                    'frame_skip': 5,  # Process every 5th frame (more aggressive for Pi Zero W)
                    'memory_cleanup_interval': 20  # More frequent cleanup
                },
                'oled_display': {
                    'update_interval': 2.0  # Update less frequently
                }
            }
    
    def init_groq(self):
        """Initialize Groq API client"""
        if not REQUESTS_AVAILABLE:
            logging.warning("Requests library not available. Groq API disabled.")
            return
        
        # Try to get API key from config or environment
        self.groq_api_key = self.config.get('groq_api_key', '')
        if not self.groq_api_key:
            self.groq_api_key = os.getenv('GROQ_API_KEY', '')
        
        if self.groq_api_key:
            self.groq_enabled = True
            logging.warning("Groq API initialized")
        else:
            logging.warning("Groq API key not found. Groq features disabled.")
    
    def image_to_base64(self, image):
        """Convert PIL Image to base64 string for API - optimized for Pi Zero W"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to smaller size for Pi Zero W (faster API calls, less data)
            max_size = 256  # Reduced from 512 for faster processing
            if image.size[0] > max_size or image.size[1] > max_size:
                # Use NEAREST for speed on Pi Zero W
                image = image.resize((max_size, max_size), Image.Resampling.NEAREST)
            
            # Convert to bytes with lower quality for speed
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=70, optimize=True)  # Lower quality = faster
            img_bytes = buffer.getvalue()
            
            # Encode to base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return img_base64
        except Exception as e:
            logging.error(f"Error converting image to base64: {e}")
            return None
    
    def query_groq_llm(self, image):
        """Query Groq LLM with image for food detection and calorie estimation"""
        if not self.groq_enabled or not REQUESTS_AVAILABLE:
            return None, None
        
        try:
            # Convert image to base64
            img_base64 = self.image_to_base64(image)
            if not img_base64:
                return None, None
            
            # Prepare API request
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            }

            # Build prompt text separately to avoid complex escaping
            prompt_text = (
                "Analyze this food image. Identify the food item(s) and estimate the total calories. "
                "Respond in JSON format: "
                '{"food": "food name", "calories": number}. '
                "Be specific about the food item and provide an accurate calorie estimate based on typical serving size."
            )

            # Use vision model for image analysis
            payload = {
                "model": self.groq_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_text,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.3,
            }
            
            # Make API request with shorter timeout for Pi Zero W (faster failure)
            response = requests.post(url, json=payload, headers=headers, timeout=8)
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                
                # Try to extract JSON from response
                try:
                    # Look for JSON in the response
                    json_match = re.search(r'\{[^}]+\}', content)
                    if json_match:
                        food_data = json.loads(json_match.group())
                        food_name = food_data.get('food', '')
                        calories = food_data.get('calories', 0)
                        return food_name, calories
                    else:
                        # Fallback: try to extract food name and calories from text
                        # Look for patterns like "food: X" or "calories: Y"
                        food_match = re.search(r'food["\']?\s*:\s*["\']?([^,"\']+)', content, re.IGNORECASE)
                        cal_match = re.search(r'calories?["\']?\s*:\s*(\d+)', content, re.IGNORECASE)
                        
                        food_name = food_match.group(1).strip() if food_match else ''
                        calories = int(cal_match.group(1)) if cal_match else 0
                        return food_name, calories
                except Exception as e:
                    logging.error(f"Error parsing Groq response: {e}")
                    logging.error(f"Response content: {content}")
                    return None, None
            
            return None, None
            
        except requests.exceptions.Timeout:
            logging.error("Groq API request timed out")
            return None, None
        except requests.exceptions.RequestException as e:
            logging.error(f"Groq API request failed: {e}")
            return None, None
        except Exception as e:
            logging.error(f"Error querying Groq LLM: {e}")
            return None, None
    
    def load_tflite_model(self):
        """Load TensorFlow Lite model and labels"""
        tflite_config = self.config.get('tflite', {})
        # If runtime is not available or disabled in config, skip quietly
        if not TFLITE_AVAILABLE or not tflite_config.get('enabled', True):
            logging.info("TFLite detection disabled in config")
            return False
        
        models_dir = Path('models')
        if not models_dir.exists():
            logging.warning("Models directory not found. Run setup_models.py first.")
            return False
        
        # Try to load models in priority order
        model_priority = tflite_config.get('model_priority', 
            ['efficientnet_food', 'mobilenet_food_v2', 'food101_mobilenet'])
        
        for model_name in model_priority:
            model_path = models_dir / f"{model_name}.tflite"
            labels_path = models_dir / f"{model_name}_labels.txt"
            
            if model_path.exists():
                try:
                    # Load interpreter
                    self.tflite_interpreter = tflite.Interpreter(model_path=str(model_path))
                    self.tflite_interpreter.allocate_tensors()
                    
                    # Get input and output details (cache for performance)
                    self.input_details = self.tflite_interpreter.get_input_details()
                    self.output_details = self.tflite_interpreter.get_output_details()
                    self.input_index = self.input_details[0]['index']
                    self.output_index = self.output_details[0]['index']
                    
                    # Get input size from model
                    input_shape = self.input_details[0]['shape']
                    if len(input_shape) >= 3:
                        self.model_input_size = (input_shape[1], input_shape[2])
                    
                    # Load labels
                    if labels_path.exists():
                        with open(labels_path, 'r') as f:
                            self.tflite_labels = [line.strip() for line in f.readlines()]
                    else:
                        logging.warning(f"Labels file not found: {labels_path}")
                        # Use Food-101 labels as fallback
                        self.tflite_labels = self._get_food101_labels()
                    
                    logging.info(f"Loaded TFLite model: {model_name}")
                    logging.info(f"Input size: {self.model_input_size}")
                    logging.info(f"Labels count: {len(self.tflite_labels)}")
                    return True
                    
                except Exception as e:
                    logging.error(f"Failed to load model {model_name}: {e}")
                    continue
        
        logging.warning("No TFLite models found or loaded")
        return False
    
    def _get_food101_labels(self):
        """Get Food-101 labels as fallback"""
        return [
            "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
            "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
            "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
            "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
            "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
            "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
            "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
            "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
            "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
            "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
            "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
            "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
            "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
            "mussels", "nachos", "omelette", "onion_rings", "oysters",
            "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
            "pho", "pizza", "pork_chop", "poutine", "prime_rib",
            "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
            "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
            "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
            "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
            "waffles"
        ]
    
    def init_camera(self):
        """Initialize camera with optimized OpenCV settings for Pi Zero W.
        
        We now use OpenCV directly instead of Picamera2 to avoid libcamera
        pipeline conflicts and simplify deployment on Pi Zero W.
        """
        if not CAMERA_AVAILABLE:
            return False
        
        # Use smaller resolution for Pi Zero W
        cam_width = self.config.get('camera_width', 224)
        cam_height = self.config.get('camera_height', 224)

        # OpenCV backend (works on Pi Zero W; may use more CPU but avoids libcamera pipeline issues)
        try:
            self.camera = cv2.VideoCapture(self.config.get('camera_index', 0))
            # Set lower resolution and frame rate for Pi Zero W
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
            self.camera.set(cv2.CAP_PROP_FPS, 5)  # Very low FPS for Pi Zero W
            if self.camera.isOpened():
                logging.warning("Camera initialized (OpenCV backend) on Pi Zero W")
                return True
        except Exception as e2:
            logging.error(f"Failed to initialize camera via OpenCV: {e2}")
            return False
        
        return False
    
    def capture_frame(self):
        """Capture a frame from camera"""
        if not self.camera:
            return None
            
        try:
            # OpenCV capture
            ret, frame = self.camera.read()
            if ret:
                return frame
        except Exception as e:
            logging.error(f"Error capturing frame: {e}")
        return None
    
    def preprocess_frame(self, frame):
        """Preprocess frame for TFLite model input - optimized for Pi Zero W"""
        try:
            # Optimize: if frame is already close to model size, skip resize
            if isinstance(frame, np.ndarray):
                # Handle different color formats
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Convert BGR to RGB if needed (more efficient than PIL conversion)
                    try:
                        if cv2 is not None and hasattr(cv2, 'COLOR_BGR2RGB'):
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        else:
                            frame_rgb = frame
                    except Exception:
                        frame_rgb = frame
                else:
                    frame_rgb = frame
                
                # Use cv2 resize if available (faster than PIL)
                if cv2 is not None and hasattr(cv2, 'resize') and frame_rgb.shape[:2] != self.model_input_size:
                    frame_rgb = cv2.resize(frame_rgb, self.model_input_size, 
                                           interpolation=cv2.INTER_LINEAR)
                    pil_image = Image.fromarray(frame_rgb)
                else:
                    pil_image = Image.fromarray(frame_rgb)
                    if pil_image.size != self.model_input_size:
                        # Use NEAREST for speed on Pi Zero W (lower quality but faster)
                        pil_image = pil_image.resize(self.model_input_size, Image.Resampling.NEAREST)
            else:
                pil_image = frame
                if pil_image.size != self.model_input_size:
                    pil_image = pil_image.resize(self.model_input_size, Image.Resampling.NEAREST)
            
            # Convert to numpy array - use uint8 first, then convert to float
            # This is more memory efficient
            img_array = np.array(pil_image, dtype=np.uint8)
            
            # Normalize in-place to save memory
            img_array = img_array.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logging.error(f"Error preprocessing frame: {e}")
            return None
    
    def detect_food_tflite(self, frame):
        """Detect food using TFLite model - optimized for Pi Zero W"""
        if frame is None or self.tflite_interpreter is None:
            return None, 0.0
        
        try:
            # Preprocess frame
            input_data = self.preprocess_frame(frame)
            if input_data is None:
                return None, 0.0
            
            # Use cached tensor indices for performance
            # Set input tensor
            self.tflite_interpreter.set_tensor(self.input_index, input_data)
            
            # Run inference
            self.tflite_interpreter.invoke()
            
            # Get output (reuse output_details)
            output_data = self.tflite_interpreter.get_tensor(self.output_index)
            predictions = output_data[0]  # Remove batch dimension
            
            # Get top prediction (use argmax directly - more efficient)
            confidence_threshold = self.config.get('tflite', {}).get('confidence_threshold', 0.35)
            top_idx = int(np.argmax(predictions))
            confidence = float(predictions[top_idx])
            
            if confidence >= confidence_threshold and top_idx < len(self.tflite_labels):
                food_name = self.tflite_labels[top_idx]
                # Clean up food name (remove underscores, format nicely)
                food_name = food_name.replace('_', ' ').title()
                return food_name, confidence
            else:
                return None, confidence
                
        except Exception as e:
            logging.error(f"Error in TFLite detection: {e}")
            return None, 0.0
        finally:
            # Memory cleanup
            if 'input_data' in locals():
                del input_data
            gc.collect()
    
    def frame_hash(self, frame):
        """Quick hash of frame to detect changes (optimized for Pi Zero W)"""
        try:
            # Use downsampled frame for hash (much faster)
            if isinstance(frame, np.ndarray):
                # Downsample to 32x32 for hash
                if cv2 is not None and hasattr(cv2, 'resize'):
                    small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_NEAREST)
                else:
                    # Fallback: simple downsampling without cv2
                    h, w = frame.shape[:2]
                    step_h, step_w = max(1, h // 32), max(1, w // 32)
                    small = frame[::step_h, ::step_w]
                return hash(small.tobytes())
            return hash(str(frame))
        except Exception:
            return None
    
    def detect_food(self, frame):
        """Main food detection: uses Groq LLM (when enabled) with caching, no TensorFlow Lite."""
        if frame is None:
            return None, 0
        
        # Frame skipping for Pi Zero W
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return None, 0
        
        # Check if frame changed significantly (skip processing if same)
        frame_hash = self.frame_hash(frame)
        current_time = time.time()
        if (
            frame_hash == self.last_processed_frame_hash
            and self.last_result
            and current_time - self.last_result_time < self.result_cache_duration
        ):
            return self.last_result

        final_food = None
        final_calories = 0

        # Use Groq LLM for detection if enabled
        if self.groq_enabled:
            time_since_groq = current_time - self.last_groq_call
            should_call_groq = time_since_groq >= self.groq_call_interval

            if should_call_groq:
                # Convert frame to PIL only when needed (expensive operation)
                try:
                    if isinstance(frame, np.ndarray):
                        if len(frame.shape) == 3:
                            if cv2 is not None and hasattr(cv2, 'COLOR_BGR2RGB'):
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            else:
                                frame_rgb = frame
                            pil_image = Image.fromarray(frame_rgb)
                        else:
                            pil_image = Image.fromarray(frame)
                    else:
                        pil_image = frame
                    
                    # Submit to thread pool instead of blocking
                    self.executor.submit(self.process_groq_async, pil_image)
                    self.last_groq_call = current_time
                    logging.info("Submitted Groq task to background thread")
                except Exception as e:
                    logging.error(f"Error preparing frame for Groq: {e}")

            # If async worker has already produced a result, use it
            if (
                self.last_result
                and current_time - self.last_result_time < self.result_cache_duration
            ):
                final_food, final_calories = self.last_result

        # Cache frame hash if we have a valid detection
        if final_food:
            self.last_processed_frame_hash = frame_hash
            return final_food, final_calories

        return None, 0

    def process_groq_async(self, image):
        """Process Groq API call asynchronously"""
        try:
            food, calories = self.query_groq_llm(image)
            if food:
                logging.warning(f"🤖 Groq LLM Analysis (Async): {food} ({calories} cal)")
                
                # Update shared state safely
                with self.lock:
                    self.detected_food = food
                    self.calories = calories
                
                # Update cache
                self.last_result = (food, calories)
                self.last_result_time = time.time()
                
                # Signal new data available
                self.new_data_event.set()
        except Exception as e:
            logging.error(f"Error in async Groq processing: {e}")
    
    def estimate_calories(self, food_name):
        """Estimate calories for detected food"""
        if not food_name:
            return 0
        
        food_lower = food_name.lower().replace('_', ' ').replace('-', ' ')
        
        # Try exact match first
        if food_lower in FOOD_CALORIES:
            return FOOD_CALORIES[food_lower]
        
        # Try partial match - check if any key is in the food name or vice versa
        for food, cal in FOOD_CALORIES.items():
            food_clean = food.replace('_', ' ').replace('-', ' ')
            if food_clean in food_lower or food_lower in food_clean:
                return cal
        
        # Try matching individual words
        food_words = food_lower.split()
        for word in food_words:
            if word in FOOD_CALORIES:
                return FOOD_CALORIES[word]
            for food, cal in FOOD_CALORIES.items():
                if word in food or food in word:
                    return cal
        
        # Default calorie estimate based on food type keywords
        if any(word in food_lower for word in ['pizza', 'burger', 'sandwich', 'taco']):
            return 250
        elif any(word in food_lower for word in ['salad', 'soup']):
            return 100
        elif any(word in food_lower for word in ['cake', 'dessert', 'sweet']):
            return 200
        elif any(word in food_lower for word in ['fruit', 'apple', 'banana', 'orange']):
            return 80
        elif any(word in food_lower for word in ['meat', 'chicken', 'beef', 'steak']):
            return 200
        
        return 150  # Default estimate
    
    def run_detection_loop(self):
        """Main detection loop - optimized for Pi Zero W"""
        # TensorFlow Lite support has been removed; we rely on Groq LLM (when enabled)
        # and heuristic calorie estimation instead.
        
        if not self.init_camera():
            logging.warning("Camera not available. Using mock detection.")
            # Mock mode for testing
            self.detected_food = "apple"
            self.calories = 95
            return
        
        detection_interval = self.config.get('detection_interval', 5.0)
        memory_cleanup_interval = self.config.get('performance', {}).get('memory_cleanup_interval', 20)
        last_cleanup = time.time()
        
        # Optimize: Skip initial frames to let camera stabilize
        warmup_frames = 3
        warmup_count = 0
        
        while self.detection_active:
            try:
                frame = self.capture_frame()
                
                # Warmup period - skip processing
                if warmup_count < warmup_frames:
                    warmup_count += 1
                    if frame is not None:
                        del frame
                    time.sleep(0.5)
                    continue
                
                if frame is not None:
                    # Detect food using optimized pipeline (TFLite first, Groq when needed)
                    food, calories = self.detect_food(frame)
                    if food and calories > 0:
                        # Thread-safe update
                        with self.lock:
                            self.detected_food = food
                            self.calories = calories
                        # Signal new data
                        self.new_data_event.set()
                        logging.warning(f"Final: {food} - {calories} cal")
                    
                    # Immediate memory cleanup
                    del frame
                    # Force garbage collection more aggressively on Pi Zero W
                    if self.frame_count % 10 == 0:
                        gc.collect()
                else:
                    # Mock detection if camera fails
                    with self.lock:
                        self.detected_food = "apple"
                        self.calories = 95
                    self.new_data_event.set()
                
                # Periodic memory cleanup for Pi Zero W
                current_time = time.time()
                if current_time - last_cleanup > memory_cleanup_interval:
                    gc.collect()
                    last_cleanup = current_time
                
                time.sleep(detection_interval)
            except Exception as e:
                logging.error(f"Error in detection loop: {e}")
                time.sleep(detection_interval)
    
    def stop(self):
        """Stop detection"""
        self.detection_active = False
        if self.executor:
            self.executor.shutdown(wait=False)
        if self.camera:
            # OpenCV VideoCapture cleanup
            try:
                if hasattr(self.camera, "release"):
                    self.camera.release()
            except Exception:
                pass

def draw_vertical_text(image, text, x, y, font, fill=0):
    """Draw text vertically (rotated 90 degrees) - optimized for Pi Zero W"""
    try:
        # Use smaller temp image to save memory
        temp_img = Image.new('1', (100, 100), 255)
        temp_draw = ImageDraw.Draw(temp_img)
        temp_draw.text((0, 0), text, font=font, fill=fill)
        
        # Rotate 90 degrees counter-clockwise
        rotated = temp_img.rotate(90, expand=True)
        
        # Get bounding box of non-white pixels
        bbox = rotated.getbbox()
        if bbox:
            cropped = rotated.crop(bbox)
            cw, ch = cropped.size
            
            # Calculate paste position
            paste_y = max(0, min(y, image.height - ch))
            
            # Paste the rotated text
            image.paste(cropped, (x, paste_y))
    except Exception:
        pass  # Fail silently to avoid display errors


def draw_centered_text(image, text, font, fill=0):
    """
    Draw horizontal text centered on the given image.
    Centers both horizontally and vertically.
    """
    try:
        draw = ImageDraw.Draw(image)
        # textbbox gives accurate metrics on Pillow 8+
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (image.width - text_width) // 2
        y = (image.height - text_height) // 2
        draw.text((x, y), text, font=font, fill=fill)
    except Exception:
        # Fail silently to avoid crashing the display loop
        pass


def draw_centered_text_at_y(image, text, font, y, fill=0):
    """
    Draw horizontal text centered at a specific vertical position (y).
    Useful for multi-line layouts on the OLED.
    """
    try:
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        x = (image.width - text_width) // 2
        draw.text((x, y), text, font=font, fill=fill)
    except Exception:
        pass

def main():
    disp = None
    detector = None
    detection_thread = None
    
    try:
        # Try to release GPIO pins that the OLED typically uses (e.g., RST on GPIO27)
        # This helps if a previous process left them exported/busy.
        try_release_gpio_pins([27, 25, 8, 10, 11])

        # Initialize OLED display using Waveshare library
        if OLED_1in51:
            logging.info("Initializing 1.51inch OLED (Waveshare)...")
            try:
                disp = OLED_1in51.OLED_1in51()
                disp.Init()
                disp.clear()
            except Exception as e:
                # Handle GPIO or wiring issues gracefully instead of crashing
                logging.error(f"Failed to initialize Waveshare OLED display: {e}")
                traceback.print_exc()
                disp = None
        else:
            logging.error("Waveshare OLED library not loaded")
        
        # Initialize food detector
        detector = FoodDetector()
        
        # Start detection in background thread
        detection_thread = threading.Thread(target=detector.run_detection_loop, daemon=True)
        detection_thread.start()
        
        # Load fonts
        font_path = os.path.join(picdir, 'Font.ttc')
        if not os.path.exists(font_path):
            # Fallback to default font
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
        else:
            font_large = ImageFont.truetype(font_path, 20)
            font_medium = ImageFont.truetype(font_path, 16)
        
        # Read OLED display settings from config (for orientation)
        oled_config = detector.config.get("oled_display", {})
        oled_rotation = int(oled_config.get("rotation", 180))  # degrees: 0, 90, 180, 270

        # Main display loop - optimized for Pi Zero W
        # Reuse image buffer to reduce memory allocation
        # Use display dimensions when available, otherwise default to 128x64
        oled_width = getattr(disp, "width", 128) if disp else 128
        oled_height = getattr(disp, "height", 64) if disp else 64
        image = Image.new('1', (oled_width, oled_height), 255)
        last_displayed_food = None
        last_displayed_calories = 0

        # Initial splash screen: show "V.A.R.G" centered on the OLED
        if disp:
            image.paste(255, (0, 0, oled_width, oled_height))  # Clear to white
            draw_centered_text(image, "V.A.R.G", font_large, fill=0)
            rotated_image = image.rotate(oled_rotation)
            disp.ShowImage(disp.getbuffer(rotated_image))
        
        while True:
            # Wait for new data or timeout (1.0s)
            # This ensures we update immediately when data is ready (sync fix)
            # but also refresh occasionally if needed
            detector.new_data_event.wait(timeout=1.0)
            detector.new_data_event.clear()
            
            # Read state safely
            current_food = None
            current_calories = 0
            with detector.lock:
                current_food = detector.detected_food
                current_calories = detector.calories
            
            # Only update display if data changed (avoid unnecessary redraws)
            data_changed = (current_food != last_displayed_food or 
                          current_calories != last_displayed_calories)
            
            if data_changed and disp:
                # Clear image (reuse buffer)
                image.paste(255, (0, 0, oled_width, oled_height))  # Fill with white
                
                # Display food name and calories in a clean horizontal layout
                if current_food and current_calories > 0:
                    logging.info(f"Displaying: {current_food} - {current_calories} cal")
                    # Food name: uppercase, truncated to fit
                    food_text = current_food.upper()
                    if len(food_text) > 12:
                        food_text = food_text[:12]

                    # Calories: integer + "CAL"
                    cal_int = int(current_calories)
                    cal_text = f"{cal_int} CAL"

                    # Two-line centered layout
                    top_y = max(0, oled_height // 6)
                    bottom_y = max(0, oled_height // 2)
                    draw_centered_text_at_y(image, food_text, font_medium, top_y, fill=0)
                    draw_centered_text_at_y(image, cal_text, font_large, bottom_y, fill=0)
                    
                    last_displayed_food = current_food
                    last_displayed_calories = current_calories
                else:
                    # Waiting for detection: show a simple centered status message
                    status_text = "SCAN"
                    draw_centered_text(image, status_text, font_medium, fill=0)
                
                # Rotate image for proper orientation on OLED
                rotated_image = image.rotate(oled_rotation)
                
                # Display on OLED
                disp.ShowImage(disp.getbuffer(rotated_image))
            
    except IOError as e:
        logging.error(f"IO Error: {e}")
        traceback.print_exc()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error(f"Error: {e}")
        traceback.print_exc()
    finally:
        if detector:
            detector.stop()
            # Give detection thread a moment to clean up
            time.sleep(0.5)
        if disp:
            try:
                disp.clear()
                disp.module_exit()
            except Exception:
                pass
        logging.info("Exiting...")

if __name__ == "__main__":
    main()