import sys
import os
import json
from pathlib import Path
import gc

picdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'pic')
libdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib')
if os.path.exists(libdir):
    sys.path.append(libdir)

import logging
import time
import traceback
import threading
import base64
import io
import re
from waveshare_OLED import OLED_1in51
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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

# Try to import TensorFlow Lite (prefer tflite_runtime for Pi Zero W)
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
    TFLITE_RUNTIME = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
        TFLITE_RUNTIME = False
    except ImportError:
        TFLITE_AVAILABLE = False
        TFLITE_RUNTIME = False
        logging.warning("TensorFlow Lite not available. Food detection will be limited.")

# Try to import camera and CV libraries
# NOTE: For Raspberry Pi Zero W, Picamera2 is MUCH more efficient than OpenCV
# OpenCV works but is resource-heavy. Picamera2 is the recommended choice.
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
    CAMERA_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    try:
        import cv2
        CAMERA_AVAILABLE = True
        logging.warning("Picamera2 not available. Using OpenCV (slower on Pi Zero W)")
    except ImportError:
        CAMERA_AVAILABLE = False
        logging.warning("Camera libraries not available. Using mock detection.")

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
                "Content-Type": "application/json"
            }
            
            # Use vision model for image analysis
            payload = {
                "model": self.groq_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this food image. Identify the food item(s) and estimate the total calories. Respond in JSON format: {\"food\": \"food name\", \"calories\": number}. Be specific about the food item and provide an accurate calorie estimate based on typical serving size."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 150,
                "temperature": 0.3
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
        if not TFLITE_AVAILABLE:
            logging.warning("TensorFlow Lite not available")
            return False
        
        tflite_config = self.config.get('tflite', {})
        if not tflite_config.get('enabled', True):
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
        """Initialize camera with optimized settings for Pi Zero W
        
        Priority:
        1. Picamera2 (BEST for Pi Zero W - native, efficient, hardware-accelerated)
        2. OpenCV (FALLBACK - works but slower and more resource-intensive)
        
        OpenCV on Pi Zero W:
        - Will work but uses more CPU and memory
        - Slower frame rates
        - Higher latency
        - Not recommended for production use on Pi Zero W
        """
        if not CAMERA_AVAILABLE:
            return False
        
        # Use smaller resolution for Pi Zero W
        cam_width = self.config.get('camera_width', 224)
        cam_height = self.config.get('camera_height', 224)
            
        # Try Picamera2 first (RECOMMENDED for Pi Zero W)
        if PICAMERA2_AVAILABLE:
            try:
                self.camera = Picamera2()
                # Use lower quality mode for Pi Zero W
                config = self.camera.create_preview_configuration(
                    main={"size": (cam_width, cam_height)},
                    buffer_count=1  # Reduce buffer for memory
                )
                self.camera.configure(config)
                self.camera.start()
                logging.warning("Camera initialized (Picamera2) - OPTIMAL for Pi Zero W")
                return True
            except Exception as e:
                logging.warning(f"Picamera2 failed: {e}. Trying OpenCV fallback...")
        
        # Fallback to OpenCV (works but not optimal for Pi Zero W)
        try:
            import cv2
            self.camera = cv2.VideoCapture(self.config.get('camera_index', 0))
            # Set lower resolution and frame rate for Pi Zero W
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
            self.camera.set(cv2.CAP_PROP_FPS, 5)  # Very low FPS for Pi Zero W
            if self.camera.isOpened():
                logging.warning("Camera initialized (OpenCV) - FALLBACK mode (slower on Pi Zero W)")
                return True
        except Exception as e2:
            logging.error(f"Failed to initialize camera: {e2}")
            return False
        
        return False
    
    def capture_frame(self):
        """Capture a frame from camera"""
        if not self.camera:
            return None
            
        try:
            if hasattr(self.camera, 'capture_array'):
                # Picamera2
                frame = self.camera.capture_array()
                return frame
            else:
                # OpenCV
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
                        if hasattr(cv2, 'COLOR_BGR2RGB'):
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        else:
                            frame_rgb = frame
                    except Exception:
                        frame_rgb = frame
                else:
                    frame_rgb = frame
                
                # Use cv2 resize if available (faster than PIL)
                if hasattr(cv2, 'resize') and frame_rgb.shape[:2] != self.model_input_size:
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
                small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_NEAREST) if hasattr(cv2, 'resize') else frame[::8, ::8]
                return hash(small.tobytes())
            return hash(str(frame))
        except Exception:
            return None
    
    def detect_food(self, frame):
        """Main food detection: Optimized for Pi Zero W - TFLite first, Groq only when needed"""
        if frame is None:
            return None, 0
        
        # Frame skipping for Pi Zero W
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return None, 0
        
        # Check if frame changed significantly (skip processing if same)
        frame_hash = self.frame_hash(frame)
        if frame_hash == self.last_processed_frame_hash:
            # Return cached result if available
            current_time = time.time()
            if (self.last_result and 
                current_time - self.last_result_time < self.result_cache_duration):
                return self.last_result
        
        # Step 1: Use TFLite FIRST (fast, local) - only call Groq if TFLite finds food
        tflite_food = None
        tflite_confidence = 0.0
        
        if self.tflite_interpreter is not None:
            tflite_food, tflite_confidence = self.detect_food_tflite(frame)
            if tflite_food:
                logging.warning(f"TFLite: {tflite_food} (conf: {tflite_confidence:.2f})")
        
        # Step 2: Only call Groq if:
        # - TFLite detected food (to get accurate calories)
        # - AND enough time has passed since last Groq call
        # - OR TFLite didn't detect anything but we haven't called Groq in a while
        groq_food = None
        groq_calories = 0
        current_time = time.time()
        should_call_groq = False
        
        if self.groq_enabled:
            time_since_groq = current_time - self.last_groq_call
            if tflite_food and time_since_groq >= self.groq_call_interval:
                # TFLite found food, get accurate calories from Groq
                should_call_groq = True
            elif not tflite_food and time_since_groq >= (self.groq_call_interval * 2):
                # No TFLite detection, but check with Groq occasionally
                should_call_groq = True
            
            if should_call_groq:
                # Convert frame to PIL only when needed (expensive operation)
                try:
                    if isinstance(frame, np.ndarray):
                        if len(frame.shape) == 3:
                            if hasattr(cv2, 'COLOR_BGR2RGB'):
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            else:
                                frame_rgb = frame
                            pil_image = Image.fromarray(frame_rgb)
                        else:
                            pil_image = Image.fromarray(frame)
                    else:
                        pil_image = frame
                    
                    groq_food, groq_calories = self.query_groq_llm(pil_image)
                    self.last_groq_call = current_time
                    if groq_food:
                        logging.warning(f"Groq: {groq_food} ({groq_calories} cal)")
                except Exception as e:
                    logging.error(f"Error preparing frame for Groq: {e}")
        
        # Step 3: Combine results intelligently
        final_food = None
        final_calories = 0
        
        if groq_food and groq_calories > 0:
            # Groq result available - use it
            final_food = groq_food
            final_calories = groq_calories
            
            # Validate with TFLite if available
            if tflite_food:
                groq_lower = groq_food.lower()
                tflite_lower = tflite_food.lower()
                # Simple similarity check
                if not (groq_lower in tflite_lower or tflite_lower in groq_lower or
                       any(w in groq_lower for w in tflite_lower.split())):
                    # Don't match - if TFLite confidence is high, average the calories
                    if tflite_confidence > 0.7:
                        tflite_cal = self.estimate_calories(tflite_food)
                        final_calories = (groq_calories + tflite_cal) // 2
        elif tflite_food:
            # Use TFLite result with estimated calories
            final_food = tflite_food
            final_calories = self.estimate_calories(tflite_food)
        elif groq_food:
            # Groq detected but no calories
            final_food = groq_food
            final_calories = self.estimate_calories(groq_food)
        
        # Cache result
        if final_food:
            self.last_result = (final_food, final_calories)
            self.last_result_time = current_time
            self.last_processed_frame_hash = frame_hash
        
        return final_food, final_calories
    
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
        # Load TFLite model first
        if not self.load_tflite_model():
            logging.warning("TFLite model not loaded. Detection will be limited.")
        
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
                        self.detected_food = food
                        self.calories = calories
                        logging.warning(f"Final: {food} - {calories} cal")
                    
                    # Immediate memory cleanup
                    del frame
                    # Force garbage collection more aggressively on Pi Zero W
                    if self.frame_count % 10 == 0:
                        gc.collect()
                else:
                    # Mock detection if camera fails
                    self.detected_food = "apple"
                    self.calories = 95
                
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
        if self.camera:
            try:
                if hasattr(self.camera, 'stop'):
                    self.camera.stop()
                else:
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

def main():
    disp = None
    detector = None
    detection_thread = None
    
    try:
        # Initialize OLED display
        disp = OLED_1in51.OLED_1in51()
        logging.info("Initializing 1.51inch OLED")
        disp.Init()
        disp.clear()
        
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
        
        # Main display loop - optimized for Pi Zero W
        last_update = 0
        update_interval = detector.config.get('oled_display', {}).get('update_interval', 3.0)  # Less frequent updates
        
        # Reuse image buffer to reduce memory allocation
        image = Image.new('1', (64, 128), 255)
        last_displayed_food = None
        last_displayed_calories = 0
        
        while True:
            current_time = time.time()
            
            # Only update display if data changed or interval passed (avoid unnecessary redraws)
            data_changed = (detector.detected_food != last_displayed_food or 
                          detector.calories != last_displayed_calories)
            
            if current_time - last_update >= update_interval or data_changed:
                # Clear image (reuse buffer)
                image.paste(255, (0, 0, 64, 128))  # Fill with white
                
                # Display calories vertically
                if detector.detected_food and detector.calories > 0:
                    # Food name (vertical, on left side)
                    food_text = detector.detected_food.upper()[:8]  # Limit length
                    draw_vertical_text(image, food_text, 5, 10, font_medium, fill=0)
                    
                    # Calories (vertical, on right side) - simplified for performance
                    cal_text = str(detector.calories)
                    draw_vertical_text(image, cal_text, 40, 10, font_large, fill=0)
                    draw_vertical_text(image, "CAL", 40, 40, font_medium, fill=0)
                    
                    last_displayed_food = detector.detected_food
                    last_displayed_calories = detector.calories
                else:
                    # Waiting for detection
                    status_text = "SCAN"
                    draw_vertical_text(image, status_text, 20, 20, font_medium, fill=0)
                
                # Rotate image 180 degrees for proper orientation on OLED
                rotated_image = image.rotate(180)
                
                # Display on OLED
                disp.ShowImage(disp.getbuffer(rotated_image))
                
                # Clean up rotated image
                del rotated_image
                last_update = current_time
            
            # Longer sleep for Pi Zero W to reduce CPU usage
            time.sleep(1.0)  # Increased from 0.5 to 1.0
            
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
        if disp:
            try:
                disp.clear()
                disp.module_exit()
            except Exception:
                pass
        logging.info("Exiting...")

if __name__ == "__main__":
    main() 