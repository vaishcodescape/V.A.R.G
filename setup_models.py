#!/usr/bin/env python3
"""
Model Setup Script for V.A.R.G
Downloads and sets up TensorFlow Lite food detection models
"""

import os
import json
import urllib.request
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelSetup:
    """Setup and download TensorFlow Lite food detection models"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Available food detection models
        self.available_models = {
            "mobilenet_food_v2": {
                "url": "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/aiy/vision/classifier/food_V1/1.tflite",
                "filename": "mobilenet_food_v2.tflite",
                "labels_url": "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/aiy/vision/classifier/food_V1/1_labels.txt",
                "labels_filename": "mobilenet_food_v2_labels.txt",
                "description": "Google's MobileNet V2 food classifier",
                "size_mb": 10.3,
                "accuracy": "High",
                "speed": "Fast"
            },
            "food101_mobilenet": {
                "url": "https://tfhub.dev/google/lite-model/aiy/vision/classifier/food_V1/1?lite-format=tflite",
                "filename": "food101_mobilenet.tflite",
                "description": "Food-101 dataset trained MobileNet",
                "size_mb": 16.9,
                "accuracy": "Very High",
                "speed": "Medium"
            },
            "efficientnet_food": {
                "url": "https://storage.googleapis.com/tfhub-lite-models/tensorflow/lite-model/efficientnet/lite0/classification/2.tflite",
                "filename": "efficientnet_food.tflite",
                "description": "EfficientNet Lite for food classification",
                "size_mb": 6.9,
                "accuracy": "High",
                "speed": "Very Fast"
            }
        }
        
        # Food-101 class labels (most comprehensive food dataset)
        self.food101_labels = [
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
    
    def download_file(self, url: str, filename: str) -> bool:
        """Download a file from URL"""
        try:
            filepath = self.models_dir / filename
            
            if filepath.exists():
                logger.info(f"File {filename} already exists, skipping download")
                return True
            
            logger.info(f"Downloading {filename} from {url}")
            
            # Download with progress
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    print(f"\rProgress: {percent}%", end="", flush=True)
            
            urllib.request.urlretrieve(url, filepath, progress_hook)
            print()  # New line after progress
            
            logger.info(f"Successfully downloaded {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return False
    
    def create_labels_file(self, model_name: str, labels: list):
        """Create labels file for a model"""
        try:
            labels_filename = f"{model_name}_labels.txt"
            labels_path = self.models_dir / labels_filename
            
            with open(labels_path, 'w') as f:
                for label in labels:
                    f.write(f"{label}\n")
            
            logger.info(f"Created labels file: {labels_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create labels file: {e}")
            return False
    
    def setup_model(self, model_name: str) -> bool:
        """Setup a specific model"""
        if model_name not in self.available_models:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        model_info = self.available_models[model_name]
        
        logger.info(f"Setting up {model_name}:")
        logger.info(f"  Description: {model_info['description']}")
        logger.info(f"  Size: {model_info['size_mb']} MB")
        logger.info(f"  Accuracy: {model_info['accuracy']}")
        logger.info(f"  Speed: {model_info['speed']}")
        
        # Download model file
        success = self.download_file(model_info["url"], model_info["filename"])
        if not success:
            return False
        
        # Download or create labels file
        if "labels_url" in model_info:
            self.download_file(model_info["labels_url"], model_info["labels_filename"])
        else:
            # Use Food-101 labels as default
            model_base = model_info["filename"].replace(".tflite", "")
            self.create_labels_file(model_base, self.food101_labels)
        
        return True
    
    def setup_all_models(self):
        """Setup all available models"""
        logger.info("Setting up all TensorFlow Lite food detection models...")
        
        success_count = 0
        for model_name in self.available_models:
            if self.setup_model(model_name):
                success_count += 1
        
        logger.info(f"Successfully set up {success_count}/{len(self.available_models)} models")
        return success_count > 0
    
    def setup_recommended_model(self):
        """Setup the recommended model for Raspberry Pi Zero W"""
        logger.info("Setting up recommended model for Raspberry Pi Zero W...")
        
        # EfficientNet Lite is best balance of speed/accuracy for Pi Zero W
        recommended = "efficientnet_food"
        
        if self.setup_model(recommended):
            logger.info("✅ Recommended model setup complete!")
            return True
        else:
            # Fallback to MobileNet
            logger.info("Trying fallback model...")
            return self.setup_model("mobilenet_food_v2")
    
    def list_models(self):
        """List all available models"""
        print("\n📋 Available TensorFlow Lite Food Detection Models:")
        print("=" * 60)
        
        for name, info in self.available_models.items():
            status = "✅ Downloaded" if (self.models_dir / info["filename"]).exists() else "❌ Not downloaded"
            print(f"\n🤖 {name}")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size_mb']} MB")
            print(f"   Accuracy: {info['accuracy']}")
            print(f"   Speed: {info['speed']}")
            print(f"   Status: {status}")
        
        print("\n" + "=" * 60)
    
    def create_model_info_file(self):
        """Create model information file"""
        info = {
            "models_directory": str(self.models_dir),
            "available_models": self.available_models,
            "setup_date": str(Path(__file__).stat().st_mtime),
            "usage_instructions": [
                "1. Run 'python setup_models.py' to download models",
                "2. Models will be saved in the 'models' directory",
                "3. V.A.R.G will automatically detect and use available models",
                "4. For best performance on Pi Zero W, use EfficientNet Lite"
            ]
        }
        
        with open(self.models_dir / "model_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info("Created model information file")

def main():
    """Main setup function"""
    import sys
    
    print("🤖 V.A.R.G TensorFlow Lite Model Setup")
    print("=" * 50)
    
    setup = ModelSetup()
    
    # Create model info file
    setup.create_model_info_file()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            logger.info("Setting up all models...")
            setup.setup_all_models()
            return
        elif sys.argv[1] == "--recommended":
            logger.info("Setting up recommended model...")
            setup.setup_recommended_model()
            return
        elif sys.argv[1] == "--non-interactive":
            logger.info("Non-interactive mode: setting up recommended model...")
            setup.setup_recommended_model()
            return
    
    # List available models
    setup.list_models()
    
    # Check if we're in a non-interactive environment (pipe, script, etc.)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        logger.info("Non-interactive environment detected, setting up recommended model...")
        setup.setup_recommended_model()
        logger.info("\n✅ Model setup complete!")
        return

    # Ask user what to do
    print("\n🔧 Setup Options:")
    print("1. Setup recommended model (EfficientNet Lite - best for Pi Zero W)")
    print("2. Setup all models (requires more storage)")
    print("3. Setup specific model")
    print("4. List models only")
    print("5. Skip model setup")
    
    try:
        choice = input("\nEnter your choice (1-5) [default: 1]: ").strip()
        
        # Default to recommended if empty
        if not choice:
            choice = "1"
        
        if choice == "1":
            setup.setup_recommended_model()
        elif choice == "2":
            setup.setup_all_models()
        elif choice == "3":
            print("\nAvailable models:")
            for i, name in enumerate(setup.available_models.keys(), 1):
                print(f"{i}. {name}")
            
            model_choice = input("Enter model number: ").strip()
            try:
                model_idx = int(model_choice) - 1
                model_name = list(setup.available_models.keys())[model_idx]
                setup.setup_model(model_name)
            except (ValueError, IndexError):
                print("Invalid model choice")
        elif choice == "4":
            print("Models listed above.")
        elif choice == "5":
            print("Skipping model setup.")
            return
        else:
            print("Invalid choice, setting up recommended model...")
            setup.setup_recommended_model()
    
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
    except EOFError:
        # Handle EOF in non-interactive contexts
        logger.info("EOF detected, setting up recommended model...")
        setup.setup_recommended_model()
    except Exception as e:
        logger.error(f"Setup failed: {e}")
    
    print("\n✅ Model setup complete!")
    print("You can now run V.A.R.G with TensorFlow Lite food detection.")

if __name__ == "__main__":
    main()
