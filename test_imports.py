#!/usr/bin/env python3
"""
V.A.R.G Import Test Script
Tests all imports to ensure the system is ready to run
"""

import sys
import traceback
from pathlib import Path

def test_import(module_name, description="", optional=False):
    """Test importing a module"""
    try:
        if '.' in module_name:
            # Handle submodule imports
            parts = module_name.split('.')
            module = __import__(module_name)
            for part in parts[1:]:
                module = getattr(module, part)
        else:
            __import__(module_name)
        
        status = "✅"
        message = f"{status} {module_name}"
        if description:
            message += f" - {description}"
        print(message)
        return True
        
    except ImportError as e:
        status = "⚠️" if optional else "❌"
        message = f"{status} {module_name}"
        if description:
            message += f" - {description}"
        if optional:
            message += " (optional)"
        else:
            message += f" - MISSING: {e}"
        print(message)
        return False
    except Exception as e:
        print(f"❌ {module_name} - ERROR: {e}")
        return False

def main():
    """Test all imports required by V.A.R.G"""
    print("🧪 V.A.R.G Import Test")
    print("=" * 50)
    
    # Track results
    core_passed = 0
    core_total = 0
    optional_passed = 0
    optional_total = 0
    
    print("\n📦 Core Dependencies (Required):")
    print("-" * 30)
    
    core_imports = [
        ("numpy", "Numerical computing"),
        ("PIL", "Image processing"),
        ("PIL.Image", "PIL Image module"),
        ("PIL.ImageDraw", "PIL drawing"),
        ("PIL.ImageFont", "PIL fonts"),
        ("PIL.ImageFilter", "PIL filters"),
        ("PIL.ImageEnhance", "PIL enhancement"),
        ("psutil", "System monitoring"),
        ("requests", "HTTP client"),
        ("json", "JSON handling"),
        ("time", "Time functions"),
        ("logging", "Logging"),
        ("datetime", "Date/time"),
        ("os", "Operating system interface"),
        ("threading", "Threading"),
        ("queue", "Queue data structure"),
        ("gc", "Garbage collection"),
        ("collections", "Collections"),
        ("base64", "Base64 encoding"),
        ("io", "I/O operations"),
        ("pathlib", "Path handling"),
        ("typing", "Type hints"),
    ]
    
    for module, desc in core_imports:
        core_total += 1
        if test_import(module, desc):
            core_passed += 1
    
    print(f"\n📊 Core Dependencies: {core_passed}/{core_total} passed")
    
    print("\n🔧 Optional Dependencies (Enhanced Features):")
    print("-" * 40)
    
    optional_imports = [
        ("skimage", "Scikit-image for advanced CV"),
        ("skimage.measure", "Image measurement"),
        ("skimage.morphology", "Morphological operations"),
        ("skimage.color", "Color space conversions"),
        ("skimage.feature", "Feature detection"),
        ("skimage.util", "Image utilities"),
        ("groq", "Groq LLM client"),
        ("tflite_runtime", "TensorFlow Lite runtime"),
        ("tflite_runtime.interpreter", "TFLite interpreter"),
        ("tensorflow", "Full TensorFlow"),
        ("cv2", "OpenCV"),
    ]
    
    for module, desc in optional_imports:
        optional_total += 1
        if test_import(module, desc, optional=True):
            optional_passed += 1
    
    print(f"\n📊 Optional Dependencies: {optional_passed}/{optional_total} available")
    
    print("\n🔌 Hardware-Specific Dependencies:")
    print("-" * 35)
    
    hardware_imports = [
        ("picamera2", "Pi Camera v2"),
        ("RPi", "Raspberry Pi GPIO"),
        ("RPi.GPIO", "GPIO control"),
        ("board", "CircuitPython board"),
        ("adafruit_displayio_ssd1306", "OLED display driver"),
        ("displayio", "Display I/O"),
    ]
    
    hw_passed = 0
    hw_total = len(hardware_imports)
    
    for module, desc in hardware_imports:
        if test_import(module, desc, optional=True):
            hw_passed += 1
    
    print(f"\n📊 Hardware Dependencies: {hw_passed}/{hw_total} available")
    
    # Overall assessment
    print("\n" + "=" * 50)
    print("📋 ASSESSMENT:")
    
    if core_passed == core_total:
        print("✅ All core dependencies available - V.A.R.G can run!")
    else:
        print(f"❌ Missing {core_total - core_passed} core dependencies - V.A.R.G may not work properly")
        print("   Run: python3 install_dependencies.py")
    
    if optional_passed > 0:
        print(f"🔧 {optional_passed} optional features available")
        if optional_passed < optional_total:
            print(f"   {optional_total - optional_passed} optional features missing (enhanced functionality)")
    
    if hw_passed > 0:
        print(f"🔌 {hw_passed} hardware features available")
        if hw_passed < hw_total:
            print(f"   {hw_total - hw_passed} hardware features missing (may be normal on non-Pi systems)")
    
    # Specific recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if core_passed < core_total:
        print("1. Install missing core dependencies: python3 install_dependencies.py")
    
    if "tflite_runtime" not in [imp[0] for imp in optional_imports if test_import(imp[0], "", True)]:
        if "tensorflow" not in [imp[0] for imp in optional_imports if test_import(imp[0], "", True)]:
            print("2. Install TensorFlow Lite for AI food detection: pip install tflite-runtime")
    
    if "skimage" not in [imp[0] for imp in optional_imports if test_import(imp[0], "", True)]:
        print("3. Install scikit-image for better computer vision: pip install scikit-image")
    
    # Test basic functionality
    print("\n🧪 FUNCTIONALITY TEST:")
    try:
        import numpy as np
        from PIL import Image
        import json
        
        # Test basic operations
        arr = np.array([1, 2, 3])
        img = Image.new('RGB', (100, 100), color='red')
        data = json.dumps({"test": True})
        
        print("✅ Basic functionality test passed")
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
    
    print("\n" + "=" * 50)
    
    # Return success status
    return core_passed == core_total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
