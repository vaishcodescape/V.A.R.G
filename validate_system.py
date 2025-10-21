#!/usr/bin/env python3
"""
V.A.R.G System Validation Script
Comprehensive validation for Raspberry Pi Zero W deployment
Tests camera, API calls, OLED display, and all critical components
"""

import sys
import os
import json
import time
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemValidator:
    """Validates all V.A.R.G components for Pi Zero W deployment"""
    
    def __init__(self):
        self.is_pi = self.detect_raspberry_pi()
        self.validation_results = {
            'system_info': {},
            'imports': {},
            'camera': {},
            'api': {},
            'oled': {},
            'performance': {},
            'overall_status': 'unknown'
        }
    
    def detect_raspberry_pi(self):
        """Detect if running on Raspberry Pi"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
            return 'BCM' in cpuinfo and 'ARM' in cpuinfo
        except:
            return False
    
    def get_system_info(self):
        """Collect system information"""
        info = {
            'platform': sys.platform,
            'python_version': sys.version,
            'is_raspberry_pi': self.is_pi,
            'architecture': os.uname().machine if hasattr(os, 'uname') else 'unknown'
        }
        
        if self.is_pi:
            try:
                # Get Pi model
                with open('/proc/device-tree/model', 'r') as f:
                    info['pi_model'] = f.read().strip()
                
                # Get memory info
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if 'MemTotal:' in line:
                            info['total_memory'] = line.split()[1] + ' kB'
                            break
                
                # Get temperature
                try:
                    result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        info['temperature'] = result.stdout.strip()
                except:
                    info['temperature'] = 'unavailable'
                
            except Exception as e:
                logger.warning(f"Could not get Pi-specific info: {e}")
        
        self.validation_results['system_info'] = info
        return info
    
    def validate_imports(self):
        """Validate all required imports"""
        print("🔍 Validating imports...")
        
        # Core imports (must work)
        core_imports = [
            'json', 'time', 'logging', 'datetime', 'os', 'threading', 
            'queue', 'gc', 'collections', 'base64', 'io', 'pathlib', 'typing'
        ]
        
        # Required packages
        required_packages = [
            'numpy', 'PIL', 'psutil', 'groq'
        ]
        
        # Optional packages
        optional_packages = [
            'skimage', 'tflite_runtime', 'tensorflow', 'cv2'
        ]
        
        # Pi-specific packages
        pi_packages = [
            'picamera2', 'RPi.GPIO', 'board', 'adafruit_displayio_ssd1306', 'displayio'
        ] if self.is_pi else []
        
        results = {
            'core': {},
            'required': {},
            'optional': {},
            'pi_specific': {}
        }
        
        # Test core imports
        for module in core_imports:
            try:
                __import__(module)
                results['core'][module] = True
                print(f"✅ {module}")
            except ImportError:
                results['core'][module] = False
                print(f"❌ {module}")
        
        # Test required packages
        for module in required_packages:
            try:
                __import__(module)
                results['required'][module] = True
                print(f"✅ {module}")
            except ImportError:
                results['required'][module] = False
                print(f"❌ {module} (REQUIRED)")
        
        # Test optional packages
        for module in optional_packages:
            try:
                __import__(module)
                results['optional'][module] = True
                print(f"✅ {module} (optional)")
            except ImportError:
                results['optional'][module] = False
                print(f"⚠️  {module} (optional)")
        
        # Test Pi-specific packages
        for module in pi_packages:
            try:
                __import__(module)
                results['pi_specific'][module] = True
                print(f"✅ {module} (Pi-specific)")
            except ImportError:
                results['pi_specific'][module] = False
                print(f"⚠️  {module} (Pi-specific)")
        
        self.validation_results['imports'] = results
        
        # Check if core requirements are met
        core_ok = all(results['core'].values())
        required_ok = all(results['required'].values())
        
        if core_ok and required_ok:
            print("✅ All critical imports available")
            return True
        else:
            print("❌ Missing critical imports")
            return False
    
    def validate_camera(self):
        """Validate camera functionality"""
        print("\n📷 Validating camera...")
        
        camera_results = {
            'pi_camera': False,
            'usb_camera': False,
            'test_capture': False
        }
        
        # Test Pi Camera
        if self.is_pi:
            try:
                import picamera2
                picam = picamera2.Picamera2()
                
                # Try to configure and start
                config = picam.create_preview_configuration(main={"size": (320, 240)})
                picam.configure(config)
                picam.start()
                
                # Test capture
                time.sleep(1)
                array = picam.capture_array()
                picam.stop()
                picam.close()
                
                if array is not None and array.size > 0:
                    camera_results['pi_camera'] = True
                    camera_results['test_capture'] = True
                    print("✅ Pi Camera working")
                else:
                    print("❌ Pi Camera capture failed")
                
            except Exception as e:
                print(f"⚠️  Pi Camera not available: {e}")
        
        # Test USB Camera (fallback)
        if not camera_results['pi_camera']:
            try:
                import cv2
                cap = cv2.VideoCapture(0)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret and frame is not None:
                        camera_results['usb_camera'] = True
                        camera_results['test_capture'] = True
                        print("✅ USB Camera working")
                    else:
                        print("❌ USB Camera capture failed")
                else:
                    print("⚠️  USB Camera not available")
                    
            except Exception as e:
                print(f"⚠️  USB Camera test failed: {e}")
        
        self.validation_results['camera'] = camera_results
        return camera_results['test_capture']
    
    def validate_groq_api(self):
        """Validate Groq API connectivity"""
        print("\n🤖 Validating Groq API...")
        
        api_results = {
            'client_init': False,
            'api_key_present': False,
            'connection_test': False,
            'response_test': False
        }
        
        try:
            # Check if API key is available
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                # Try to load from config
                if os.path.exists('config.json'):
                    with open('config.json', 'r') as f:
                        config = json.load(f)
                        api_key = config.get('groq_api_key', '')
                
                if not api_key:
                    # Try .env file
                    if os.path.exists('.env'):
                        with open('.env', 'r') as f:
                            for line in f:
                                if line.startswith('GROQ_API_KEY='):
                                    api_key = line.split('=', 1)[1].strip()
                                    break
            
            if api_key and api_key != 'your_groq_api_key_here':
                api_results['api_key_present'] = True
                print("✅ Groq API key found")
                
                # Test client initialization
                from groq import Groq
                client = Groq(api_key=api_key)
                api_results['client_init'] = True
                print("✅ Groq client initialized")
                
                # Test simple API call (if we have network)
                try:
                    response = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[{"role": "user", "content": "Hello, respond with just 'OK'"}],
                        max_tokens=10,
                        temperature=0
                    )
                    
                    if response and response.choices:
                        api_results['connection_test'] = True
                        api_results['response_test'] = True
                        print("✅ Groq API connection successful")
                    else:
                        print("⚠️  Groq API response empty")
                        
                except Exception as e:
                    print(f"⚠️  Groq API test failed: {e}")
                    # This might be due to network issues, not necessarily a problem
                    
            else:
                print("❌ Groq API key not found or invalid")
                print("   Please set GROQ_API_KEY in .env file")
        
        except Exception as e:
            print(f"❌ Groq API validation failed: {e}")
        
        self.validation_results['api'] = api_results
        return api_results['api_key_present'] and api_results['client_init']
    
    def validate_oled_display(self):
        """Validate OLED display functionality"""
        print("\n📺 Validating OLED display...")
        
        oled_results = {
            'i2c_available': False,
            'display_detected': False,
            'libraries_available': False,
            'display_test': False
        }
        
        if not self.is_pi:
            print("⚠️  Not on Raspberry Pi, skipping OLED test")
            self.validation_results['oled'] = oled_results
            return False
        
        # Check I2C
        try:
            result = subprocess.run(['i2cdetect', '-y', '1'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                oled_results['i2c_available'] = True
                print("✅ I2C interface available")
                
                # Check for common OLED addresses
                output = result.stdout
                if '3c' in output or '3d' in output:
                    oled_results['display_detected'] = True
                    print("✅ OLED display detected on I2C")
                else:
                    print("⚠️  No OLED display detected on I2C")
            else:
                print("❌ I2C interface not available")
                
        except Exception as e:
            print(f"⚠️  I2C test failed: {e}")
        
        # Test OLED libraries
        try:
            import board
            import adafruit_displayio_ssd1306
            import displayio
            from PIL import Image, ImageDraw, ImageFont
            
            oled_results['libraries_available'] = True
            print("✅ OLED libraries available")
            
            # Test display initialization (if detected)
            if oled_results['display_detected']:
                try:
                    displayio.release_displays()
                    i2c = board.I2C()
                    display_bus = displayio.I2CDisplay(i2c, device_address=0x3D)
                    display = adafruit_displayio_ssd1306.SSD1306(display_bus, width=128, height=64)
                    
                    # Create test image
                    img = Image.new('1', (128, 64), 0)
                    draw = ImageDraw.Draw(img)
                    draw.text((10, 20), "V.A.R.G", fill=1)
                    draw.text((10, 35), "Test OK", fill=1)
                    
                    # Convert and display
                    bitmap = displayio.Bitmap(128, 64, 2)
                    for y in range(64):
                        for x in range(128):
                            pixel = img.getpixel((x, y))
                            bitmap[x, y] = 1 if pixel else 0
                    
                    palette = displayio.Palette(2)
                    palette[0] = 0x000000
                    palette[1] = 0xFFFFFF
                    
                    tile_grid = displayio.TileGrid(bitmap, pixel_shader=palette)
                    group = displayio.Group()
                    group.append(tile_grid)
                    display.show(group)
                    
                    oled_results['display_test'] = True
                    print("✅ OLED display test successful")
                    
                    # Clear display after test
                    time.sleep(2)
                    display.show(displayio.Group())
                    
                except Exception as e:
                    print(f"⚠️  OLED display test failed: {e}")
            
        except Exception as e:
            print(f"⚠️  OLED libraries not available: {e}")
        
        self.validation_results['oled'] = oled_results
        return oled_results['libraries_available']
    
    def validate_performance(self):
        """Validate system performance for Pi Zero W"""
        print("\n⚡ Validating system performance...")
        
        perf_results = {
            'memory_available': 0,
            'cpu_info': '',
            'temperature': 0,
            'performance_adequate': False
        }
        
        try:
            import psutil
            
            # Memory check
            memory = psutil.virtual_memory()
            perf_results['memory_available'] = memory.available // (1024 * 1024)  # MB
            print(f"📊 Available memory: {perf_results['memory_available']} MB")
            
            # CPU check
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            perf_results['cpu_info'] = f"{cpu_count} cores"
            if cpu_freq:
                perf_results['cpu_info'] += f" @ {cpu_freq.current:.0f} MHz"
            print(f"🖥️  CPU: {perf_results['cpu_info']}")
            
            # Temperature check (Pi only)
            if self.is_pi:
                try:
                    result = subprocess.run(['vcgencmd', 'measure_temp'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        temp_str = result.stdout.strip()
                        temp_val = float(temp_str.split('=')[1].replace("'C", ""))
                        perf_results['temperature'] = temp_val
                        print(f"🌡️  Temperature: {temp_val}°C")
                        
                        if temp_val > 70:
                            print("⚠️  High temperature detected")
                        elif temp_val > 80:
                            print("❌ Critical temperature!")
                except:
                    pass
            
            # Performance assessment for Pi Zero W
            if perf_results['memory_available'] > 100:  # At least 100MB free
                if self.is_pi:
                    # Pi Zero W specific checks
                    if perf_results['memory_available'] > 200:
                        perf_results['performance_adequate'] = True
                        print("✅ Performance adequate for Pi Zero W")
                    else:
                        print("⚠️  Low memory for Pi Zero W")
                else:
                    perf_results['performance_adequate'] = True
                    print("✅ Performance adequate")
            else:
                print("❌ Insufficient memory")
        
        except Exception as e:
            print(f"⚠️  Performance check failed: {e}")
        
        self.validation_results['performance'] = perf_results
        return perf_results['performance_adequate']
    
    def run_comprehensive_validation(self):
        """Run all validation tests"""
        print("🔍 V.A.R.G System Validation for Raspberry Pi Zero W")
        print("=" * 60)
        
        # Get system info
        system_info = self.get_system_info()
        print(f"🖥️  Platform: {system_info.get('pi_model', system_info['platform'])}")
        print(f"🐍 Python: {system_info['python_version'].split()[0]}")
        print(f"🏗️  Architecture: {system_info['architecture']}")
        if 'total_memory' in system_info:
            print(f"💾 Total Memory: {system_info['total_memory']}")
        print()
        
        # Run validation tests
        tests = [
            ("Imports", self.validate_imports),
            ("Camera", self.validate_camera),
            ("Groq API", self.validate_groq_api),
            ("OLED Display", self.validate_oled_display),
            ("Performance", self.validate_performance)
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"❌ {test_name} test failed with error: {e}")
                results[test_name] = False
        
        # Overall assessment
        print("\n" + "=" * 60)
        print("📋 VALIDATION SUMMARY:")
        print("-" * 30)
        
        critical_tests = ['Imports', 'Camera']
        important_tests = ['Groq API', 'Performance']
        optional_tests = ['OLED Display']
        
        critical_passed = all(results.get(test, False) for test in critical_tests)
        important_passed = sum(results.get(test, False) for test in important_tests)
        optional_passed = sum(results.get(test, False) for test in optional_tests)
        
        if critical_passed:
            if important_passed >= 1:
                if important_passed == 2 and optional_passed == 1:
                    status = "🎉 EXCELLENT - All systems operational!"
                    self.validation_results['overall_status'] = 'excellent'
                elif important_passed == 2:
                    status = "✅ GOOD - Core functionality ready"
                    self.validation_results['overall_status'] = 'good'
                else:
                    status = "⚠️  BASIC - Limited functionality"
                    self.validation_results['overall_status'] = 'basic'
            else:
                status = "❌ POOR - Missing critical components"
                self.validation_results['overall_status'] = 'poor'
        else:
            status = "❌ FAILED - Cannot run V.A.R.G"
            self.validation_results['overall_status'] = 'failed'
        
        print(f"Overall Status: {status}")
        print()
        
        # Specific recommendations
        print("💡 RECOMMENDATIONS:")
        if not results.get('Imports', False):
            print("1. Install missing dependencies: python3 install_dependencies.py")
        
        if not results.get('Camera', False):
            print("2. Check camera connection and enable camera interface")
            print("   sudo raspi-config -> Interface Options -> Camera -> Enable")
        
        if not results.get('Groq API', False):
            print("3. Set up Groq API key in .env file")
            print("   GROQ_API_KEY=your_api_key_here")
        
        if not results.get('OLED Display', False) and self.is_pi:
            print("4. Enable I2C and check OLED display connection")
            print("   sudo raspi-config -> Interface Options -> I2C -> Enable")
        
        if not results.get('Performance', False):
            print("5. Free up memory or consider Pi upgrade")
        
        # Save validation report
        self.save_validation_report()
        
        return self.validation_results['overall_status'] in ['excellent', 'good', 'basic']
    
    def save_validation_report(self):
        """Save validation report to file"""
        try:
            with open('validation_report.json', 'w') as f:
                json.dump(self.validation_results, f, indent=2, default=str)
            print(f"\n📄 Validation report saved to: validation_report.json")
        except Exception as e:
            print(f"⚠️  Could not save validation report: {e}")

def main():
    """Main validation function"""
    validator = SystemValidator()
    
    try:
        success = validator.run_comprehensive_validation()
        
        print("\n" + "=" * 60)
        if success:
            print("🚀 V.A.R.G is ready to deploy!")
            print("   Run: python3 v.a.r.g.py")
        else:
            print("🔧 V.A.R.G needs setup before deployment")
            print("   Follow the recommendations above")
        
        return success
        
    except KeyboardInterrupt:
        print("\n\nValidation cancelled by user")
        return False
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
