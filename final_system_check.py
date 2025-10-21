#!/usr/bin/env python3
"""
V.A.R.G Final System Check
Comprehensive pre-deployment validation for Raspberry Pi Zero W
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def print_section(title):
    print(f"\n📋 {title}")
    print('-'*40)

def check_system_readiness():
    """Check if system is ready for V.A.R.G deployment"""
    
    print_header("V.A.R.G FINAL SYSTEM CHECK")
    print("Comprehensive validation for Raspberry Pi Zero W deployment")
    
    issues = []
    warnings = []
    
    # 1. Check Python version
    print_section("Python Environment")
    python_version = sys.version_info
    if python_version >= (3, 7):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        issues.append(f"Python version too old: {python_version}")
        print(f"❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} (need 3.7+)")
    
    # 2. Check critical files
    print_section("Required Files")
    required_files = [
        'v.a.r.g.py',
        'config.json',
        'requirements.txt',
        'install_dependencies.py',
        'setup_models.py'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            issues.append(f"Missing required file: {file}")
            print(f"❌ {file}")
    
    # 3. Check configuration
    print_section("Configuration")
    if os.path.exists('config.json'):
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            # Check critical config values
            if config.get('groq_api_key'):
                if config['groq_api_key'] == 'your_groq_api_key_here' or not config['groq_api_key']:
                    warnings.append("Groq API key not set in config.json")
                    print("⚠️  Groq API key not configured")
                else:
                    print("✅ Groq API key configured")
            
            # Check camera settings
            cam_width = config.get('camera_width', 320)
            cam_height = config.get('camera_height', 240)
            if cam_width <= 320 and cam_height <= 240:
                print(f"✅ Camera resolution optimized for Pi Zero W: {cam_width}x{cam_height}")
            else:
                warnings.append(f"Camera resolution may be too high for Pi Zero W: {cam_width}x{cam_height}")
                print(f"⚠️  Camera resolution: {cam_width}x{cam_height} (consider reducing)")
            
            # Check detection interval
            interval = config.get('detection_interval', 3.0)
            if interval >= 2.0:
                print(f"✅ Detection interval appropriate: {interval}s")
            else:
                warnings.append(f"Detection interval may be too fast for Pi Zero W: {interval}s")
                print(f"⚠️  Detection interval: {interval}s (consider increasing)")
                
        except Exception as e:
            issues.append(f"Invalid config.json: {e}")
            print(f"❌ config.json error: {e}")
    
    # 4. Check environment file
    print_section("Environment Configuration")
    env_files = ['.env', '.env.template']
    env_found = False
    
    for env_file in env_files:
        if os.path.exists(env_file):
            env_found = True
            print(f"✅ {env_file} exists")
            
            if env_file == '.env':
                try:
                    with open('.env', 'r') as f:
                        content = f.read()
                        if 'GROQ_API_KEY=' in content:
                            if 'your_groq_api_key_here' in content:
                                warnings.append("Groq API key not set in .env file")
                                print("⚠️  Groq API key placeholder in .env")
                            else:
                                print("✅ Groq API key configured in .env")
                        else:
                            warnings.append("No Groq API key in .env file")
                            print("⚠️  No Groq API key in .env")
                except Exception as e:
                    warnings.append(f"Could not read .env file: {e}")
    
    if not env_found:
        warnings.append("No environment file found")
        print("⚠️  No .env file found")
    
    # 5. Check directories
    print_section("Directory Structure")
    directories = ['models', 'detections', 'logs']
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory}/ directory exists")
        else:
            print(f"📁 {directory}/ will be created automatically")
    
    # 6. Check Pi-specific settings (if on Pi)
    print_section("Raspberry Pi Configuration")
    is_pi = False
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
        is_pi = 'BCM' in cpuinfo and 'ARM' in cpuinfo
    except:
        pass
    
    if is_pi:
        print("✅ Running on Raspberry Pi")
        
        # Check camera interface
        try:
            result = subprocess.run(['vcgencmd', 'get_camera'], capture_output=True, text=True)
            if 'detected=1' in result.stdout:
                print("✅ Camera interface enabled")
            else:
                warnings.append("Camera interface may not be enabled")
                print("⚠️  Camera interface status unclear")
        except:
            warnings.append("Could not check camera interface")
            print("⚠️  Could not check camera interface")
        
        # Check I2C interface
        try:
            result = subprocess.run(['i2cdetect', '-y', '1'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ I2C interface available")
            else:
                warnings.append("I2C interface not available")
                print("⚠️  I2C interface not available")
        except:
            warnings.append("Could not check I2C interface")
            print("⚠️  Could not check I2C interface")
        
        # Check memory
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if 'MemTotal:' in line:
                        total_kb = int(line.split()[1])
                        total_mb = total_kb // 1024
                        if total_mb >= 400:
                            print(f"✅ Total memory: {total_mb} MB")
                        else:
                            warnings.append(f"Low memory: {total_mb} MB")
                            print(f"⚠️  Total memory: {total_mb} MB (may be insufficient)")
                        break
        except:
            warnings.append("Could not check memory")
            print("⚠️  Could not check memory")
    else:
        print("ℹ️  Not running on Raspberry Pi (development mode)")
    
    # 7. Test basic imports
    print_section("Critical Dependencies")
    critical_imports = ['json', 'time', 'os', 'sys', 'logging']
    
    for module in critical_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            issues.append(f"Missing critical module: {module}")
            print(f"❌ {module}")
    
    # 8. Check for common issues
    print_section("Common Issues Check")
    
    # Check for old OpenCV installations that might conflict
    try:
        import cv2
        print("ℹ️  OpenCV available (may be used for USB camera fallback)")
    except ImportError:
        print("ℹ️  OpenCV not available (will use Pi Camera only)")
    
    # Check disk space
    try:
        stat = os.statvfs('.')
        free_bytes = stat.f_bavail * stat.f_frsize
        free_mb = free_bytes // (1024 * 1024)
        
        if free_mb >= 1000:
            print(f"✅ Free disk space: {free_mb} MB")
        elif free_mb >= 500:
            print(f"⚠️  Free disk space: {free_mb} MB (consider cleanup)")
            warnings.append(f"Low disk space: {free_mb} MB")
        else:
            print(f"❌ Free disk space: {free_mb} MB (insufficient)")
            issues.append(f"Insufficient disk space: {free_mb} MB")
    except:
        warnings.append("Could not check disk space")
        print("⚠️  Could not check disk space")
    
    # Final assessment
    print_header("FINAL ASSESSMENT")
    
    if issues:
        print("❌ CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"   • {issue}")
        print("\n🔧 REQUIRED ACTIONS:")
        print("   1. Fix critical issues above")
        print("   2. Run: python3 install_dependencies.py")
        print("   3. Run: python3 validate_system.py")
        status = "FAILED"
    elif warnings:
        print("⚠️  WARNINGS FOUND:")
        for warning in warnings:
            print(f"   • {warning}")
        print("\n💡 RECOMMENDED ACTIONS:")
        print("   1. Address warnings above")
        print("   2. Set Groq API key in .env file")
        print("   3. Run: python3 validate_system.py")
        print("   4. Test: python3 v.a.r.g.py")
        status = "READY WITH WARNINGS"
    else:
        print("✅ ALL CHECKS PASSED!")
        print("\n🚀 READY TO DEPLOY:")
        print("   1. Run: python3 v.a.r.g.py")
        print("   2. Or: ./start_varg.sh")
        print("   3. Monitor: ./monitor_varg.sh")
        status = "READY"
    
    print(f"\n📊 System Status: {status}")
    
    # Save check results
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'status': status,
        'issues': issues,
        'warnings': warnings,
        'is_raspberry_pi': is_pi,
        'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}"
    }
    
    try:
        with open('system_check_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Results saved to: system_check_results.json")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    return len(issues) == 0

def main():
    """Main function"""
    try:
        success = check_system_readiness()
        return success
    except KeyboardInterrupt:
        print("\n\nSystem check cancelled by user")
        return False
    except Exception as e:
        print(f"\n❌ System check failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
