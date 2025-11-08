#!/usr/bin/env python3
"""
V.A.R.G Dependencies Installation Script
Handles all imports and dependencies with proper error handling
"""

import sys
import subprocess
import importlib
import platform
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DependencyManager:
    """Manages installation and verification of all V.A.R.G dependencies"""
    
    def __init__(self):
        self.is_raspberry_pi = self.detect_raspberry_pi()
        self.python_version = sys.version_info
        self.platform_info = platform.platform()
        self.missing_packages = []
        self.optional_packages = []
        
        # Core dependencies that must be available
        self.core_dependencies = {
            'numpy': 'numpy>=1.21.0,<1.25.0',
            'PIL': 'Pillow>=9.0.0,<11.0.0',
            'psutil': 'psutil>=5.8.0,<6.0.0',
            'dotenv': 'python-dotenv>=0.19.0,<2.0.0',
            'groq': 'groq>=0.4.0,<1.0.0',
            'json': None,  # Built-in
            'time': None,  # Built-in
            'logging': None,  # Built-in
            'datetime': None,  # Built-in
            'os': None,  # Built-in
            'threading': None,  # Built-in
            'queue': None,  # Built-in
            'gc': None,  # Built-in
            'collections': None,  # Built-in
            'base64': None,  # Built-in
            'io': None,  # Built-in
        }
        
        # Optional dependencies for enhanced functionality
        self.optional_dependencies = {
            'skimage': 'scikit-image>=0.19.0,<0.22.0',
            'tflite_runtime': 'tflite-runtime>=2.13.0,<2.15.0',
            'tensorflow': 'tensorflow>=2.13.0,<2.15.0',
            'cv2': 'opencv-python-headless>=4.5.0,<5.0.0',
            'kagglehub': 'kagglehub>=0.2.0,<1.0.0',
        }
        
        # Raspberry Pi specific dependencies
        self.pi_dependencies = {
            'picamera2': 'picamera2>=0.3.0',
            'RPi.GPIO': 'RPi.GPIO>=0.7.0',
            'board': 'adafruit-blinka>=8.0.0,<9.0.0',
            'adafruit_displayio_ssd1306': 'adafruit-circuitpython-displayio-ssd1306>=1.5.0,<2.0.0',
            'displayio': 'adafruit-circuitpython-displayio-ssd1306>=1.5.0,<2.0.0',
        }
    
    def detect_raspberry_pi(self):
        """Detect if running on Raspberry Pi"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
            return 'BCM' in cpuinfo and 'ARM' in cpuinfo
        except:
            return False
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        if self.python_version < (3, 7):
            logger.error(f"Python 3.7+ required, found {self.python_version}")
            return False
        
        logger.info(f"Python version: {sys.version}")
        return True
    
    def install_package(self, package_spec):
        """Install a package using pip"""
        try:
            logger.info(f"Installing {package_spec}...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', package_spec
            ], capture_output=True, text=True, check=True)
            
            logger.info(f"Successfully installed {package_spec}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package_spec}: {e.stderr}")
            return False
    
    def check_import(self, module_name, package_spec=None):
        """Check if a module can be imported"""
        try:
            importlib.import_module(module_name)
            logger.info(f"✅ {module_name} is available")
            return True
        except ImportError:
            logger.warning(f"❌ {module_name} is not available")
            if package_spec:
                self.missing_packages.append((module_name, package_spec))
            return False
    
    def install_system_dependencies(self):
        """Install system-level dependencies on Raspberry Pi"""
        if not self.is_raspberry_pi:
            logger.info("Not on Raspberry Pi, skipping system dependencies")
            return True
        
        logger.info("Installing system dependencies...")
        
        system_packages = [
            'python3-dev',
            'libopenblas-dev',
            'liblapack-dev',
            'libjpeg-dev',
            'libpng-dev',
            'libtiff-dev',
            'libv4l-dev',
            'libfontconfig1-dev',
            'libcairo2-dev',
            'libgdk-pixbuf-2.0-dev',
            'libpango1.0-dev',
            'libglib2.0-dev',
            'libgtk-3-dev',
            'libgstreamer1.0-dev',
            'gfortran',
            'libhdf5-dev',
            'libhdf5-serial-dev',
            'pkg-config',
            'i2c-tools',
        ]
        
        try:
            logger.info("Updating APT package lists...")
            # Update package list
            subprocess.run([
                'sudo', 'apt-get',
                '-o', 'Acquire::Retries=3',
                '-o', 'Acquire::http::Timeout=30',
                '-o', 'Acquire::ForceIPv4=true',
                'update', '-y'
            ], check=True)
            
            # Install packages
            logger.info("Installing system packages (batch, no recommends):")
            for p in system_packages:
                logger.info(f" - {p}")
            cmd = [
                'sudo', 'apt-get',
                '-o', 'Acquire::Retries=3',
                '-o', 'Acquire::http::Timeout=30',
                '-o', 'Acquire::ForceIPv4=true',
                '-o', 'Dpkg::Progress-Fancy=1',
                'install', '-y'
            ] + ['--no-install-recommends'] + system_packages
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                # Retry with --fix-missing once
                logger.warning("Batch install failed; retrying with --fix-missing after update")
                subprocess.run([
                    'sudo', 'apt-get',
                    '-o', 'Acquire::Retries=3',
                    '-o', 'Acquire::http::Timeout=30',
                    '-o', 'Acquire::ForceIPv4=true',
                    'update', '-y'
                ], check=True)
                subprocess.run(cmd + ['--fix-missing'], check=True)
                logger.info("Batch install succeeded on retry")
            
            logger.info("System dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install system dependencies: {e}")
            return False
    
    def check_all_dependencies(self):
        """Check all dependencies"""
        logger.info("Checking dependencies...")
        
        # Check core dependencies
        logger.info("Checking core dependencies...")
        for module, package in self.core_dependencies.items():
            self.check_import(module, package)
        
        # Check optional dependencies
        logger.info("Checking optional dependencies...")
        for module, package in self.optional_dependencies.items():
            if not self.check_import(module, package):
                self.optional_packages.append((module, package))
        
        # Check Pi-specific dependencies
        if self.is_raspberry_pi:
            logger.info("Checking Raspberry Pi specific dependencies...")
            for module, package in self.pi_dependencies.items():
                self.check_import(module, package)
    
    def install_missing_dependencies(self):
        """Install missing dependencies"""
        if not self.missing_packages:
            logger.info("All core dependencies are available!")
            return True
        
        logger.info(f"Installing {len(self.missing_packages)} missing packages...")
        
        success_count = 0
        for module, package in self.missing_packages:
            if self.install_package(package):
                success_count += 1
        
        logger.info(f"Successfully installed {success_count}/{len(self.missing_packages)} packages")
        return success_count == len(self.missing_packages)
    
    def install_optional_dependencies(self):
        """Install optional dependencies with user choice"""
        if not self.optional_packages:
            logger.info("All optional dependencies are available!")
            return True
        
        logger.info("Optional dependencies for enhanced functionality:")
        for i, (module, package) in enumerate(self.optional_packages, 1):
            description = self.get_package_description(module)
            print(f"{i}. {module}: {description}")
        
        # In non-interactive contexts (e.g., scripts), skip prompts and continue
        try:
            import sys as _sys
            if not _sys.stdin.isatty():
                logger.info("Non-interactive mode detected; skipping optional dependencies")
                return True
        except Exception:
            pass

        try:
            choice = input("\nInstall optional dependencies? (y/n/selective): ").lower().strip()
            
            if choice == 'y':
                # Install all
                for module, package in self.optional_packages:
                    self.install_package(package)
            elif choice == 'selective':
                # Let user choose
                for module, package in self.optional_packages:
                    description = self.get_package_description(module)
                    install = input(f"Install {module} ({description})? (y/n): ").lower().strip()
                    if install == 'y':
                        self.install_package(package)
            else:
                logger.info("Skipping optional dependencies")
        
        except KeyboardInterrupt:
            logger.info("Installation cancelled by user")
        
        return True
    
    def get_package_description(self, module):
        """Get description for a package"""
        descriptions = {
            'skimage': 'Advanced computer vision (better food detection)',
            'tflite_runtime': 'TensorFlow Lite for AI food classification',
            'tensorflow': 'Full TensorFlow (larger but more features)',
            'cv2': 'OpenCV for USB camera support',
        }
        return descriptions.get(module, 'Enhanced functionality')
    
    def create_mock_modules(self):
        """Create mock modules for missing Pi-specific dependencies"""
        if self.is_raspberry_pi:
            return  # Don't create mocks on actual Pi
        
        logger.info("Creating mock modules for development...")
        
        # Create a simple mock for Pi-specific modules
        mock_dir = Path("mocks")
        mock_dir.mkdir(exist_ok=True)
        
        # Create __init__.py
        (mock_dir / "__init__.py").write_text("")
        
        # Create mock RPi.GPIO
        mock_gpio = '''
class GPIO:
    BCM = 11
    OUT = 0
    IN = 1
    HIGH = 1
    LOW = 0
    
    @staticmethod
    def setmode(mode): pass
    
    @staticmethod
    def setup(pin, mode): pass
    
    @staticmethod
    def output(pin, state): pass
    
    @staticmethod
    def input(pin): return 0
    
    @staticmethod
    def cleanup(): pass

# Make it available as RPi.GPIO
import sys
sys.modules['RPi'] = sys.modules[__name__]
sys.modules['RPi.GPIO'] = GPIO
'''
        (mock_dir / "rpi_gpio.py").write_text(mock_gpio)
        
        # Add mock directory to Python path
        sys.path.insert(0, str(mock_dir))
        
        logger.info("Mock modules created for development")
    
    def verify_installation(self):
        """Verify that all critical components can be imported"""
        logger.info("Verifying installation...")
        
        critical_imports = [
            'numpy', 'PIL', 'psutil', 'groq', 'json', 'time', 'logging'
        ]
        
        failed_imports = []
        for module in critical_imports:
            try:
                importlib.import_module(module)
                logger.info(f"✅ {module} verified")
            except ImportError as e:
                logger.error(f"❌ {module} failed: {e}")
                failed_imports.append(module)
        
        if failed_imports:
            logger.error(f"Critical imports failed: {failed_imports}")
            return False
        
        logger.info("✅ All critical components verified!")
        return True
    
    def generate_import_report(self):
        """Generate a report of available imports"""
        report = {
            'system_info': {
                'platform': self.platform_info,
                'python_version': str(self.python_version),
                'is_raspberry_pi': self.is_raspberry_pi,
            },
            'available_modules': {},
            'missing_modules': {},
        }
        
        # Check all modules
        all_modules = {**self.core_dependencies, **self.optional_dependencies}
        if self.is_raspberry_pi:
            all_modules.update(self.pi_dependencies)
        
        for module, package in all_modules.items():
            try:
                importlib.import_module(module)
                report['available_modules'][module] = package or 'built-in'
            except ImportError:
                report['missing_modules'][module] = package or 'built-in'
        
        # Save report
        import json
        with open('dependency_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Dependency report saved to dependency_report.json")
        return report
    
    def run_full_setup(self):
        """Run complete dependency setup"""
        logger.info("🚀 Starting V.A.R.G dependency setup...")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Install system dependencies on Pi
        if self.is_raspberry_pi:
            self.install_system_dependencies()
        
        # Check current state
        self.check_all_dependencies()
        
        # Install missing core dependencies
        if not self.install_missing_dependencies():
            logger.error("Failed to install core dependencies")
            return False
        
        # Install optional dependencies
        self.install_optional_dependencies()
        
        # Create mocks for development
        if not self.is_raspberry_pi:
            self.create_mock_modules()
        
        # Verify installation
        if not self.verify_installation():
            logger.error("Installation verification failed")
            return False
        
        # Generate report
        self.generate_import_report()
        
        logger.info("✅ V.A.R.G dependency setup complete!")
        return True

def main():
    """Main setup function"""
    print("🔧 V.A.R.G Dependency Manager")
    print("=" * 40)
    
    manager = DependencyManager()
    
    # Print system info
    print(f"Platform: {manager.platform_info}")
    print(f"Python: {sys.version}")
    print(f"Raspberry Pi: {'Yes' if manager.is_raspberry_pi else 'No'}")
    print()
    
    try:
        success = manager.run_full_setup()
        
        if success:
            print("\n🎉 Setup completed successfully!")
            print("You can now run V.A.R.G with: python3 v.a.r.g.py")
        else:
            print("\n❌ Setup failed. Check the logs above.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
