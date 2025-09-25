#!/usr/bin/env python3
"""
Setup script for Advanced RAG Customer Support AI Assistant
Automates the complete installation and setup process
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a system command with error handling"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def setup_virtual_environment():
    """Create and activate virtual environment"""
    if not os.path.exists("venv"):
        if not run_command("python -m venv venv", "Creating virtual environment"):
            return False
    
    # Activation command depends on OS
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    print(f"✅ Virtual environment ready")
    print(f"📝 To activate manually: {activate_cmd}")
    return pip_cmd

def install_dependencies(pip_cmd):
    """Install Python dependencies"""
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        return False
    
    return True

def test_installation():
    """Test if installation was successful"""
    print("\n🧪 Testing installation...")
    
    # Test imports
    try:
        import torch
        import transformers
        import sentence_transformers
        import faiss
        import fastapi
        import requests
        print("✅ All core dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def main():
    """Main setup process"""
    print("🚀 Advanced RAG Customer Support AI Assistant - Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup virtual environment
    pip_cmd = setup_virtual_environment()
    if not pip_cmd:
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies(pip_cmd):
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("\n⚠️ Installation may be incomplete. Please check error messages above.")
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("2. Start the system:")
    print("   python main.py")
    
    print("3. Test the API:")
    print("   python test_script.py")
    
    print("4. Run evaluation:")
    print("   python simple_eval.py")
    
    print("\n🌐 API will be available at: http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("=" * 60)

if __name__ == "__main__":
    main()