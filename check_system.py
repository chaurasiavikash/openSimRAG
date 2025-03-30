#!/usr/bin/env python
"""
Check if the system meets the requirements for running Mistral-7B.
"""

import os
import sys
import subprocess
import importlib.util

def check_package(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"❌ {package_name} is not installed")
        return False
    print(f"✅ {package_name} is installed")
    return True

def check_gpu():
    """Check if a GPU is available and its memory."""
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA is available")
            
            # Get the number of GPUs
            gpu_count = torch.cuda.device_count()
            print(f"   Found {gpu_count} GPU(s)")
            
            # Get the name and memory of each GPU
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                
                # Try to get detailed GPU info using nvidia-smi
                try:
                    import nvidia_ml_py3 as nvml
                    nvml.nvmlInit()
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    total_memory = info.total / 1024**2  # Convert to MB
                    print(f"   GPU {i}: {gpu_name} with {total_memory:.0f} MB memory")
                except:
                    # Fallback if nvidia-smi is not available
                    print(f"   GPU {i}: {gpu_name}")
            
            # Check if there's enough GPU memory for Mistral-7B (needs ~8GB minimum with 4-bit quantization)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            if total_memory < 8:
                print(f"⚠️ GPU memory ({total_memory:.1f} GB) may not be sufficient for Mistral-7B")
                print("   Consider using 4-bit quantization or a smaller model")
            else:
                print(f"✅ GPU memory ({total_memory:.1f} GB) should be sufficient with quantization")
            
            return True
        else:
            print("❌ CUDA is not available")
            print("   Mistral-7B will run on CPU, which will be extremely slow")
            return False
    except ImportError:
        print("❌ PyTorch with CUDA support is not installed")
        return False

def main():
    """Run all checks."""
    print("Checking system requirements for running Mistral-7B...\n")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 8:
        print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        print("   Python 3.8 or newer is recommended")
    
    # Check required packages
    print("\nChecking required packages:")
    packages = [
        "torch", "transformers", "accelerate", "bitsandbytes", 
        "langchain", "chromadb", "langchain_huggingface"
    ]
    all_packages_installed = all(check_package(pkg) for pkg in packages)
    
    # Check GPU
    print("\nChecking GPU:")
    has_gpu = check_gpu()
    
    # Check disk space
    print("\nChecking disk space:")
    try:
        # Get available space in the current directory (in GB)
        if os.name == 'posix':  # Linux/Mac
            output = subprocess.check_output(['df', '-h', '.']).decode('utf-8')
            available = float(output.split('\n')[1].split()[3].replace('G', ''))
        else:  # Windows
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p('.'), None, None, ctypes.pointer(free_bytes))
            available = free_bytes.value / 1024**3  # Convert to GB
        
        if available < 10:
            print(f"⚠️ Available disk space: {available:.1f} GB")
            print("   At least 10 GB is recommended for downloading and using Mistral-7B")
        else:
            print(f"✅ Available disk space: {available:.1f} GB")
    except:
        print("⚠️ Could not determine available disk space")
    
    # Summary
    print("\nSummary:")
    if all_packages_installed and has_gpu:
        print("✅ Your system meets the requirements for running Mistral-7B")
    else:
        print("⚠️ Your system may not be fully ready for running Mistral-7B")
        print("   Please check the issues above and install missing components")
    
    # Recommendation
    print("\nRecommendation:")
    if not has_gpu:
        print("1. Install CUDA and PyTorch with CUDA support")
        print("2. Consider using a cloud-based solution like Google Colab or a GPU server")
    elif not all_packages_installed:
        print("Run the installation script: bash install_mistral.sh")

if __name__ == "__main__":
    main()