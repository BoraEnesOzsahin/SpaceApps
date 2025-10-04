"""Quick test script to verify XGBoost setup works locally."""

import sys
from pathlib import Path

def check_xgboost():
    """Check if XGBoost is available and can use GPU."""
    print("="*60)
    print("XGBOOST GPU SETUP CHECK")
    print("="*60)
    
    # Check if XGBoost is installed
    try:
        import xgboost as xgb
        print(f"✅ XGBoost installed: version {xgb.__version__}")
    except ImportError:
        print("❌ XGBoost not installed")
        print("\nInstall with: pip install xgboost")
        return False
    
    # Check build info
    try:
        build_info = xgb.build_info()
        cuda_support = build_info.get('USE_CUDA', False)
        
        if cuda_support:
            print(f"✅ XGBoost built with CUDA support")
        else:
            print(f"⚠️  XGBoost built WITHOUT CUDA support (CPU-only)")
            print("   This is OK for local testing, but GPU training requires CUDA build")
            print("   Google Colab has GPU-enabled XGBoost by default")
    except Exception as e:
        print(f"⚠️  Could not check CUDA support: {e}")
    
    # Check if model registry was updated
    try:
        from src.model import MODEL_REGISTRY, XGBOOST_AVAILABLE
        
        if XGBOOST_AVAILABLE:
            print(f"✅ XGBoost models available in registry")
            
            if "xgboost_gpu" in MODEL_REGISTRY:
                print(f"   - xgboost_gpu: {MODEL_REGISTRY['xgboost_gpu'].description}")
            if "xgboost_cpu" in MODEL_REGISTRY:
                print(f"   - xgboost_cpu: {MODEL_REGISTRY['xgboost_cpu'].description}")
        else:
            print(f"❌ XGBoost not available in model registry")
    except Exception as e:
        print(f"❌ Error checking model registry: {e}")
    
    # Check training script
    train_script = Path("train_xgboost_gpu.py")
    if train_script.exists():
        print(f"✅ Training script found: {train_script}")
    else:
        print(f"❌ Training script not found: {train_script}")
    
    # Check Colab notebook
    colab_notebook = Path("Colab_Training.ipynb")
    if colab_notebook.exists():
        print(f"✅ Colab notebook found: {colab_notebook}")
    else:
        print(f"❌ Colab notebook not found: {colab_notebook}")
    
    print("="*60)
    print("\nNEXT STEPS:")
    print("="*60)
    
    # Check if GPU is available locally
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected locally!")
            print("\nYou can train locally with GPU:")
            print("   python train_xgboost_gpu.py --cv-splits 5")
        else:
            print("⚠️  No NVIDIA GPU detected locally")
            print("\nOptions:")
            print("1. Train on CPU locally (will be slower):")
            print("   python train_xgboost_gpu.py --no-gpu --cv-splits 5")
            print("\n2. Train on Google Colab with T4 GPU (recommended):")
            print("   - Upload Colab_Training.ipynb to Google Colab")
            print("   - Select Runtime → Change runtime type → T4 GPU")
            print("   - Run all cells")
    except Exception:
        print("⚠️  Could not detect GPU (nvidia-smi not found)")
        print("\nRecommended: Use Google Colab with T4 GPU")
        print("   - Upload Colab_Training.ipynb to Google Colab")
        print("   - Select Runtime → Change runtime type → T4 GPU")
        print("   - Run all cells")
    
    print("="*60)
    
    return True


if __name__ == "__main__":
    check_xgboost()
