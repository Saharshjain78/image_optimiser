import os
import sys
import inspect

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def debug_imports():
    """Test importing key modules and report any issues"""
    print("Debugging imports...")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import error: {e}")
    
    try:
        import streamlit
        print(f"✓ Streamlit version: {streamlit.__version__}")
    except ImportError as e:
        print(f"✗ Streamlit import error: {e}")
    
    try:
        import PIL
        print(f"✓ PIL version: {PIL.__version__}")
    except ImportError as e:
        print(f"✗ PIL import error: {e}")
    
    try:
        from models.noise_unet import NoiseAwareUNet
        model = NoiseAwareUNet(3, 1)
        print("✓ NoiseAwareUNet imported and instantiated successfully")
    except Exception as e:
        print(f"✗ NoiseAwareUNet error: {e}")
    
    try:
        from utils.simple_noise_injector import NeuralNoiseInjector
        injector = NeuralNoiseInjector()
        print("✓ NeuralNoiseInjector imported and instantiated successfully")
    except Exception as e:
        print(f"✗ NeuralNoiseInjector error: {e}")
    
    try:
        from utils.data_utils import preprocess_image
        print("✓ preprocess_image imported successfully")
        print(f"  Function signature: {inspect.signature(preprocess_image)}")
    except Exception as e:
        print(f"✗ preprocess_image error: {e}")
    
    try:
        from utils.image_utils import get_sample_image, create_sample_images
        print("✓ image_utils imported successfully")
    except Exception as e:
        print(f"✗ image_utils error: {e}")
    
    try:
        from app.app import main
        print("✓ app.main imported successfully")
    except Exception as e:
        print(f"✗ app.main error: {e}")

def check_model_file():
    """Check the structure of noise_unet.py for indentation issues"""
    model_path = os.path.join(project_root, "models", "noise_unet.py")
    
    with open(model_path, 'r') as f:
        lines = f.readlines()
    
    # Check indentation consistency
    print("Checking model file indentation...")
    
    prev_indent = 0
    in_class = False
    in_method = False
    errors = []
    
    for i, line in enumerate(lines):
        line_num = i + 1
        stripped = line.rstrip()
        
        if not stripped or stripped.startswith('#'):
            continue
            
        # Calculate indentation
        indent = len(line) - len(line.lstrip())
        
        # Check class definitions
        if stripped.startswith('class '):
            in_class = True
            if indent != 0:
                errors.append(f"Line {line_num}: Class definition should have 0 indentation")
        
        # Check method definitions
        if in_class and stripped.lstrip().startswith('def '):
            if not stripped.startswith('def '):  # Not a module-level function
                in_method = True
                if indent != 4:
                    errors.append(f"Line {line_num}: Method definition should have 4 spaces indentation")
        
        # End of method or class
        if in_method and indent <= 4 and (stripped.startswith('def ') or stripped.startswith('class ')):
            in_method = False
            
        if in_class and indent == 0 and not stripped.startswith('class '):
            in_class = False
            in_method = False
        
        prev_indent = indent
    
    if errors:
        print(f"Found {len(errors)} indentation issues:")
        for error in errors:
            print(f"  {error}")
    else:
        print("No indentation issues found in model file.")

if __name__ == "__main__":
    debug_imports()
    check_model_file()
    
    print("\nTo run the app, execute one of the following commands:")
    print(" - python -m streamlit run streamlit_app.py")
    print(" - powershell -File run_app.ps1")
    print(" - start run_app.bat")
