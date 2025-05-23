import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Try to import the model
try:
    from models.noise_unet import NoiseAwareUNet
    print("✅ Successfully imported NoiseAwareUNet - indentation fixed!")
except IndentationError as e:
    print(f"❌ Indentation error still exists: {e}")
except Exception as e:
    print(f"❌ Other error occurred: {e}")
    
print("Script completed.")
