import os
import sys
import subprocess

# Define path to the project directory
project_dir = os.path.dirname(os.path.abspath(__file__))

# Run the check script
print("Checking for syntax errors...")
try:
    # Check model file
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.append('{}'); from models.noise_unet import NoiseAwareUNet".format(project_dir)],
        capture_output=True, 
        text=True
    )
    
    if result.returncode == 0:
        print("✅ Model file is valid!")
    else:
        print(f"❌ Error in model file: {result.stderr}")
        sys.exit(1)
    
    # Check app file
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.append('{}'); from app.app import main".format(project_dir)],
        capture_output=True, 
        text=True
    )
    
    if result.returncode == 0:
        print("✅ App file is valid!")
    else:
        print(f"❌ Error in app file: {result.stderr}")
        sys.exit(1)
    
    print("All checks passed! Starting Streamlit app...")
    
    # Start Streamlit app
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.join(project_dir, "streamlit_app.py")])
    
except Exception as e:
    print(f"Error running check: {e}")
    sys.exit(1)
