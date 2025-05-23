import os
import subprocess
import sys

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

# Install requirements
print("Installing dependencies...")
subprocess.call([sys.executable, "-m", "pip", "install", "-r", os.path.join(current_dir, "requirements.txt")])

# Run Streamlit app
print("Starting Streamlit app...")
subprocess.call([sys.executable, "-m", "streamlit", "run", os.path.join(current_dir, "streamlit_app.py")])

print("Done!")
