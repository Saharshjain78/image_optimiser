import streamlit as st
import subprocess
import sys
import os

# Get the project root directory
root_dir = os.path.dirname(os.path.abspath(__file__))

# Add project root to Python path
sys.path.append(root_dir)

# Ensure test_images directory exists and create sample images if needed
from utils.image_utils import create_sample_images
create_sample_images()

# Import the Streamlit app main function
from app.app import main

# Run the Streamlit app
if __name__ == "__main__":
    main()
