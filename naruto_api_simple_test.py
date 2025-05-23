import requests
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

# Path to the Naruto image
image_path = r"C:\Users\Asus\Documents\Projects\image_optimisation\test\HD-wallpaper-anime-naruto-running-naruto-grass-anime.jpg"

# Create a directory for results
os.makedirs("naruto_api_results", exist_ok=True)

print(f"Testing with image: {image_path}")
print(f"Image exists: {os.path.exists(image_path)}")

# Read the image
image = Image.open(image_path)
print(f"Image size: {image.size}")

# Save a copy of the original image
image.save("naruto_api_results/original_naruto.jpg")

# Set up the API request
with open(image_path, "rb") as img_file:
    files = {"file": ("naruto.jpg", img_file, "image/jpeg")}
    data = {
        "noise_mode": "single",
        "noise_strength": "0.2",
        "confidence_map": "true"
    }
    
    print("Sending request to API...")
    response = requests.post("http://localhost:8000/upload", files=files, data=data)
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"API Response: {result}")
        
        # Download the segmentation result
        seg_url = f"http://localhost:8000{result['segmentation_url']}"
        conf_url = f"http://localhost:8000{result['confidence_map_url']}"
        
        print(f"Downloading segmentation from: {seg_url}")
        seg_response = requests.get(seg_url)
        
        print(f"Downloading confidence map from: {conf_url}")
        conf_response = requests.get(conf_url)
        
        # Save the results
        seg_file = "naruto_api_results/segmentation.png"
        conf_file = "naruto_api_results/confidence.png"
        
        with open(seg_file, "wb") as f:
            f.write(seg_response.content)
        
        with open(conf_file, "wb") as f:
            f.write(conf_response.content)
        
        print(f"Saved segmentation to {seg_file}")
        print(f"Saved confidence map to {conf_file}")
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(np.array(image))
        plt.title("Original Image")
        plt.axis("off")
        
        # Segmentation mask
        plt.subplot(1, 3, 2)
        seg_img = Image.open(seg_file)
        plt.imshow(np.array(seg_img), cmap="gray")
        plt.title("Segmentation Mask")
        plt.axis("off")
        
        # Confidence map
        plt.subplot(1, 3, 3)
        conf_img = Image.open(conf_file)
        plt.imshow(np.array(conf_img))
        plt.title("Confidence Map")
        plt.axis("off")
        
        plt.suptitle(f"Neural Noise Segmentation: {result.get('noise_mode', 'unknown')} mode, strength {result.get('noise_strength', 'unknown')}")
        plt.tight_layout()
        
        # Save the visualization
        plt.savefig("naruto_api_results/naruto_visualization.png", dpi=200, bbox_inches="tight")
        print("Saved visualization to naruto_api_results/naruto_visualization.png")
    else:
        print(f"API Error: {response.text}")

print("Test complete!")
