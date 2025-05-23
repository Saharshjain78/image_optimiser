import requests
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Set the path to the Naruto image
image_path = r"C:\Users\Asus\Documents\Projects\image_optimisation\test\HD-wallpaper-anime-naruto-running-naruto-grass-anime.jpg"

# Create a directory for the results
results_dir = "naruto_test_results"
os.makedirs(results_dir, exist_ok=True)

# Save a copy of the original image to the results directory
original_image = Image.open(image_path)
original_image.save(os.path.join(results_dir, "original_naruto.jpg"))

print(f"Testing with image: {image_path}")
print(f"Image exists: {os.path.exists(image_path)}")
print(f"Image size: {original_image.size}")

# Test different noise modes and strengths
test_configs = [
    {"noise_mode": "disabled", "noise_strength": 0.1, "label": "no_noise"},
    {"noise_mode": "single", "noise_strength": 0.1, "label": "low_noise"},
    {"noise_mode": "single", "noise_strength": 0.3, "label": "high_noise"},
    {"noise_mode": "ensemble", "noise_strength": 0.1, "label": "ensemble"}
]

results = []

# Test each configuration
for config in test_configs:
    print(f"\nTesting with noise_mode={config['noise_mode']}, strength={config['noise_strength']}...")
    
    # Prepare the API request
    with open(image_path, "rb") as img_file:
        files = {"file": ("naruto.jpg", img_file, "image/jpeg")}
        data = {
            "noise_mode": config["noise_mode"],
            "noise_strength": str(config["noise_strength"]),
            "confidence_map": "true"
        }
        
        # Call the API
        start_time = time.time()
        response = requests.post("http://localhost:8000/upload", files=files, data=data)
        api_time = time.time() - start_time
        
        # Process the response
        if response.status_code == 200:
            api_result = response.json()
            print(f"API Response: {api_result}")
            
            # Download the segmentation result
            seg_url = f"http://localhost:8000{api_result['segmentation_url']}"
            conf_url = f"http://localhost:8000{api_result['confidence_map_url']}"
            
            seg_response = requests.get(seg_url)
            conf_response = requests.get(conf_url)
            
            # Save the results
            seg_file = os.path.join(results_dir, f"naruto_seg_{config['label']}.png")
            conf_file = os.path.join(results_dir, f"naruto_conf_{config['label']}.png")
            
            with open(seg_file, "wb") as f:
                f.write(seg_response.content)
            
            with open(conf_file, "wb") as f:
                f.write(conf_response.content)
                
            print(f"Saved segmentation to {seg_file}")
            print(f"Saved confidence map to {conf_file}")
            
            results.append({
                "config": config,
                "api_time": api_time,
                "inference_time": api_result.get("inference_time_ms", 0),
                "seg_file": seg_file,
                "conf_file": conf_file
            })
        else:
            print(f"API Error: {response.status_code} - {response.text}")

# Create a visualization of the results
if results:
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(3, 3, 1)
    plt.imshow(np.array(original_image))
    plt.title("Original Naruto Image")
    plt.axis("off")
    
    # Display segmentation results
    for i, result in enumerate(results):
        # Segmentation mask
        plt.subplot(3, 3, i+2)
        seg_img = Image.open(result["seg_file"])
        plt.imshow(np.array(seg_img), cmap="gray")
        plt.title(f"{result['config']['noise_mode'].capitalize()} {result['config']['noise_strength']}")
        plt.axis("off")
        
        # Confidence map (if we have enough space)
        if i < 3:  # Only show confidence maps for the first 3 results
            plt.subplot(3, 3, i+5)
            conf_img = Image.open(result["conf_file"])
            plt.imshow(np.array(conf_img))
            plt.title(f"Confidence Map ({result['config']['label']})")
            plt.axis("off")
    
    plt.suptitle("Naruto Image Segmentation with Neural Noise", fontsize=16)
    plt.tight_layout()
    
    # Save the comparison visualization
    comparison_file = os.path.join(results_dir, "naruto_comparison.png")
    plt.savefig(comparison_file, dpi=200, bbox_inches="tight")
    print(f"\nSaved comparison visualization to {comparison_file}")
    
    # Print inference time comparison
    print("\nInference Time Comparison:")
    for result in results:
        config = result["config"]
        print(f"- {config['noise_mode']} (strength {config['noise_strength']}): {result['inference_time']} ms")

print("\nAPI testing complete!")
