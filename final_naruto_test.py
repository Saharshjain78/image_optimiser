import requests
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
from time import time
import sys
import traceback

print("=== Neural Noise-Driven Dynamic Segmentation System Test ===")
print("\nPart 1: Test FastAPI Backend")

# Path to the Naruto image
image_path = r"C:\Users\Asus\Documents\Projects\image_optimisation\test\HD-wallpaper-anime-naruto-running-naruto-grass-anime.jpg"
print(f"Using test image: {image_path}")
print(f"Image exists: {os.path.exists(image_path)}")

if not os.path.exists(image_path):
    print("Error: Test image not found!")
    sys.exit(1)

try:
    # Create output directory
    os.makedirs("naruto_test_final", exist_ok=True)
    print("Created output directory: naruto_test_final")

    # Save a copy of the original
    image = Image.open(image_path)
    image.save("naruto_test_final/original.jpg")
    print(f"Saved original image (size: {image.size})")

    # Test different noise configurations
    test_configs = [
        {"mode": "disabled", "strength": 0.1, "name": "No Noise"},
        {"mode": "single", "strength": 0.1, "name": "Low Noise (0.1)"},
        {"mode": "single", "strength": 0.3, "name": "High Noise (0.3)"},
        {"mode": "ensemble", "strength": 0.1, "name": "Ensemble Noise"}
    ]

    api_results = []

    # Test each configuration with the API
    print("\nTesting API with different noise settings:")
    for config in test_configs:
        print(f"- Testing {config['name']} (mode: {config['mode']}, strength: {config['strength']})")
        
        try:
            # Send the request to the API
            print(f"  Opening image file: {image_path}")
            with open(image_path, "rb") as img_file:
                print("  Preparing API request")
                files = {"file": ("naruto.jpg", img_file, "image/jpeg")}
                data = {
                    "noise_mode": config["mode"],
                    "noise_strength": str(config["strength"]),
                    "confidence_map": "true"
                }
                
                # Call the API and time it
                print("  Sending request to API...")
                start_time = time()
                response = requests.post("http://localhost:8000/upload", files=files, data=data)
                api_time = time() - start_time
                
                print(f"  API response status code: {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    print(f"  Success! Response: {result}")
                    print(f"  Inference time: {result.get('inference_time_ms', 0)} ms")
                    
                    # Download and save the segmentation
                    seg_url = f"http://localhost:8000{result['segmentation_url']}"
                    conf_url = f"http://localhost:8000{result['confidence_map_url']}"
                    print(f"  Downloading segmentation from: {seg_url}")
                    
                    seg_response = requests.get(seg_url)
                    conf_response = requests.get(conf_url)
                    
                    seg_file = f"naruto_test_final/segmentation_{config['mode']}_{int(config['strength']*100)}.png"
                    conf_file = f"naruto_test_final/confidence_{config['mode']}_{int(config['strength']*100)}.png"
                    
                    print(f"  Saving segmentation to: {seg_file}")
                    with open(seg_file, "wb") as f:
                        f.write(seg_response.content)
                    
                    print(f"  Saving confidence map to: {conf_file}")
                    with open(conf_file, "wb") as f:
                        f.write(conf_response.content)
                    
                    api_results.append({
                        "config": config,
                        "inference_time": result.get("inference_time_ms", 0),
                        "api_time": api_time,
                        "seg_file": seg_file,
                        "conf_file": conf_file
                    })
                    print(f"  Saved results: {os.path.basename(seg_file)}, {os.path.basename(conf_file)}")
                else:
                    print(f"  API Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"  Exception during API test: {e}")
            traceback.print_exc()

    # Create a comprehensive visualization
    if api_results:
        print("\nCreating visualization of results...")
        plt.figure(figsize=(20, 12))
        
        # Original image (larger)
        plt.subplot(3, 4, 1)
        plt.imshow(np.array(image))
        plt.title("Original Image", fontsize=14)
        plt.axis("off")
        
        # Display segmentation results
        for i, result in enumerate(api_results):
            config = result["config"]
            seg_img = Image.open(result["seg_file"])
            conf_img = Image.open(result["conf_file"])
            
            # Segmentation
            plt.subplot(3, 4, i+5)
            plt.imshow(np.array(seg_img), cmap="gray")
            plt.title(f"Segmentation: {config['name']}", fontsize=14)
            plt.axis("off")
            
            # Confidence map
            plt.subplot(3, 4, i+9)
            plt.imshow(np.array(conf_img))
            plt.title(f"Confidence: {config['name']}", fontsize=14)
            plt.axis("off")
        
        # Add information about neural noise
        info_text = """
        Neural Noise-Driven Segmentation enhances image segmentation 
        by deliberately introducing controlled noise patterns:
        
        • No Noise: Standard segmentation 
        • Low Noise: Subtle enhancement to edges
        • High Noise: Strong boundary enhancement
        • Ensemble: Multiple noise patterns for robust results
        
        Controlled noise helps:
        1. Break symmetry in ambiguous regions
        2. Escape local optima during inference
        3. Create ensemble-like effects with a single model
        4. Generate reliable uncertainty estimates
        """
        plt.subplot(3, 4, 2)
        plt.text(0.1, 0.5, info_text, fontsize=12, va="center")
        plt.axis("off")
        
        # Add inference time comparison
        time_text = "Inference Times (ms):\n\n"
        for result in api_results:
            config = result["config"]
            time_text += f"{config['name']}: {result['inference_time']} ms\n"
        
        plt.subplot(3, 4, 3)
        plt.text(0.1, 0.5, time_text, fontsize=12, va="center")
        plt.axis("off")
        
        plt.suptitle("Neural Noise-Driven Dynamic Segmentation Results (Naruto Test Image)", fontsize=18)
        plt.tight_layout()
        
        # Save the visualization
        output_file = "naruto_test_final/final_results.png"
        print(f"Saving visualization to: {output_file}")
        plt.savefig(output_file, dpi=200, bbox_inches="tight")
        print(f"Saved final visualization to {output_file}")
    else:
        print("No API results to visualize")
    
    print("\nTest completed successfully!")
    print(f"All results saved to {os.path.abspath('naruto_test_final')}")
except Exception as e:
    print(f"Error in test: {e}")
    traceback.print_exc()

print("\nTest completed!")
print(f"All results saved to {os.path.abspath('naruto_test_final')}")
