import requests
import os
from PIL import Image
import matplotlib.pyplot as plt
import io
import sys

print("API Test script starting...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Create a directory for results if it doesn't exist
os.makedirs('api_test_results', exist_ok=True)
print(f"Created api_test_results directory")

# Path to a test image
test_image_path = 'test_images/test_image_512x512.jpg'
print(f"Test image path: {test_image_path}")
print(f"Test image exists: {os.path.exists(test_image_path)}")

# Function to test the API with different noise modes
def test_api_with_noise_mode(noise_mode, noise_strength=0.1):
    print(f"\nTesting API with noise_mode={noise_mode}, strength={noise_strength}...")
    
    # Open image file
    with open(test_image_path, 'rb') as img_file:
        # API call
        files = {'file': ('image.jpg', img_file, 'image/jpeg')}
        data = {
            'noise_mode': noise_mode,
            'noise_strength': str(noise_strength),
            'confidence_map': 'true'
        }
        
        # Send request to API
        response = requests.post(
            'http://localhost:8000/upload',
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"API Response: {result}")
            
            # Download segmentation result
            seg_url = f"http://localhost:8000{result['segmentation_url']}"
            seg_response = requests.get(seg_url)
            
            if seg_response.status_code == 200:
                # Save the segmentation result
                output_filename = f"api_test_results/segmentation_{noise_mode}_{int(noise_strength*100)}.png"
                with open(output_filename, 'wb') as f:
                    f.write(seg_response.content)
                print(f"Saved segmentation to {output_filename}")
                
                # If confidence map is available, download it too
                if result.get('confidence_map_url'):
                    conf_url = f"http://localhost:8000{result['confidence_map_url']}"
                    conf_response = requests.get(conf_url)
                    
                    if conf_response.status_code == 200:
                        conf_filename = f"api_test_results/confidence_{noise_mode}_{int(noise_strength*100)}.png"
                        with open(conf_filename, 'wb') as f:
                            f.write(conf_response.content)
                        print(f"Saved confidence map to {conf_filename}")
            else:
                print(f"Error fetching segmentation: {seg_response.status_code}")
        else:
            print(f"API Error: {response.status_code}")
            print(response.text)

# Test with different noise modes
print(f"Using test image: {test_image_path}")
test_api_with_noise_mode('disabled')
test_api_with_noise_mode('single', 0.1)
test_api_with_noise_mode('single', 0.3)
test_api_with_noise_mode('ensemble', 0.1)

print("\nAPI testing completed! Results saved to api_test_results directory.")

# Display the test image
img = Image.open(test_image_path)
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title("Test Image")
plt.axis('off')
plt.savefig('api_test_results/original_test_image.png')
print("Original test image saved for reference.")
