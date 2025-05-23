import numpy as np
import cv2
import os

print("Starting test image generator...")

# Create a directory for test images if it doesn't exist
test_dir = 'test_images'
os.makedirs(test_dir, exist_ok=True)
print(f"Created directory: {test_dir}")

# Generate a synthetic test image with shapes
def generate_test_image(size=(256, 256)):
    print(f"Generating image with size {size}...")
    # Create a black background
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Add a white background with gradient
    for i in range(size[0]):
        for j in range(size[1]):
            # Create a gradient background
            val = int(255 * (i + j) / (size[0] + size[1]))
            image[i, j] = [val, val, val]
    
    # Add a red circle
    center = (size[0] // 3, size[1] // 3)
    radius = min(size) // 6
    cv2.circle(image, center, radius, (0, 0, 255), -1)
    
    # Add a blue rectangle
    pt1 = (size[0] // 2, size[1] // 2)
    pt2 = (int(size[0] * 0.8), int(size[1] * 0.8))
    cv2.rectangle(image, pt1, pt2, (255, 0, 0), -1)
    
    # Add a green triangle
    pts = np.array([[size[0]//5, size[1]*4//5], 
                    [size[0]//3, size[1]*3//5], 
                    [size[0]*2//5, size[1]*4//5]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(image, [pts], (0, 255, 0))
    
    # Add noise to make it more challenging
    noise = np.random.normal(0, 15, size + (3,)).astype(np.int32)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return noisy_image

# Generate a few test images with different sizes
test_sizes = [(256, 256), (384, 384), (512, 512)]

for idx, size in enumerate(test_sizes):
    print(f"Processing size {size}...")
    output_path = os.path.join(test_dir, f"test_image_{size[0]}x{size[1]}.jpg")
    test_image = generate_test_image(size)
    print(f"Saving to {output_path}...")
    try:
        cv2.imwrite(output_path, test_image)
        print(f"Successfully saved: {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

print(f"Test images should be in: {os.path.abspath(test_dir)}")
