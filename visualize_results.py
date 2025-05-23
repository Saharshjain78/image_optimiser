import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import sys

print("Starting visualization script...")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Path to the results directory
results_dir = 'api_test_results'
print(f"Results directory: {results_dir}")
print(f"Directory exists: {os.path.exists(results_dir)}")

# List files in the directory
if os.path.exists(results_dir):
    print("Files in the results directory:")
    for file in os.listdir(results_dir):
        print(f"  - {file}")
else:
    print("Results directory does not exist")
    sys.exit(1)

try:
    # Load the original test image
    original_image_path = os.path.join(results_dir, 'original_test_image.png')
    print(f"Loading original image from: {original_image_path}")
    original_image = Image.open(original_image_path)

    # Load the segmentation results
    segmentation_disabled_path = os.path.join(results_dir, 'segmentation_disabled_10.png')
    print(f"Loading segmentation (no noise) from: {segmentation_disabled_path}")
    segmentation_disabled = Image.open(segmentation_disabled_path)
    
    segmentation_single_10_path = os.path.join(results_dir, 'segmentation_single_10.png')
    print(f"Loading segmentation (single noise 0.1) from: {segmentation_single_10_path}")
    segmentation_single_10 = Image.open(segmentation_single_10_path)
    
    segmentation_single_30_path = os.path.join(results_dir, 'segmentation_single_30.png')
    print(f"Loading segmentation (single noise 0.3) from: {segmentation_single_30_path}")
    segmentation_single_30 = Image.open(segmentation_single_30_path)
    
    segmentation_ensemble_path = os.path.join(results_dir, 'segmentation_ensemble_10.png')
    print(f"Loading segmentation (ensemble) from: {segmentation_ensemble_path}")
    segmentation_ensemble = Image.open(segmentation_ensemble_path)

    # Load the confidence maps
    confidence_disabled_path = os.path.join(results_dir, 'confidence_disabled_10.png')
    print(f"Loading confidence map (no noise) from: {confidence_disabled_path}")
    confidence_disabled = Image.open(confidence_disabled_path)
    
    confidence_single_10_path = os.path.join(results_dir, 'confidence_single_10.png')
    print(f"Loading confidence map (single noise 0.1) from: {confidence_single_10_path}")
    confidence_single_10 = Image.open(confidence_single_10_path)
    
    confidence_single_30_path = os.path.join(results_dir, 'confidence_single_30.png')
    print(f"Loading confidence map (single noise 0.3) from: {confidence_single_30_path}")
    confidence_single_30 = Image.open(confidence_single_30_path)
    
    confidence_ensemble_path = os.path.join(results_dir, 'confidence_ensemble_10.png')
    print(f"Loading confidence map (ensemble) from: {confidence_ensemble_path}")
    confidence_ensemble = Image.open(confidence_ensemble_path)

    # Create a figure to display the results
    print("Creating figure for visualization...")
    plt.figure(figsize=(15, 10))

    # Display the original image
    plt.subplot(3, 3, 1)
    plt.imshow(np.array(original_image))
    plt.title('Original Image')
    plt.axis('off')

    # Display the segmentation results
    plt.subplot(3, 3, 4)
    plt.imshow(np.array(segmentation_disabled), cmap='gray')
    plt.title('Segmentation (No Noise)')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(np.array(segmentation_single_10), cmap='gray')
    plt.title('Segmentation (Single Noise, 0.1)')
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(np.array(segmentation_single_30), cmap='gray')
    plt.title('Segmentation (Single Noise, 0.3)')
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.imshow(np.array(confidence_disabled))
    plt.title('Confidence Map (No Noise)')
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.imshow(np.array(confidence_single_10))
    plt.title('Confidence Map (Single Noise, 0.1)')
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.imshow(np.array(confidence_ensemble))
    plt.title('Confidence Map (Ensemble Noise, 0.1)')
    plt.axis('off')

    # Display special results for ensemble
    plt.subplot(3, 3, 3)
    plt.imshow(np.array(segmentation_ensemble), cmap='gray')
    plt.title('Segmentation (Ensemble Noise)')
    plt.axis('off')

    # Add a title
    plt.suptitle('Neural Noise-Driven Segmentation Comparison', fontsize=16)
    plt.tight_layout()

    # Save the comparison image
    output_path = 'results_comparison.png'
    print(f"Saving comparison to: {output_path}")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Results comparison saved to '{output_path}'")

except Exception as e:
    print(f"Error during visualization: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Load the original test image
original_image = Image.open(os.path.join(results_dir, 'original_test_image.png'))

# Load the segmentation results
segmentation_disabled = Image.open(os.path.join(results_dir, 'segmentation_disabled_10.png'))
segmentation_single_10 = Image.open(os.path.join(results_dir, 'segmentation_single_10.png'))
segmentation_single_30 = Image.open(os.path.join(results_dir, 'segmentation_single_30.png'))
segmentation_ensemble = Image.open(os.path.join(results_dir, 'segmentation_ensemble_10.png'))

# Load the confidence maps
confidence_disabled = Image.open(os.path.join(results_dir, 'confidence_disabled_10.png'))
confidence_single_10 = Image.open(os.path.join(results_dir, 'confidence_single_10.png'))
confidence_single_30 = Image.open(os.path.join(results_dir, 'confidence_single_30.png'))
confidence_ensemble = Image.open(os.path.join(results_dir, 'confidence_ensemble_10.png'))

# Create a figure to display the results
plt.figure(figsize=(15, 10))

# Display the original image
plt.subplot(3, 3, 1)
plt.imshow(np.array(original_image))
plt.title('Original Image')
plt.axis('off')

# Display the segmentation results
plt.subplot(3, 3, 4)
plt.imshow(np.array(segmentation_disabled), cmap='gray')
plt.title('Segmentation (No Noise)')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(np.array(segmentation_single_10), cmap='gray')
plt.title('Segmentation (Single Noise, 0.1)')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(np.array(segmentation_single_30), cmap='gray')
plt.title('Segmentation (Single Noise, 0.3)')
plt.axis('off')

plt.subplot(3, 3, 7)
plt.imshow(np.array(confidence_disabled))
plt.title('Confidence Map (No Noise)')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.imshow(np.array(confidence_single_10))
plt.title('Confidence Map (Single Noise, 0.1)')
plt.axis('off')

plt.subplot(3, 3, 9)
plt.imshow(np.array(confidence_ensemble))
plt.title('Confidence Map (Ensemble Noise, 0.1)')
plt.axis('off')

# Display special results for ensemble
plt.subplot(3, 3, 3)
plt.imshow(np.array(segmentation_ensemble), cmap='gray')
plt.title('Segmentation (Ensemble Noise)')
plt.axis('off')

# Add a title
plt.suptitle('Neural Noise-Driven Segmentation Comparison', fontsize=16)
plt.tight_layout()

# Save the comparison image
plt.savefig('results_comparison.png', dpi=200, bbox_inches='tight')
print("Results comparison saved to 'results_comparison.png'")

# Show the plot
plt.show()
