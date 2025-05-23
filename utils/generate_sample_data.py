import os
import numpy as np
import cv2
from PIL import Image, ImageDraw
import random
import argparse

def create_synthetic_dataset(output_dir, num_samples=100, image_size=(256, 256)):
    """
    Generate a synthetic dataset for segmentation with random shapes.
    
    Args:
        output_dir (str): Directory to save the dataset
        num_samples (int): Number of image-mask pairs to generate
        image_size (tuple): Size of the images (width, height)
    """
    # Create directories for images and masks
    images_dir = os.path.join(output_dir, 'images')
    masks_dir = os.path.join(output_dir, 'masks')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Create base image (random background)
        image = Image.new('RGB', image_size, color=(
            random.randint(100, 240),
            random.randint(100, 240),
            random.randint(100, 240)
        ))
        draw_img = ImageDraw.Draw(image)
        
        # Create mask (black background)
        mask = Image.new('L', image_size, color=0)
        draw_mask = ImageDraw.Draw(mask)
        
        # Number of shapes to draw
        num_shapes = random.randint(1, 5)
        
        for _ in range(num_shapes):
            # Choose shape type: circle, rectangle, or polygon
            shape_type = random.choice(['circle', 'rectangle', 'polygon'])
            
            # Random color for the image shape
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            
            # Create shapes
            if shape_type == 'circle':
                # Random circle
                center_x = random.randint(0, image_size[0])
                center_y = random.randint(0, image_size[1])
                radius = random.randint(10, min(image_size) // 4)
                
                bbox = [
                    center_x - radius, center_y - radius,
                    center_x + radius, center_y + radius
                ]
                
                draw_img.ellipse(bbox, fill=color)
                draw_mask.ellipse(bbox, fill=255)  # White in mask
                
            elif shape_type == 'rectangle':
                # Random rectangle
                x1 = random.randint(0, image_size[0] - 10)
                y1 = random.randint(0, image_size[1] - 10)
                x2 = random.randint(x1 + 10, min(image_size[0], x1 + 100))
                y2 = random.randint(y1 + 10, min(image_size[1], y1 + 100))
                
                bbox = [x1, y1, x2, y2]
                
                draw_img.rectangle(bbox, fill=color)
                draw_mask.rectangle(bbox, fill=255)
                
            else:  # polygon
                # Random polygon
                num_points = random.randint(3, 8)
                points = []
                
                center_x = random.randint(image_size[0] // 4, 3 * image_size[0] // 4)
                center_y = random.randint(image_size[1] // 4, 3 * image_size[1] // 4)
                
                for j in range(num_points):
                    angle = 2 * np.pi * j / num_points
                    radius = random.randint(20, min(image_size) // 6)
                    x = center_x + int(radius * np.cos(angle))
                    y = center_y + int(radius * np.sin(angle))
                    points.append((x, y))
                
                draw_img.polygon(points, fill=color)
                draw_mask.polygon(points, fill=255)
        
        # Add noise to the image
        img_array = np.array(image)
        noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        
        # Save the images
        Image.fromarray(noisy_img).save(os.path.join(images_dir, f'image_{i:05d}.png'))
        mask.save(os.path.join(masks_dir, f'image_{i:05d}.png'))
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_samples} samples")
    
    print(f"Dataset generated successfully at {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for segmentation")
    parser.add_argument('--output_dir', type=str, default='data/synthetic',
                       help='Directory to save the dataset')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of image-mask pairs to generate')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                       help='Image dimensions (width height)')
    
    args = parser.parse_args()
    
    create_synthetic_dataset(args.output_dir, args.num_samples, tuple(args.image_size))
