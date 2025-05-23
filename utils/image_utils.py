import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import os
import random

def create_placeholder_image(width=512, height=512, text="Sample Image", color_mode="random"):
    """Create a placeholder image with text."""
    # Choose background color
    if color_mode == "random":
        bg_color = (
            random.randint(50, 200),
            random.randint(50, 200),
            random.randint(50, 200)
        )
    else:
        bg_color = (73, 109, 137)  # Default blue-gray
    
    # Create a new image with background color
    image = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(image)
    
    # Try to use a system font, fall back to default if not available
    try:
        # Try to find a font on the system
        font_path = None
        system_fonts = [
            # Windows fonts
            "C:\\Windows\\Fonts\\Arial.ttf",
            "C:\\Windows\\Fonts\\Verdana.ttf",
            # Linux fonts
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            # macOS fonts
            "/System/Library/Fonts/Helvetica.ttc"
        ]
        
        for path in system_fonts:
            if os.path.exists(path):
                font_path = path
                break
        
        if font_path:
            # Calculate a reasonable font size (1/10 of the smallest dimension)
            font_size = min(width, height) // 10
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Use default font if no system font is found
            font = ImageFont.load_default()
            
    except Exception:
        # Fall back to default font if there's any error
        font = ImageFont.load_default()
    
    # Calculate text position to center it
    text_width = draw.textlength(text, font=font) if hasattr(draw, 'textlength') else font.getlength(text)
    text_position = ((width - text_width) // 2, height // 2)
    
    # Draw text with shadow for better visibility
    shadow_color = (bg_color[0] // 2, bg_color[1] // 2, bg_color[2] // 2)
    draw.text((text_position[0] + 2, text_position[1] + 2), text, font=font, fill=shadow_color)
    draw.text(text_position, text, font=font, fill=(255, 255, 255))
    
    # Add a border
    border_width = 4
    border_color = (255, 255, 255, 128)  # Semi-transparent white
    draw.rectangle(
        [(border_width, border_width), (width - border_width, height - border_width)],
        outline=border_color,
        width=border_width
    )
    
    # Add a grid pattern for visual reference
    grid_spacing = 64
    grid_color = (bg_color[0] ^ 32, bg_color[1] ^ 32, bg_color[2] ^ 32)  # XOR to create contrast
    
    for x in range(0, width, grid_spacing):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    
    for y in range(0, height, grid_spacing):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)
    
    return image

def get_sample_image(category, size=(512, 512)):
    """Get a sample image or create a placeholder if not available."""
    # Define paths for sample images based on category
    sample_paths = {
        "Nature": "test_images/nature_sample.jpg",
        "Portrait": "test_images/portrait_sample.jpg",
        "Urban": "test_images/urban_sample.jpg",
        "Medical": "test_images/medical_sample.jpg",
        "Satellite": "test_images/satellite_sample.jpg"
    }
    
    # Try to load the requested sample image
    if category in sample_paths and os.path.exists(sample_paths[category]):
        try:
            image = Image.open(sample_paths[category]).convert("RGB")
            # Resize if needed
            if image.size != size:
                image = image.resize(size, Image.LANCZOS)
            return image
        except Exception:
            # Fall back to placeholder if image can't be loaded
            pass
    
    # Create a placeholder with the category name
    return create_placeholder_image(size[0], size[1], f"Sample {category} Image")

def image_to_bytes(image, format="PNG"):
    """Convert a PIL Image to bytes for download."""
    buf = io.BytesIO()
    image.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()

def create_sample_images():
    """Create sample images for testing if they don't exist."""
    categories = ["Nature", "Portrait", "Urban", "Medical", "Satellite"]
    
    # Ensure test_images directory exists
    os.makedirs("test_images", exist_ok=True)
    
    for category in categories:
        file_path = f"test_images/{category.lower()}_sample.jpg"
        if not os.path.exists(file_path):
            # Create a placeholder image
            img = create_placeholder_image(512, 512, f"Sample {category} Image", "random")
            img.save(file_path, "JPEG")
