import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import io
from skimage import img_as_ubyte
import random

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, target_size=(256, 256)):
        """Dataset for image segmentation with optional transforms"""
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_size = target_size
        
        # Get all image files (assumes image and mask have same filename)
        self.images = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure mask exists (some datasets use different extensions)
        mask_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        mask = None
        for ext in mask_extensions:
            potential_mask = os.path.splitext(mask_path)[0] + ext
            if os.path.exists(potential_mask):
                mask = cv2.imread(potential_mask, cv2.IMREAD_GRAYSCALE)
                break
        
        if mask is None:
            # Create a blank mask if no mask file exists
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
        # Resize to target size
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize mask to [0, 1]
        if mask.max() > 1:
            mask = mask / 255.0
            
        # Convert to PyTorch tensors
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # Add channel dimension
        
        # Apply any additional transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
            
        return image, mask

def get_data_loaders(data_dir, batch_size=8, train_val_split=0.8, target_size=(256, 256)):
    """Create train and validation data loaders"""
    images_dir = os.path.join(data_dir, 'images')
    masks_dir = os.path.join(data_dir, 'masks')
    
    # List all images
    all_images = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle and split into train/val
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_val_split)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Create custom datasets that only use certain filenames
    class TrainDataset(SegmentationDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.images = train_images
            
    class ValDataset(SegmentationDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.images = val_images
    
    # Create the datasets
    train_dataset = TrainDataset(images_dir, masks_dir, target_size=target_size)
    val_dataset = ValDataset(images_dir, masks_dir, target_size=target_size)
      # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess an image for model input with robust handling of various formats"""
    # Handle PIL image conversion
    if isinstance(image, Image.Image):
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Handle transparency (RGBA)
        if hasattr(image, 'mode') and image.mode == 'RGBA':
            # Create a white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            # Paste the image using alpha channel as mask
            background.paste(image, mask=image.split()[3])
            image = background
            
        # Convert PIL to numpy
        img_array = np.array(image)
    else:
        img_array = image
    
    # Convert to RGB if grayscale
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    
    # Resize to target size
    img_array = cv2.resize(img_array, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to PyTorch tensor with batch dimension [B, C, H, W]
    img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float().unsqueeze(0)
    
    # Normalize to [0, 1]
    img_tensor = img_tensor / 255.0
    
    return img_tensor, target_size
    
    return img_tensor

def save_segmentation(segmentation, filename):
    """Save segmentation mask to file"""
    # Convert tensor to numpy if needed
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    
    # If 4D tensor [B, C, H, W], take first item from batch
    if len(segmentation.shape) == 4:
        segmentation = segmentation[0]
    
    # If 3D tensor [C, H, W], take first channel
    if len(segmentation.shape) == 3:
        segmentation = segmentation[0]
    
    # Convert to uint8 format (0-255)
    segmentation = (segmentation * 255).astype(np.uint8)
    
    # Save the image
    cv2.imwrite(filename, segmentation)
    
    return filename

def generate_confidence_map(segmentations, colormap='viridis'):
    """Generate a colored confidence map from multiple segmentations"""
    # Average segmentations
    avg_segmentation = np.mean(segmentations, axis=0)
    
    # Calculate std deviation as uncertainty measure
    std_dev = np.std(segmentations, axis=0)
    
    # Apply colormap
    cm = plt.get_cmap(colormap)
    colored_map = cm(std_dev)[:, :, :3]  # Remove alpha channel
    
    # Convert to uint8
    colored_map = (colored_map * 255).astype(np.uint8)
    
    return colored_map, avg_segmentation

def display_image(image, title=None):
    """Display an image using matplotlib and return bytes"""
    plt.figure(figsize=(8, 8))
    
    # Convert tensor to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Handle different image formats
    if len(image.shape) == 4:  # [B, C, H, W]
        image = image[0]  # Take first item from batch
        
    if len(image.shape) == 3:
        # If image has 3 channels (RGB)
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))  # CHW to HWC
        # If image has 1 channel (mask/segmentation)
        elif image.shape[0] == 1:
            image = image[0]
    
    plt.imshow(image)
    if title:
        plt.title(title)

# Enhanced version of preprocess_image is now unified with the main implementation

def save_segmentation(output, save_path):
    """Save model output as a segmentation mask image"""
    # Convert from tensor to numpy
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
        
    # Extract first item from batch and first channel
    if output.ndim == 4:
        output = output[0, 0]
    elif output.ndim == 3:
        output = output[0]
        
    # Apply sigmoid for binary segmentation if necessary
    if output.min() < 0 or output.max() > 1:
        output = 1.0 / (1.0 + np.exp(-output))  # sigmoid
        
    # Scale to [0, 255] and convert to uint8
    output = (output * 255).astype(np.uint8)
    
    # Save image
    cv2.imwrite(save_path, output)
    
def generate_confidence_map(output, save_path, noise_injector):
    """Generate a confidence map based on multiple noisy inferences"""
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
        
    # Extract first item from batch and first channel
    if output.ndim == 4:
        output = output[0, 0]
    elif output.ndim == 3:
        output = output[0]
        
    # Scale to [0, 255] and convert to uint8
    output = (output * 255).astype(np.uint8)
    
    # Apply color map for visualization (hot colormap)
    confidence_map = cv2.applyColorMap(output, cv2.COLORMAP_JET)
    
    # Save image
    cv2.imwrite(save_path, confidence_map)
    
def display_image(image, title=None):
    """Display an image with optional title"""
    plt.figure(figsize=(8, 8))
    if isinstance(image, torch.Tensor):
        # Convert tensor to numpy
        image = image.detach().cpu().numpy()
        
        # If image has batch dimension, take first item
        if image.ndim == 4:
            image = image[0]
            
        # If image has 3 channels (RGB)
        if image.shape[0] == 3:
            # Move channels to the end for plt.imshow()
            image = np.transpose(image, (1, 2, 0))
            
            # Denormalize if needed (assuming standard normalization)
            if image.min() < 0 or image.max() > 1:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = image * std + mean
                image = np.clip(image, 0, 1)
        # If image has 1 channel (mask/segmentation)
        elif image.shape[0] == 1:
            image = image[0]
    
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    
    # Create a BytesIO object to store the image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    
    return buf