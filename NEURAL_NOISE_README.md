# Neural Noise-Driven Dynamic Segmentation System

## Project Overview

This project implements a novel approach to image segmentation that deliberately introduces controlled neural noise as a feature rather than a bug. The Neural Noise-Driven Dynamic Segmentation System uses noise to enhance segmentation performance, especially in edge cases and ambiguous regions.

## Key Features

- **Controlled Neural Noise**: Deliberately injects noise patterns into feature maps to enhance segmentation
- **Multiple Noise Modes**: 
  - `disabled`: No noise (standard segmentation)
  - `single`: Single noise pattern for better edge detection
  - `ensemble`: Multiple noise patterns for uncertainty quantification
- **Adaptive Noise Strength**: Configurable noise strength for different segmentation tasks
- **Web Interface**: Streamlit app for interactive segmentation with noise parameter adjustment
- **REST API**: FastAPI server for programmatic access to the segmentation system

## How Neural Noise Enhances Segmentation

Neural noise serves several important functions in our segmentation system:

1. **Breaking Symmetry**: Helps the model escape local optima in ambiguous regions
2. **Edge Enhancement**: Improves boundary detection by emphasizing edge features
3. **Uncertainty Quantification**: Multiple noise patterns provide natural uncertainty estimates
4. **Ensemble Effects**: A single model produces ensemble-like results via different noise patterns
5. **Adaptability**: Better generalization to unseen data and edge cases

## System Architecture

The system consists of three main components:

1. **Noise-Aware UNet Model**: A deep learning segmentation model that incorporates noise injection
2. **FastAPI Backend**: Provides REST API endpoints for segmentation with various noise modes
3. **Streamlit Frontend**: User-friendly web interface for interactive segmentation

## Installation and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-noise-segmentation.git
cd neural-noise-segmentation

# Install dependencies
pip install -r requirements.txt

# Run the system (starts both the FastAPI server and Streamlit app)
python run.py
```

## Usage

### Web Interface

Access the Streamlit app at http://localhost:8501 and:

1. Upload an image
2. Select a noise mode (disabled, single, ensemble)
3. Adjust noise strength (0.01-0.5)
4. Click "Apply Segmentation"

### API Usage

```python
import requests

# Prepare the image for API upload
with open('path/to/image.jpg', 'rb') as img_file:
    files = {'file': ('image.jpg', img_file, 'image/jpeg')}
    data = {
        'noise_mode': 'single',  # Options: disabled, single, ensemble
        'noise_strength': '0.1',
        'confidence_map': 'true'
    }
    
    # API call
    response = requests.post(
        'http://localhost:8000/upload',
        files=files,
        data=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Segmentation URL: {result['segmentation_url']}")
        print(f"Confidence Map URL: {result['confidence_map_url']}")
        print(f"Inference Time: {result['inference_time_ms']} ms")
```

## Results

Our tests demonstrate the effectiveness of neural noise in image segmentation:

- **Without Noise**: Standard segmentation provides a baseline but struggles with ambiguous boundaries
- **Single Noise (Low)**: Subtle enhancements to edge detection with minimal impact on clear regions
- **Single Noise (Medium)**: Greater boundary enhancement with slight noise-induced variations
- **Single Noise (High)**: Maximum boundary enhancement but potential over-segmentation
- **Ensemble Noise**: Robust segmentation with uncertainty estimation

## Project Structure

```
├── api/                  # FastAPI backend
│   └── app.py            # API endpoints
├── app/                  # Streamlit frontend
│   └── app.py            # Web interface
├── data/                 # Training and test data
│   └── synthetic/        # Synthetic training data
├── models/               # Model architecture and weights
│   └── noise_unet.py     # Noise-aware UNet model
├── utils/                # Utility functions
│   ├── data_utils.py     # Data processing utilities
│   └── simple_noise_injector.py  # Neural noise injection
├── training/             # Training scripts
│   └── train.py          # Model training script
├── run.py                # System startup script
└── requirements.txt      # Project dependencies
```

## Future Improvements

- Implement additional noise patterns (Perlin, simplex, etc.)
- Add support for different segmentation architectures
- Explore adaptive noise strength based on image complexity
- Enhance uncertainty visualization
- Implement transfer learning to fine-tune on specific domains

## License

MIT

## Acknowledgements

This project builds on concepts from uncertainty estimation, ensemble methods, and noise robustness in deep learning.
