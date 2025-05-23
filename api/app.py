from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
import uuid
import os
from PIL import Image
from io import BytesIO
import sys
import time

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import preprocess_image, save_segmentation, generate_confidence_map
from utils.simple_noise_injector import NeuralNoiseInjector
from models.noise_unet import NoiseAwareUNet

app = FastAPI(title="Neural Noise Segmentation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Model initialization
model = None

def load_model():
    global model
    try:
        model = NoiseAwareUNet(3, 1)
        model.load_state_dict(torch.load("models/noise_unet_model.pth"))
        model.eval()
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    noise_mode: str = Form("disabled"),  # Options: disabled, single, ensemble
    noise_strength: float = Form(0.1),
    confidence_map: bool = Form(False)
):
    # Validate the model is loaded
    if model is None:
        if not load_model():
            return JSONResponse(
                status_code=500,
                content={"error": "Model not loaded. Please check the server logs."}
            )
    
    # Validate noise_mode
    if noise_mode not in ["disabled", "single", "ensemble"]:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid noise_mode: {noise_mode}. Must be one of: disabled, single, ensemble"}
        )
    
    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    
    try:
        # Read and save input image
        image_data = await file.read()
        input_path = os.path.join(UPLOAD_DIR, f"{request_id}_input.jpg")
        with open(input_path, "wb") as f:
            f.write(image_data)
        
        # Process image
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_tensor = preprocess_image(image)
        
        # Configure noise parameters
        noise_injector = NeuralNoiseInjector(noise_scale=noise_strength)
        model.noise_injector = noise_injector
        
        # Generate segmentation with specified noise mode
        start_time = time.time()
        with torch.no_grad():
            output = model(image_tensor, noise_mode=noise_mode)
        inference_time = time.time() - start_time
        
        # Apply sigmoid activation for binary segmentation
        output = torch.sigmoid(output)
        
        # Save segmentation result
        result_path = os.path.join(RESULTS_DIR, f"{request_id}_segmented.png")
        save_segmentation(output, result_path)
        
        # Generate confidence map if requested
        confidence_path = None
        if confidence_map:
            confidence_path = os.path.join(RESULTS_DIR, f"{request_id}_confidence.png")
            generate_confidence_map(output, confidence_path, noise_injector)
        
        return {
            "request_id": request_id,
            "segmentation_url": f"/results/{request_id}_segmented.png",
            "confidence_map_url": f"/results/{request_id}_confidence.png" if confidence_map else None,
            "applied_noise_mode": noise_mode,
            "noise_strength": noise_strength,
            "inference_time_ms": round(inference_time * 1000, 2)
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing image: {str(e)}"}
        )

@app.get("/results/{filename}")
async def get_result(filename: str):
    file_path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse(
        status_code=404,
        content={"error": f"File not found: {filename}"}
    )

@app.post("/segment_with_uncertainty")
async def segment_with_uncertainty(
    file: UploadFile = File(...),
    iterations: int = Form(5)
):
    """Monte Carlo Dropout with noise to estimate segmentation uncertainty"""
    # Validate the model is loaded
    if model is None:
        if not load_model():
            return JSONResponse(
                status_code=500,
                content={"error": "Model not loaded. Please check the server logs."}
            )
    
    # Generate unique ID for this request
    request_id = str(uuid.uuid4())
    
    try:
        # Read and save input image
        image_data = await file.read()
        input_path = os.path.join(UPLOAD_DIR, f"{request_id}_input.jpg")
        with open(input_path, "wb") as f:
            f.write(image_data)
        
        # Process image
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_tensor = preprocess_image(image)
        
        # Configure noise parameters
        noise_injector = NeuralNoiseInjector(noise_scale=0.1)
        model.noise_injector = noise_injector
        
        # Run multiple iterations with different noise patterns
        outputs = []
        for i in range(iterations):
            with torch.no_grad():
                # Reset noise injector iteration counter
                noise_injector.reset_iteration()
                # Run inference with single noise pattern
                output = model(image_tensor, noise_mode="single")
                output = torch.sigmoid(output)
                outputs.append(output)
        
        # Calculate mean prediction
        mean_output = torch.mean(torch.stack(outputs), dim=0)
        
        # Calculate variance (uncertainty)
        variance = torch.var(torch.stack(outputs), dim=0)
        
        # Save mean segmentation result
        mean_path = os.path.join(RESULTS_DIR, f"{request_id}_mean.png")
        save_segmentation(mean_output, mean_path)
        
        # Save variance/uncertainty map
        variance_path = os.path.join(RESULTS_DIR, f"{request_id}_variance.png")
        save_segmentation(variance, variance_path)
        
        return {
            "request_id": request_id,
            "mean_segmentation_url": f"/results/{request_id}_mean.png",
            "uncertainty_map_url": f"/results/{request_id}_variance.png",
            "iterations": iterations
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing uncertainty: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)