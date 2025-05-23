# PowerShell script to run the Neural Noise Segmentation App
Write-Host "Starting Neural Noise Segmentation App..." -ForegroundColor Green

# Navigate to the project directory
Set-Location -Path "c:\Users\Asus\Documents\Projects\image_optimisation"

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "Found $pythonVersion" -ForegroundColor Cyan
} catch {
    Write-Host "Python not found. Please install Python 3.7 or later." -ForegroundColor Red
    Exit
}

# Install required dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
python -m pip install -r requirements.txt

# Ensure test images exist
Write-Host "Ensuring sample images exist..." -ForegroundColor Yellow
python -c "import sys; sys.path.append('.'); from utils.image_utils import create_sample_images; create_sample_images()"

# Run the app
Write-Host "Starting Streamlit app..." -ForegroundColor Green
python -m streamlit run streamlit_app.py

# Keep the window open
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
