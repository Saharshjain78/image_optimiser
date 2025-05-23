import os
import sys
import argparse
import subprocess
from pathlib import Path
import datetime

def run_command(cmd, desc=None):
    """Run a command and print output"""
    if desc:
        print(f"\n=== {desc} ===\n")
    
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Run advanced Naruto test")
    parser.add_argument("--model_path", type=str, default="models/noise_unet_model.pth",
                        help="Path to the model to use for testing")
    parser.add_argument("--naruto_image", type=str, 
                        default="test/HD-wallpaper-anime-naruto-running-naruto-grass-anime.jpg",
                        help="Path to Naruto test image")
    parser.add_argument("--output_dir", type=str, default="advanced_naruto_results",
                        help="Output directory")
    parser.add_argument("--optimal_ensemble_size", type=int, default=3,
                        help="Optimal ensemble size to use")
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return 1
        
    # Check if image exists
    if not os.path.exists(args.naruto_image):
        print(f"Error: Naruto image not found at {args.naruto_image}")
        return 1
    
    # Make Python command
    python_cmd = sys.executable
    
    # Run the test
    cmd = f"{python_cmd} advanced_naruto_test.py --model_path {args.model_path} --naruto_image {args.naruto_image} --output_dir {output_dir} --optimal_ensemble_size {args.optimal_ensemble_size}"
    
    success = run_command(cmd, "Running Advanced Naruto Test")
    
    if success:
        print(f"\n=== Advanced Naruto Test Completed Successfully ===")
        print(f"Results saved to {output_dir}")
    else:
        print(f"\n=== Advanced Naruto Test Failed ===")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
