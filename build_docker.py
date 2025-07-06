"""
Script to build Docker image from the latest MLflow model
Author: alpian_khairi_C1BO
"""

import os
import subprocess
import sys
from datetime import datetime

def get_latest_run_id():
    """Get the latest run ID from mlruns directory"""
    experiment_dir = "mlruns/0"
    
    if not os.path.exists(experiment_dir):
        print("❌ No mlruns/0 directory found. Please run the MLflow project first.")
        return None
    
    # Get all run directories (excluding meta.yaml)
    run_dirs = [d for d in os.listdir(experiment_dir) 
                if os.path.isdir(os.path.join(experiment_dir, d)) and d != "meta.yaml"]
    
    if not run_dirs:
        print("❌ No run directories found in mlruns/0")
        return None
    
    # Sort by creation time and get the latest
    run_paths = [(d, os.path.getctime(os.path.join(experiment_dir, d))) for d in run_dirs]
    latest_run = max(run_paths, key=lambda x: x[1])[0]
    
    print(f"✅ Found latest run: {latest_run}")
    return latest_run

def build_docker_image():
    """Build Docker image from the latest model"""
    
    print("=" * 60)
    print("MLflow Docker Image Builder - alpian_khairi_C1BO")
    print("=" * 60)
    
    # Get the latest run ID
    latest_run_id = get_latest_run_id()
    if not latest_run_id:
        return 1
    
    # Construct the model path
    model_path = f"mlruns/0/{latest_run_id}/artifacts/model"
    
    # Verify the model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return 1
    
    print(f"📦 Building Docker image from: {model_path}")
    
    # Docker image names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name_latest = "alpian-khairi/diabetes-classification-model:latest"
    image_name_timestamped = f"alpian-khairi/diabetes-classification-model:{timestamp}"
    
    # Build Docker image
    cmd = [
        "mlflow", "models", "build-docker",
        "-m", model_path,
        "-n", image_name_latest,
        "-n", image_name_timestamped
    ]
    
    print(f"🔨 Running command: {' '.join(cmd)}")
    print("This may take a few minutes...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Docker image built successfully!")
            print(f"\nImage tags created:")
            print(f"  - {image_name_latest}")
            print(f"  - {image_name_timestamped}")
            
            print(f"\n📋 To test the image locally:")
            print(f"docker run -p 8080:8080 {image_name_latest}")
            
            print(f"\n📋 To test predictions:")
            print(f"curl -X POST http://localhost:8080/invocations \\")
            print(f"  -H 'Content-Type: application/json' \\")
            print(f"  -d '{{\"inputs\": [[5.1, 3.5, 1.4, 0.2]]}}'")
            
            return 0
        else:
            print("❌ Docker build failed!")
            print(f"\nError output:")
            print(result.stderr)
            print(f"\nStandard output:")
            print(result.stdout)
            return 1
            
    except Exception as e:
        print(f"❌ Failed to build Docker image: {e}")
        return 1

def list_available_models():
    """List all available models for reference"""
    print("\n📁 Available models:")
    experiment_dir = "mlruns/0"
    
    if os.path.exists(experiment_dir):
        run_dirs = [d for d in os.listdir(experiment_dir) 
                    if os.path.isdir(os.path.join(experiment_dir, d)) and d != "meta.yaml"]
        
        for run_dir in sorted(run_dirs):
            model_path = os.path.join(experiment_dir, run_dir, "artifacts", "model")
            if os.path.exists(model_path):
                creation_time = datetime.fromtimestamp(
                    os.path.getctime(os.path.join(experiment_dir, run_dir))
                ).strftime("%Y-%m-%d %H:%M:%S")
                print(f"  - {run_dir} (created: {creation_time})")
                print(f"    Path: mlruns/0/{run_dir}/artifacts/model")

def main():
    """Main function"""
    
    # First list available models
    list_available_models()
    
    # Build Docker image from latest model
    return_code = build_docker_image()
    
    if return_code == 0:
        print("\n🎉 Docker image build completed successfully!")
    
    return return_code

if __name__ == "__main__":
    sys.exit(main())