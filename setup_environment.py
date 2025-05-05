#!/usr/bin/env python3
"""
Jigsaw Unintended Bias Audit - Environment Setup
This script sets up the environment for the Jigsaw Unintended Bias Audit project.
It checks for required dependencies, configures the Kaggle API, and downloads the dataset.
"""

import os
import sys
import subprocess
import pathlib
import json
import zipfile
import shutil
import warnings
import argparse
import secure_kaggle

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def install_dependencies():
    """Install required dependencies if not already installed."""
    print("Checking and installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("⚠ Error installing dependencies. Please check the requirements file.")
        sys.exit(1)

def setup_kaggle_api(kaggle_json_path=None):
    """
    Configure Kaggle API with credentials
    """
    home = pathlib.Path.home()
    kaggle_path = home / '.kaggle'
    kaggle_path.mkdir(exist_ok=True)
    
    # If a path is provided, copy the kaggle.json file to the .kaggle directory
    if kaggle_json_path:
        if not os.path.exists(kaggle_json_path):
            raise FileNotFoundError(f"Kaggle API key file not found at {kaggle_json_path}")
        
        # Copy the file to the .kaggle directory
        target_path = kaggle_path / 'kaggle.json'
        shutil.copy(kaggle_json_path, target_path)
        os.chmod(target_path, 0o600)  # Set proper permissions
        print(f"✓ Kaggle API key copied to {target_path}")
        return True
    
    # Try to find kaggle credentials using our secure handler
    try:
        kaggle_file = secure_kaggle.get_kaggle_file()
        if not (kaggle_path / 'kaggle.json').exists():
            # Copy the file to the .kaggle directory
            target_path = kaggle_path / 'kaggle.json'
            shutil.copy(kaggle_file, target_path)
            os.chmod(target_path, 0o600)  # Set proper permissions
        print(f"✓ Using Kaggle API credentials from {kaggle_file}")
        return True
    except:
        # If secure_kaggle fails, prompt the user to set up credentials
        print("No Kaggle API key found. Please run 'python secure_kaggle.py --setup' to configure credentials.")
        print("Alternatively, go to https://www.kaggle.com/settings and create a new API token,")
        print("then run this script again with --kaggle_json path/to/your/kaggle.json")
        raise FileNotFoundError("Kaggle API key not found")
    
    return True

def download_jigsaw_data(data_dir):
    """
    Download the Jigsaw Unintended Bias in Toxicity Classification dataset
    """
    data_dir = pathlib.Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    
    COMP = "jigsaw-unintended-bias-in-toxicity-classification"
    
    # Download the dataset
    print(f"Downloading Jigsaw dataset to {data_dir}...")
    subprocess.run(['kaggle', 'competitions', 'download', '-c', COMP, '-p', str(data_dir), '--force'], check=True)
    
    # Extract the ZIP file
    zip_path = list(data_dir.glob(f"{COMP}*.zip"))[0]
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    print(f"Files downloaded to {data_dir}:")
    for file in data_dir.iterdir():
        print(f"  - {file.name}")
    
    return True

def create_directory_structure():
    """Create the directory structure for the project."""
    print("\nCreating directory structure...")
    
    # Create directories
    dirs = ["data", "logs", "models", "artifacts", "figs"]
    for dir_name in dirs:
        pathlib.Path(dir_name).mkdir(exist_ok=True)
    
    print("✓ Directory structure created:")
    for dir_name in dirs:
        print(f"  - {dir_name}/")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Setup environment for Jigsaw Unintended Bias Audit")
    parser.add_argument('--kaggle_json', type=str, help='Path to kaggle.json API key file', default=None)
    parser.add_argument('--data_dir', type=str, help='Directory to store the dataset', default='./data')
    parser.add_argument('--setup_kaggle', action='store_true', help='Run kaggle credentials setup interactively')
    
    args = parser.parse_args()
    
    if args.setup_kaggle:
        # Run the secure kaggle setup
        secure_kaggle.setup_kaggle_credentials()
    
    # Setup Kaggle API
    setup_kaggle_api(args.kaggle_json)
    
    # Download Jigsaw data
    download_jigsaw_data(args.data_dir)
    
    print("\n✓ Environment setup complete!")
    print(f"Dataset downloaded to {args.data_dir}")
    print("\nTo run the analysis:")
    print("1. Create and activate a virtual environment:")
    print("   python -m venv venv")
    print("   venv\\Scripts\\activate  # On Windows")
    print("   source venv/bin/activate  # On Linux/Mac")
    print("2. Install required packages:")
    print("   pip install -r requirements.txt")
    print("3. Run the Jupyter notebook:")
    print("   jupyter notebook Jigsaw_Unintended_Bias_Audit.ipynb")

if __name__ == "__main__":
    main() 