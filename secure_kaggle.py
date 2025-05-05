#!/usr/bin/env python3
"""
Secure handling of Kaggle API credentials

This script helps manage Kaggle API credentials in a secure way
to avoid committing secrets to GitHub.
"""

import os
import json
import getpass
import pathlib
import sys

def setup_kaggle_credentials():
    """
    Set up Kaggle credentials interactively or from environment variables.
    If KAGGLE_USERNAME and KAGGLE_KEY env vars are set, use those.
    Otherwise, prompt the user for their credentials.
    
    Returns:
        pathlib.Path: Path to the created Kaggle JSON file
    """
    home = pathlib.Path.home()
    kaggle_dir = home / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_file = kaggle_dir / 'kaggle.json'
    
    if kaggle_file.exists():
        print(f"✓ Kaggle credentials already exist at {kaggle_file}")
        return kaggle_file
    
    # Check for environment variables
    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')
    
    # If environment variables aren't set, prompt user
    if not username or not key:
        print("Kaggle API credentials not found!")
        print("Please enter your Kaggle credentials (from kaggle.com/account)")
        
        username = input("Kaggle username: ")
        key = getpass.getpass("Kaggle API key: ")
    
    # Create credentials file
    credentials = {"username": username, "key": key}
    
    with open(kaggle_file, 'w') as f:
        json.dump(credentials, f)
    
    os.chmod(kaggle_file, 0o600)  # Make file readable only by owner
    print(f"✓ Kaggle credentials saved to {kaggle_file}")
    
    # Create local copy for convenience (will be gitignored)
    local_kaggle = pathlib.Path('kaggle.json')
    with open(local_kaggle, 'w') as f:
        json.dump(credentials, f)
    
    print(f"✓ Local copy created at {local_kaggle} (this file is gitignored)")
    
    return kaggle_file

def get_kaggle_file():
    """
    Find and return the path to the Kaggle JSON file.
    Checks in the following order:
    1. Local kaggle.json
    2. ~/.kaggle/kaggle.json
    3. Environment variables
    
    Returns:
        pathlib.Path: Path to the Kaggle JSON file
    """
    # Check for local file
    local_file = pathlib.Path('kaggle.json')
    if local_file.exists():
        return local_file
    
    # Check for home directory file
    home_file = pathlib.Path.home() / '.kaggle' / 'kaggle.json'
    if home_file.exists():
        return home_file
    
    # Check for environment variables
    if os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):
        # Create temporary file
        username = os.environ.get('KAGGLE_USERNAME')
        key = os.environ.get('KAGGLE_KEY')
        
        temp_file = pathlib.Path('temp_kaggle.json')
        with open(temp_file, 'w') as f:
            json.dump({"username": username, "key": key}, f)
        
        return temp_file
    
    # No credentials found
    print("Kaggle credentials not found!")
    print("Please run this script with --setup or set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
    sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage Kaggle API credentials securely")
    parser.add_argument('--setup', action='store_true', help='Set up Kaggle credentials interactively')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_kaggle_credentials()
    else:
        kaggle_file = get_kaggle_file()
        print(f"Using Kaggle credentials from: {kaggle_file}") 