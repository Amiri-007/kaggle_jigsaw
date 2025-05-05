#!/bin/bash

echo "========================================================"
echo "Jigsaw Unintended Bias Audit - Unix Setup"
echo "========================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH."
    echo "Please install Python from https://www.python.org/downloads/"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run environment setup
echo "Setting up project environment..."
python setup_environment.py

echo "========================================================"
echo "Setup complete!"
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo "========================================================" 