#!/bin/bash

# Create Conda environment
echo "Attempting to create Conda environment..."
conda create -n retfound python=3.11.0 -y
if [ $? -ne 0 ]; then
    echo "Conda environment creation failed. Exiting."
    exit 1
fi

echo "Activating Conda environment..."
conda activate retfound
if [ $? -ne 0 ]; then
    echo "Conda environment activation failed. Exiting."
    exit 1
fi

# Install PyTorch and related packages first
echo "Installing PyTorch and related packages..."
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
if [ $? -ne 0 ]; then
    echo "PyTorch installation failed. Skipping installation of other dependencies."
    # Verify installation (optional, but good practice)
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
    echo "Conda environment setup attempt complete."
    exit 1
fi

# If PyTorch installation is successful, install other dependencies
echo "PyTorch installation successful. Installing other dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Installation of other dependencies failed."
fi

# Verify installation (optional, but good practice)
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo "Conda environment setup attempt complete."
