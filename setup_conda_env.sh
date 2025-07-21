#!/bin/bash

# Install PyTorch and related packages
# Attempting to install PyTorch with CUDA support via pip
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Verify installation (optional, but good practice)
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
pip freeze > installed_packages.txt
echo "Environment setup complete. Installed packages listed in installed_packages.txt"
