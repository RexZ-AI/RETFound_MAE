#!/bin/bash

# Create a Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv retfound_env
if [ $? -ne 0 ]; then
    echo "Python virtual environment creation failed. Exiting."
    exit 1
fi

# Activate the virtual environment
echo "Activating Python virtual environment..."
source retfound_env/bin/activate
if [ $? -ne 0 ]; then
    echo "Python virtual environment activation failed. Exiting."
    exit 1
fi

# Install PyTorch, torchvision, torchaudio with CUDA
echo "Installing PyTorch, torchvision, and torchaudio with CUDA support..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
if [ $? -ne 0 ]; then
    echo "PyTorch installation failed. Skipping installation of other dependencies."
    # Attempt to verify PyTorch installation even if it reported an error, for debugging
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
    echo "Python virtual environment setup attempt complete."
    exit 1
fi

# Install other dependencies
echo "PyTorch installation successful. Installing other dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Installation of other dependencies from requirements.txt failed."
fi

# Verify installation
echo "Verifying final installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo "Python virtual environment setup attempt complete."
