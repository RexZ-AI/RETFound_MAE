#!/bin/bash

# Attempt to install PyTorch, torchvision, torchaudio with CUDA globally using pip
echo "Attempting global installation of PyTorch, torchvision, and torchaudio with CUDA support..."
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
if [ $? -ne 0 ]; then
    echo "PyTorch global installation failed. Skipping installation of other dependencies."
    # Attempt to verify PyTorch installation even if it reported an error, for debugging
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
    echo "Global pip install attempt complete."
    exit 1
fi

# Install other dependencies
echo "PyTorch global installation successful. Installing other dependencies globally..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Global installation of other dependencies from requirements.txt failed."
fi

# Verify installation
echo "Verifying final global installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo "Global pip install attempt complete."
