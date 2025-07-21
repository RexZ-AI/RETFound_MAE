# Guide to run RETFound Demo in Google Colab

This guide provides steps to set up the RETFound project in a Google Colab environment, download a demo dataset (IDRiD as an example), and run a fine-tuning task.

## 1. Setup Colab Runtime

*   Make sure your Colab runtime is set to use a GPU.
*   Go to **Runtime** -> **Change runtime type** -> Select **GPU** from the Hardware accelerator dropdown.

## 2. Clone Repository and Navigate to Directory

Copy and paste the following into a Colab code cell and run it:

```bash
!git clone https://github.com/rmaphoh/RETFound_MAE.git
%cd RETFound_MAE
```

## 3. Install Dependencies

Run the following in a Colab code cell. This will install PyTorch with CUDA 12.1 (as per recent README updates) and other requirements.

```bash
!pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
!pip install -r requirements.txt
```

## 4. Hugging Face Login

To download pre-trained models, you need to log in to Hugging Face.
*   First, ensure you have a Hugging Face account and a User Access Token (read role is sufficient). You can create a token [here](https://huggingface.co/settings/tokens).
*   Run the following cell. It will prompt you to enter your token.

```bash
!pip install huggingface_hub
!huggingface-cli login
```
When prompted, paste your Hugging Face token.

**Optional for some environments (if direct HF access is blocked):**
If you encounter issues downloading from Hugging Face, you might need to set an environment variable. This is usually not needed in Colab but included for completeness.
```bash
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

## 5. Download and Prepare Demo Dataset (IDRiD Example)

We'll use the IDRiD dataset as an example. The README fine-tuning example uses it.
*   You can find download links in `BENCHMARK.md`. We'll use the Google Drive link.
*   You might need to install `gdown` if it's not already in Colab.

Run the following in a Colab code cell:

```bash
!pip install gdown # Install gdown if not present

# Download IDRiD dataset (this is a placeholder for the actual gdown command from BENCHMARK.md)
# The user needs to replace 'YOUR_GDRIVE_IDRID_FILE_ID' with the actual file ID from the Google Drive link in BENCHMARK.md
# For example, if the link is https://drive.google.com/file/d/1c6zexA705z-ANEBNXJOBsk6uCvRnzmr3/view?usp=sharing
# The file ID is 1c6zexA705z-ANEBNXJOBsk6uCvRnzmr3
!gdown 1c6zexA705z-ANEBNXJOBsk6uCvRnzmr3 -O IDRiD.zip

# Unzip the dataset
!unzip -q IDRiD.zip -d IDRiD_dataset
```

**Important: Data Organization**
The fine-tuning script expects data in a specific structure:
```
IDRiD_data/ # This will be your --data_path argument
├──train
    ├──class_a
    ├──class_b
    └──...
├──val
    ├──class_a
    ├──class_b
    └──...
└──test
    ├──class_a
    ├──class_b
    └──...
```
The downloaded IDRiD dataset might have a different structure (e.g., `IDRiD_dataset/A. Segmentation/1. Original Images/a. Training Set` and `IDRiD_dataset/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv`).

**You will need to manually inspect the contents of `IDRiD_dataset` after unzipping and reorganize the images into the `train/val/test` and class subdirectories as required by the script.** This is a dataset-specific step and often requires custom scripting based on how the dataset is packaged. The example script assumes images are directly in class folders. For disease grading, you might have image files and a CSV for labels. The `util/datasets.py` script likely handles this, but the data needs to be accessible.

For a quick test, you might need to create a simplified structure or use a dataset that's already in the correct format. The `BENCHMARK.md` links point to archives that might contain pre-split data.

**Let's assume you have organized your data into `./IDRiD_organized` in the correct structure for the `main_finetune.py` script (e.g., for a classification task).** The `nb_classes` and task details will depend on this. The IDRiD dataset has multiple tasks (segmentation, grading). The example command uses `nb_classes 5`.

## 6. Run a Demo Fine-tuning Task

This example adapts the fine-tuning command from the README for the IDRiD dataset.
*   `--data_path` should point to your organized IDRiD data directory.
*   `--nb_classes` should match the number of classes for your specific IDRiD task (e.g., 5 for DR grading).
*   `--finetune` specifies the pre-trained model. `RETFound_mae_meh` is one option.
*   We'll reduce `--epochs` for a quick demo.
*   `--batch_size` might need adjustment based on Colab GPU memory (e.g., 8 or 16).

Run the following in a Colab code cell (adjust paths and parameters as needed):

```bash
!torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py \
    --model RETFound_mae \
    --savemodel \
    --global_pool \
    --batch_size 8 \
    --epochs 5 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --nb_classes 5 \
    --data_path ./IDRiD_organized \
    --input_size 224 \
    --task RETFound_mae_meh-IDRiD-demo \
    --finetune RETFound_mae_meh
```

This command will start the fine-tuning process. You should see output indicating training progress. A checkpoint will be saved in `output_dir/RETFound_mae_meh-IDRiD-demo/`.

This provides a basic framework. You'll likely need to adapt the data preparation steps based on the exact structure of the downloaded dataset and the specific task you're targeting.
