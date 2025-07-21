from google.colab import drive
import os
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from tqdm.notebook import tqdm
import sys

# For metric calculation
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, cohen_kappa_score
import numpy as np

# -----------------------------------------------------------------------------
# GLOBAL CONFIGURATION PARAMETER
# Choose how to get your input images:
# 'COLAB_UPLOAD': You'll manually upload images to a temporary Colab folder.
# 'GDRIVE': Images are already in a specified folder in your Google Drive.
# 'DOWNLOAD_URL': Images will be downloaded from a provided URL (e.g., GitHub, Hugging Face).
# -----------------------------------------------------------------------------
INPUT_IMAGE_SOURCE = 'DOWNLOAD_URL' # <--- CHANGE THIS TO YOUR DESIRED OPTION
# If using 'DOWNLOAD_URL', you MUST configure DOWNLOAD_URL_OR_ID and DOWNLOAD_FILENAME below.

# --- Parameters for DOWNLOAD_URL option ---
# IMPORTANT: You will need to find/prepare a publicly accessible, labeled test set.
# Recommended: Download APTOS 2019 training data from Kaggle, create a small Labeled test subset,
# zip it, and upload that zip to your Google Drive. Then use its Shareable Link ID here.

# HYPOTHETICAL EXAMPLE (REPLACE THESE WITH YOUR ACTUAL LINK/ID)
# Example if you put a pre-classified subset zip into your Google Drive and got its shareable ID:
DOWNLOAD_URL_OR_ID = 'YOUR_GOOGLE_DRIVE_FILE_ID_FOR_APTOS_TEST_SUBSET' # e.g., '1aBcDeFgHiJkLmNoPqRsTuvWxYzA1B2C'
DOWNLOAD_FILENAME = 'aptos_2019_test_subset_classified.zip'

# After unzipping, this will be the exact directory containing your '0', '1', '2', '3', '4' class folders.
# Common structure: YourZip.zip extracts to YourZipFolder/0/, YourZipFolder/1/, etc.
# Adjust 'aptos_2019_test_subset_classified' to match the actual folder name created by unzipping your file.
DOWNLOADED_DATA_SUBDIR = 'aptos_2019_test_subset_classified' # Adjust if the zip unzips to a different folder name

# -----------------------------------------------------------------------------

# 1. Mount Google Drive
drive.mount('/content/drive')
print("Google Drive mounted successfully.")

# 2. Define Paths and Handle Input Image Source
# Path to the directory where your model checkpoint is saved in Google Drive
GDRIVE_CHECKPOINT_PATH = '/content/drive/MyDrive/RETFound_Checkpoints/RETFound_mae_IDRiD_best_epoch36.pth'

# Path to the output folder in Google Drive where results will be saved
GDRIVE_OUTPUT_RESULTS_DIR = '/content/drive/MyDrive/RETFound_Predictions_KaggleTest' # New folder for this evaluation
os.makedirs(GDRIVE_OUTPUT_RESULTS_DIR, exist_ok=True)
print(f"Output results will be saved to: {GDRIVE_OUTPUT_RESULTS_DIR}")

# Determine the actual input directory based on INPUT_IMAGE_SOURCE
INPUT_IMAGES_DIR = '' # Initialize

if INPUT_IMAGE_SOURCE == 'COLAB_UPLOAD':
    INPUT_IMAGES_DIR = '/content/images_to_predict'
    os.makedirs(INPUT_IMAGES_DIR, exist_ok=True)
    print(f"Please manually upload your images to: {INPUT_IMAGES_DIR}")
    print("These files are temporary and will be lost when the Colab session ends.")

elif INPUT_IMAGE_SOURCE == 'GDRIVE':
    # You might want to upload your new test set (e.g., APTOS subset) to a different Drive folder
    # For instance: '/content/drive/MyDrive/APTOS_Test_Subset_Organized'
    INPUT_IMAGES_DIR = '/content/drive/MyDrive/Fundus_Images_For_Prediction' # Update if your new test set is here
    if not os.path.exists(INPUT_IMAGES_DIR):
        print(f"Warning: Input image directory '{INPUT_IMAGES_DIR}' not found in Google Drive. Please create it and upload images.")
    else:
        print(f"Images will be loaded from Google Drive: {INPUT_IMAGES_DIR}")

elif INPUT_IMAGE_SOURCE == 'DOWNLOAD_URL':
    # Temporary local folder for downloaded images
    base_download_dir = '/content/downloaded_data'
    os.makedirs(base_download_dir, exist_ok=True)

    print(f"Attempting to download images to: {base_download_dir}")

    # Install gdown if needed for Google Drive downloads
    if 'gdown' not in sys.modules: # Check if gdown is already imported (installed)
        try:
            import gdown # Try importing directly
        except ImportError:
            print("Installing gdown...")
            !pip install gdown
            import gdown # Import after installation

    archive_path = os.path.join(base_download_dir, DOWNLOAD_FILENAME)

    # Perform the download
    # Check if it's a Google Drive ID (usually a 33-char alphanumeric string, not starting with http)
    if len(DOWNLOAD_URL_OR_ID) == 33 and not DOWNLOAD_URL_OR_ID.startswith('http'):
        print(f"Downloading from Google Drive ID: {DOWNLOAD_URL_OR_ID}")
        !gdown --id {DOWNLOAD_URL_OR_ID} -O {archive_path}
    elif DOWNLOAD_URL_OR_ID.startswith('http'): # Assume a direct HTTP/HTTPS link
        print(f"Downloading from URL: {DOWNLOAD_URL_OR_ID}")
        !wget {DOWNLOAD_URL_OR_ID} -O {archive_path}
    else:
        raise ValueError(f"Error: Unrecognized download URL/ID format: {DOWNLOAD_URL_OR_ID}. Please provide a valid Google Drive ID or direct URL.")

    # Unzip if it's a zip file
    if DOWNLOAD_FILENAME.endswith('.zip') and os.path.exists(archive_path):
        print(f"Unzipping {DOWNLOAD_FILENAME}...")
        !unzip -q {archive_path} -d {base_download_dir}
        print("Unzipping complete.")

        # This is where you adjust the final INPUT_IMAGES_DIR based on the unzipped structure.
        # It's assumed your zip extracts to a folder named DOWNLOADED_DATA_SUBDIR
        INPUT_IMAGES_DIR = os.path.join(base_download_dir, DOWNLOADED_DATA_SUBDIR)

        # Verify the structure after unzipping - uncomment to see
        print(f"Contents of {INPUT_IMAGES_DIR} after unzipping (verify this path!):")
        !ls -R {INPUT_IMAGES_DIR}
    else:
        print(f"Warning: Downloaded file '{DOWNLOAD_FILENAME}' is not a zip or does not exist. Assuming it's a direct image folder or single file.")
        # If it's a folder, INPUT_IMAGES_DIR might still be base_download_dir
        # If it's a single image, you'll need to adjust the inference logic for single files.
        INPUT_IMAGES_DIR = base_download_dir # Default to base_download_dir if not a zip

    print(f"Images ready in: {INPUT_IMAGES_DIR}")

else:
    raise ValueError("Invalid INPUT_IMAGE_SOURCE. Choose 'COLAB_UPLOAD', 'GDRIVE', or 'DOWNLOAD_URL'.")


# Add the project root to sys.path to import model definition
project_root = '/content/RETFound_MAE'
if not os.path.exists(project_root):
    print(f"Warning: RETFound_MAE project not found at {project_root}. Cloning it now...")
    !git clone https://github.com/rmaphoh/RETFound_MAE.git {project_root}
    %cd {project_root}
    # You might also need to reinstall dependencies if this is a fresh Colab session
    # !pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    # !pip install -r requirements.txt
else:
    %cd {project_root}

if project_root not in sys.path:
    sys.path.append(project_root)

# Import the model definition
try:
    from models_mae import RETFound_mae
    print("RETFound_mae model definition imported.")
except ImportError as e:
    print(f"Error importing RETFound_mae model: {e}")
    print("Please ensure 'models_mae.py' is in the RETFound_MAE directory or adjust the import path.")


# --- Configuration (MUST MATCH TRAINING CONFIG) ---
INPUT_SIZE = 224
NB_CLASSES = 5
GLOBAL_POOL = True # Based on your training args

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Load the Model Architecture ---
# Ensure these match your training architecture exactly
model = RETFound_mae(
    img_size=INPUT_SIZE,
    patch_size=16,
    in_chans=3,
    num_classes=NB_CLASSES,
    embed_dim=768,
    depth=12,
    num_heads=12,
    decoder_embed_dim=512,
    decoder_depth=8,
    decoder_num_heads=16,
    mlp_ratio=4.,
    norm_layer=torch.nn.LayerNorm,
    norm_pix_loss=False,
    global_pool=GLOBAL_POOL
)

# --- 2. Load the Checkpoint Weights ---
print(f"Loading checkpoint from: {GDRIVE_CHECKPOINT_PATH}")
if os.path.exists(GDRIVE_CHECKPOINT_PATH):
    checkpoint = torch.load(GDRIVE_CHECKPOINT_PATH, map_location='cpu')
    model_state_dict = checkpoint['model']
    model.load_state_dict(model_state_dict, strict=False)
    model.to(device)
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully!")
else:
    print(f"Error: Checkpoint file not found at '{GDRIVE_CHECKPOINT_PATH}'. Please ensure it's copied to your Google Drive.")
    model = None # Set model to None to prevent further execution if load failed

# --- 3. Define Image Preprocessing ---
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
inference_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Define your class labels (MUST match the order used during training and in the new dataset)
# For IDRiD, it was: ['anoDR', 'bmildDR', 'cmoderateDR', 'dsevereDR', 'eproDR']
# For standard Kaggle/EyePACS DR, it's typically numeric for the 5 grades (0-4):
class_labels = ['0', '1', '2', '3', '4']
# IMPORTANT: Double-check the exact folder names in your new test dataset!
# If your folders are 'No DR', 'Mild DR', etc., you must update this list.


if model is None:
    print("Model was not loaded. Skipping inference and metric calculation.")
else:
    results = [] # To store results for all images
    all_true_labels = [] # To store true labels for metric calculation
    all_predicted_labels = [] # To store predicted labels for metric calculation
    all_probabilities = [] # To store probabilities for ROC AUC

    # Create a torchvision ImageFolder dataset for easy loading and access to true labels
    try:
        # ImageFolder will automatically create dataset and labels from the folder structure
        from torchvision.datasets import ImageFolder
        test_dataset = ImageFolder(root=INPUT_IMAGES_DIR, transform=inference_transform)
        # Use a smaller batch size for DataLoader if you encounter OOM errors
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
        print(f"Loaded new test dataset with {len(test_dataset)} images.")

        # Get the mapping of class names to indices that ImageFolder uses
        # This is CRUCIAL to match the predicted indices to the correct class names.
        # It's usually alphabetical (e.g., '0' will be index 0, '1' index 1, etc. if folder names are numbers)
        true_class_labels_mapping = {v: k for k, v in test_dataset.class_to_idx.items()}
        print(f"True class mapping from dataset: {true_class_labels_mapping}")

        # Ensure our defined class_labels match the order ImageFolder detected for metrics.
        # We'll use the ImageFolder's `classes` list (which is sorted alphabetically) for consistent indexing.
        consistent_class_labels = test_dataset.classes
        print(f"Using consistent class labels for metrics: {consistent_class_labels}")


    except Exception as e:
        print(f"Error loading ImageFolder dataset from {INPUT_IMAGES_DIR}: {e}")
        print("Please ensure the input directory is structured as 'root/class_label/image.jpg' and contains images.")
        test_loader = None # Set to None to prevent further execution

    if test_loader and len(test_dataset) > 0:
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Predicting on test images"):
                inputs = inputs.to(device)
                outputs = model(inputs)

                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)

                all_true_labels.extend(labels.cpu().numpy())
                all_predicted_labels.extend(predicted_classes.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # --- Calculate Metrics ---
        print("\n--- Evaluation Metrics on New Dataset ---")

        # Accuracy
        accuracy = accuracy_score(all_true_labels, all_predicted_labels)
        print(f"Accuracy: {accuracy:.4f}")

        # F1 Score (weighted for imbalance)
        # Handle cases where some labels might not be present in predictions or true labels,
        # which can happen with small test sets or imbalanced data.
        try:
            f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
            print(f"F1 Score (weighted): {f1:.4f}")
        except ValueError as e:
            print(f"Could not calculate F1 Score (weighted): {e}")

        # ROC AUC (one-vs-rest, weighted for imbalance)
        # This requires probabilities, not just predicted classes.
        try:
            # Need to ensure all classes are represented in true_labels for 'ovr' multi_class to work
            # Or handle cases where some classes might be missing
            roc_auc = roc_auc_score(all_true_labels, np.array(all_probabilities), multi_class='ovr', average='weighted')
            print(f"ROC AUC (weighted, One-vs-Rest): {roc_auc:.4f}")
        except ValueError as e:
            print(f"Could not calculate ROC AUC: {e}. This might happen if a class has only one sample or no samples in true labels in the batch/dataset, or if there's only one class present.")
            if len(np.unique(all_true_labels)) < 2:
                print("ROC AUC requires at least two classes present in true labels.")

        # Kappa Score
        try:
            kappa = cohen_kappa_score(all_true_labels, all_predicted_labels)
            print(f"Cohen's Kappa: {kappa:.4f}")
        except ValueError as e:
            print(f"Could not calculate Cohen's Kappa: {e}")

        # --- Save detailed results to CSV ---
        # Remap predicted indices back to string labels for clarity in CSV
        for i in range(len(all_predicted_labels)):
            img_path, _ = test_dataset.samples[i] # Get original image path from ImageFolder
            image_filename = os.path.basename(img_path)

            true_idx = all_true_labels[i]
            predicted_idx = all_predicted_labels[i]

            # Map indices back to consistent_class_labels (which come from ImageFolder's detection)
            true_label = consistent_class_labels[true_idx]
            predicted_label = consistent_class_labels[predicted_idx]

            probs = all_probabilities[i]
            probs_dict = {f'prob_{consistent_class_labels[j]}': p for j, p in enumerate(probs)}

            results.append({
                'image_filename': image_filename,
                'true_class_index': true_idx,
                'true_class_label': true_label,
                'predicted_class_index': predicted_idx,
                'predicted_class_label': predicted_label,
                'confidence_of_prediction': probs[predicted_idx],
                **probs_dict
            })

        results_df = pd.DataFrame(results)
        output_csv_path = os.path.join(GDRIVE_OUTPUT_RESULTS_DIR, 'inference_results_new_dataset.csv')
        results_df.to_csv(output_csv_path, index=False)
        print(f"\nInference complete! Detailed results saved to: {output_csv_path}")

        # Display first few results
        print("\nFirst 5 results:")
        print(results_df.head())

    else:
        print("Skipping inference due to no test loader or empty dataset.")
