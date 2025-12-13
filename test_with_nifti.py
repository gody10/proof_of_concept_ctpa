import torch
import logging
import nibabel as nib  # pip install nibabel
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import ColteaPairedDataset, get_transforms
from model import get_model
from evaluation import Evaluator, save_comparison_plot

# --- CONFIG ---
CSV_PATH = "data/Coltea-Lung-CT-100W/test_data.csv"
COL_NAME = "test"
DATA_ROOT = "data/Coltea_Processed_Nifti"
MODEL_PATH = "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_FILE = "test2_log.txt"

# --- LOGGER SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def test():
    # 1. Setup
    ds = ColteaPairedDataset(CSV_PATH, COL_NAME, DATA_ROOT, transform=get_transforms("test"))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    
    model = get_model(DEVICE)
    # Load weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        logging.info(f"Loaded weights from {MODEL_PATH}")
    except FileNotFoundError:
        logging.error(f"Model file {MODEL_PATH} not found.")
        return

    model.eval()

    evaluator = Evaluator(DEVICE)
    
    start_msg = f"Starting evaluation on {len(ds)} samples using {DEVICE}..."
    print(start_msg)
    logging.info(start_msg)

    # 2. Inference Loop
    loop = tqdm(loader, desc="Testing", leave=True)

    with torch.no_grad():
        for i, batch_data in enumerate(loop):
            inputs = batch_data["image"].to(DEVICE)
            targets = batch_data["label"].to(DEVICE)
            
            # Forward pass (3D output)
            outputs = model(inputs)

            # Update metrics
            evaluator.update(outputs, targets)

            # --- NEW: Save the 3D Volume for the first patient ---
            if i == 0:
                # 1. Save the visual 2D slice (Quick preview)
                save_comparison_plot(inputs, targets, outputs, "test_preview.png")
                
                # 2. Save the FULL 3D NIfTI volumes
                # We need to move tensors to CPU and convert to Numpy
                # Shape is usually (Batch, Channel, Depth, Height, Width) -> (1, 1, D, H, W)
                # We squeeze() to get (D, H, W)
                out_np = outputs.cpu().numpy()[0, 0, :, :, :]
                tgt_np = targets.cpu().numpy()[0, 0, :, :, :]
                inp_np = inputs.cpu().numpy()[0, 0, :, :, :]

                # Create NIfTI objects (using identity affine for simplicity, 
                # or copy from your original dataset if available)
                affine = np.eye(4) 
                
                img_out = nib.Nifti1Image(out_np, affine)
                img_tgt = nib.Nifti1Image(tgt_np, affine)
                img_inp = nib.Nifti1Image(inp_np, affine)

                # Save to disk
                nib.save(img_inp, "test_0_input.nii.gz")
                nib.save(img_tgt, "test_0_target.nii.gz")
                nib.save(img_out, "test_0_prediction.nii.gz")

                msg = " -> Saved 3D NIfTI files (input, target, prediction) for inspection"
                tqdm.write(msg)
                logging.info(msg)

    # ... (Results printing same as before) ...
    # 3. Print and Save Final Metrics
    results = evaluator.get_results()
    
    logging.info("\n--- Final Results ---")
    tqdm.write("\n--- Final Results ---")
    
    for metric, value in results.items():
        res_msg = f"{metric}: {value:.4f}"
        tqdm.write(res_msg)
        logging.info(res_msg)

if __name__ == "__main__":
    test()