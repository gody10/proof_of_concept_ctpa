import torch
import logging
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
LOG_FILE = "test_log.txt"

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
    # Wrap loader with tqdm for a progress bar
    loop = tqdm(loader, desc="Testing", leave=True)

    with torch.no_grad():
        for i, batch_data in enumerate(loop):
            inputs = batch_data["image"].to(DEVICE)
            targets = batch_data["label"].to(DEVICE)
            
            outputs = model(inputs)

            # Update metrics
            evaluator.update(outputs, targets)

            # Save visualization for the first patient
            if i == 0:
                save_comparison_plot(inputs, targets, outputs, "test_result.png")
                msg = " -> Saved visualization to test_result.png"
                tqdm.write(msg)
                logging.info(msg)

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