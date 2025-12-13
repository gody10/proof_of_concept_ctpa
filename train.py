import torch
import logging
from tqdm import tqdm, trange  # trange is a shortcut for tqdm(range(...))
from torch.utils.data import DataLoader
from dataset import ColteaPairedDataset, get_transforms
from model import get_model

# --- CONFIG ---
CSV_PATH = "data/Coltea-Lung-CT-100W/train_data.csv"
COL_NAME = "train"
DATA_ROOT = "data/Coltea_Processed_Nifti"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
LOG_FILE = "training_log.txt"

# --- LOGGER SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def train():
    # 1. Setup
    ds = ColteaPairedDataset(CSV_PATH, COL_NAME, DATA_ROOT, transform=get_transforms("train"))
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=8)
    
    model = get_model(DEVICE)
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    start_msg = f"Starting training on {len(ds)} samples using {DEVICE}..."
    print(start_msg)
    logging.info(start_msg)

    best_loss = float('inf')

    # 2. Outer Loop: Wraps the Epochs
    # position=0 ensures this bar stays at the top
    epoch_iterator = tqdm(range(EPOCHS), desc="Total Progress", position=0)

    for epoch in epoch_iterator:
        model.train()
        epoch_loss = 0
        step = 0
        
        # 3. Inner Loop: Wraps the Batches
        # position=1 puts this bar below the total progress
        # leave=False clears this bar after the epoch ends, keeping the terminal clean for logs
        batch_iterator = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", position=1, leave=False)

        for batch_data in batch_iterator:
            inputs = batch_data["image"].to(DEVICE)
            targets = batch_data["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss
            step += 1
            
            # Update the inner bar with the instantaneous loss
            batch_iterator.set_postfix(loss=f"{current_loss:.4f}")

        # End of Epoch Stats
        epoch_loss /= step
        log_msg = f"Epoch {epoch + 1}/{EPOCHS} - Avg Loss: {epoch_loss:.4f}"
        
        # Use tqdm.write so it prints ABOVE the progress bars without breaking layout
        tqdm.write(log_msg) 
        logging.info(log_msg)

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_model.pth")
            
            save_msg = f" -> Saved best model (Loss: {best_loss:.4f})"
            tqdm.write(save_msg)
            logging.info(save_msg)

if __name__ == "__main__":
    train()