import torch
from torch.utils.data import DataLoader
from dataset import ColteaPairedDataset, get_transforms
from model import get_model

# --- CONFIG ---
CSV_PATH = "Coltea-Lung-CT-100W/Coltea-Lung-CT-100W/train_data.csv"
COL_NAME = "train"
DATA_ROOT = "Coltea_Processed_NIfTI"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100

def train():
    # 1. Setup
    ds = ColteaPairedDataset(CSV_PATH, COL_NAME, DATA_ROOT, transform=get_transforms("train"))
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=2)
    
    model = get_model(DEVICE)
    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(f"Starting training on {len(ds)} samples...")
    best_loss = float('inf')

    # 2. Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in loader:
            inputs = batch_data["image"].to(DEVICE)
            targets = batch_data["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1
            
        epoch_loss /= step
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  -> Saved best model")

if __name__ == "__main__":
    train()