import torch
from torch.utils.data import DataLoader
from dataset import ColteaPairedDataset, get_transforms
from model import get_model
from evaluation import Evaluator, save_comparison_plot

# --- CONFIG ---
CSV_PATH = "Coltea-Lung-CT-100W/Coltea-Lung-CT-100W/test_data.csv"
COL_NAME = "test"
DATA_ROOT = "Coltea_Processed_NIfTI"
MODEL_PATH = "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test():
    # 1. Setup
    ds = ColteaPairedDataset(CSV_PATH, COL_NAME, DATA_ROOT, transform=get_transforms("test"))
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    model = get_model(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    evaluator = Evaluator(DEVICE)
    
    print(f"Starting evaluation on {len(ds)} samples...")

    # 2. Inference Loop
    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            inputs = batch_data["image"].to(DEVICE)
            targets = batch_data["label"].to(DEVICE)
            
            outputs = model(inputs)

            # Update metrics
            evaluator.update(outputs, targets)

            # Save visualization for the first patient
            if i == 0:
                save_comparison_plot(inputs, targets, outputs, "test_result.png")
                print("Saved visualization to test_result.png")

    # 3. Print Final Metrics
    results = evaluator.get_results()
    print("\n--- Final Results ---")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    test()