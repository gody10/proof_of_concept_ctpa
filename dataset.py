import os
import pandas as pd
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    ScaleIntensityRangePercentilesd, Resized, ToTensord
)

class ColteaPairedDataset(Dataset):
    def __init__(self, csv_file, col_name, root_dir, transform=None):
        """
        Args:
            csv_file: Path to train.csv, eval.csv, or test.csv
            col_name: The column name containing IDs ('train', 'eval', or 'test')
            root_dir: Path to the processed NIfTI folder
        """
        self.df = pd.read_csv(csv_file)
        self.col_name = col_name
        self.root_dir = root_dir
        self.transform = transform
        self.valid_samples = self._filter_valid_files()

    def _filter_valid_files(self):
        valid_data = []
        # Convert IDs to string to avoid type mismatches
        patient_ids = self.df[self.col_name].astype(str).tolist()
        
        for pid in patient_ids:
            art_path = os.path.join(self.root_dir, pid, "arterial.nii.gz")
            nat_path = os.path.join(self.root_dir, pid, "native.nii.gz")
            
            if os.path.exists(art_path) and os.path.exists(nat_path):
                valid_data.append({"image": art_path, "label": nat_path})
        return valid_data

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        data = self.valid_samples[idx]
        if self.transform:
            data = self.transform(data)
        return data

def get_transforms(mode="train"):
    """
    Returns transforms. 
    mode="train" can include augmentations (rotations, flips) in the future.
    mode="test" should be deterministic.
    """
    base_transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Clip [-1000, 1000] and normalize to [0, 1]
        ScaleIntensityRangePercentilesd(
            keys=["image", "label"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        ),
        # Resize to manageable VRAM size for PoC
        Resized(
            keys=["image", "label"], spatial_size=(128, 128, 64), mode=("trilinear", "trilinear")
        ),
        ToTensord(keys=["image", "label"]),
    ]
    
    return Compose(base_transforms)