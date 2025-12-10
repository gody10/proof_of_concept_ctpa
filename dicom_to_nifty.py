import os
import dicom2nifti
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
# Point this to the outer 'Coltea-Lung-CT-100W' folder
RAW_DATA_ROOT = Path("Coltea-Lung-CT-100W/Coltea-Lung-CT-100W/Coltea-Lung-CT-100W") 
# Where we will save the clean NIfTIs
OUTPUT_ROOT = Path("Coltea_Processed_NIfTI") 

# Ensure output directory exists
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def convert_dataset():
    # We loop through all patient folders in the raw directory
    # Assuming folders like 'Patient_ID' are the direct children
    patient_dirs = [p for p in RAW_DATA_ROOT.iterdir() if p.is_dir()]
    
    print(f"Found {len(patient_dirs)} patient folders. Starting conversion...")

    for patient_dir in tqdm(patient_dirs):
        patient_id = patient_dir.name
        
        # Create a destination folder for this patient
        dest_dir = OUTPUT_ROOT / patient_id
        dest_dir.mkdir(exist_ok=True)
        
        # Define the subfolders we care about
        # We want to convert ARTERIAL (Input) and NATIVE (Target)
        phases = {
            "ARTERIAL": "arterial.nii.gz",
            "NATIVE": "native.nii.gz"
            # We can skip VENOUS for this specific specific project to save space/time
        }
        
        for phase_folder, output_filename in phases.items():
            # Construct path: Patient_ID -> ARTERIAL -> DICOM
            dicom_source = patient_dir / phase_folder / "DICOM"
            
            if dicom_source.exists():
                output_path = dest_dir / output_filename
                
                # Skip if already exists (good for resuming interrupted scripts)
                if output_path.exists():
                    continue

                try:
                    # Convert the DICOM folder to a single NIfTI file
                    dicom2nifti.dicom_series_to_nifti(
                        original_dicom_directory=dicom_source,
                        output_file=output_path,
                        reorient_nifti=True 
                    )
                except Exception as e:
                    print(f"Error converting {patient_id} - {phase_folder}: {e}")
            else:
                # print(f"Warning: {phase_folder} not found for {patient_id}")
                pass

if __name__ == "__main__":
    convert_dataset()