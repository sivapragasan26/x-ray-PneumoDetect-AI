"""
Dataset Balancing Script for Chest X-Ray Pneumonia Detection
Creates a balanced dataset with equal samples per class to eliminate bias
"""

import os
import shutil
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

def create_balanced_dataset():
    """Create balanced dataset with equal samples per class"""
    print("Creating balanced dataset...")
    print("=" * 40)
    
    # Define paths
    source_path = Path("../data/raw/chest_xray")
    processed_path = Path("../data/processed")
    
    # Create processed folder structure
    for split in ['train', 'val', 'test']:
        for cls in ['normal', 'pneumonia']:
            (processed_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    print("Collecting training images...")
    
    # Collect all training images
    train_normal_path = source_path / "train" / "normal"
    train_pneumonia_path = source_path / "train" / "pneumonia"
    
    # Try both lowercase and uppercase folder names
    if not train_normal_path.exists():
        train_normal_path = source_path / "train" / "NORMAL"
    if not train_pneumonia_path.exists():
        train_pneumonia_path = source_path / "train" / "PNEUMONIA"
    
    # Collect image files
    normal_images = []
    if train_normal_path.exists():
        normal_images = (list(train_normal_path.glob("*.jpeg")) + 
                        list(train_normal_path.glob("*.jpg")))
        print(f"  Found {len(normal_images):,} normal images")
    
    pneumonia_images = []
    if train_pneumonia_path.exists():
        pneumonia_images = (list(train_pneumonia_path.glob("*.jpeg")) + 
                           list(train_pneumonia_path.glob("*.jpg")))
        print(f"  Found {len(pneumonia_images):,} pneumonia images")
    
    if len(normal_images) == 0 or len(pneumonia_images) == 0:
        print("Could not find training images!")
        return False
    
    # Balance the dataset using minority class count
    min_count = min(len(normal_images), len(pneumonia_images))
    print(f"\nBalancing to {min_count:,} images per class...")
    
    # Randomly sample equal numbers
    np.random.seed(42)  # For reproducible results
    selected_normal = np.random.choice(normal_images, min_count, replace=False)
    selected_pneumonia = np.random.choice(pneumonia_images, min_count, replace=False)
    
    # Combine all selected images
    all_images = list(selected_normal) + list(selected_pneumonia)
    all_labels = ['normal'] * min_count + ['pneumonia'] * min_count
    
    print(f"Selected {len(all_images):,} balanced images total")
    
    # Create train/val/test splits: 70%/20%/10%
    print("\nCreating train/val/test splits...")
    
    # First split: separate test set (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_images, all_labels, test_size=0.1, stratify=all_labels, random_state=42
    )
    
    # Second split: separate train/val from remaining 90%
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.222, stratify=y_temp, random_state=42  # 0.222 * 0.9 ≈ 0.2
    )
    
    print(f"  Training: {len(X_train):,} images ({len(X_train)//2} per class)")
    print(f"  Validation: {len(X_val):,} images ({len(X_val)//2} per class)")
    print(f"  Testing: {len(X_test):,} images ({len(X_test)//2} per class)")
    
    # Copy files to processed structure
    def copy_files(file_list, label_list, split_name):
        print(f"  Copying {split_name} files...")
        for file_path, label in zip(file_list, label_list):
            filename = file_path.name
            dest_dir = processed_path / split_name / label
            dest_path = dest_dir / filename
            shutil.copy2(file_path, dest_path)
    
    copy_files(X_train, y_train, 'train')
    copy_files(X_val, y_val, 'val')
    copy_files(X_test, y_test, 'test')
    
    # Verify the balanced distribution
    print(f"\nBALANCED DATASET CREATED:")
    print("=" * 35)
    total_balanced = 0
    
    for split in ['train', 'val', 'test']:
        normal_count = len(list((processed_path / split / 'normal').glob('*')))
        pneumonia_count = len(list((processed_path / split / 'pneumonia').glob('*')))
        total = normal_count + pneumonia_count
        total_balanced += total
        
        print(f"{split.upper()}:")
        print(f"  Normal: {normal_count:,} images")
        print(f"  Pneumonia: {pneumonia_count:,} images")
        print(f"  Total: {total:,} images")
        print(f"  Balance: {normal_count}/{pneumonia_count} = {normal_count/pneumonia_count:.2f}")
        print()
    
    print(f"TOTAL BALANCED DATASET: {total_balanced:,} images")
    print(f"New Balance Ratio: 1.00:1 (Perfect Balance)")
    
    return True

if __name__ == "__main__":
    print("Starting dataset balancing...")
    
    success = create_balanced_dataset()
    
    if success:
        print("\nSUCCESS! Balanced dataset ready!")
        print("\nACCOMPLISHMENTS:")
        print("- Fixed class imbalance (1.00:1 ratio)")
        print("- Created proper train/val/test splits")
        print("- Equal samples per class (eliminates bias)")
        print("- Preserved image quality and diversity")
        print(f"\nNEXT STEP: Model Training")
        print("Bias-free dataset is ready in: data/processed/")
    else:
        print("Failed to create balanced dataset")
