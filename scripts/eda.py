import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataloader import OffroadDataset, CLASS_NAMES

def calculate_class_distribution(dataset_path):
    print(f"Analyzing dataset at: {dataset_path}")
    dataset = OffroadDataset(dataset_path)
    
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
        
    class_pixel_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    total_pixels = 0
    
    # Calculate for all images. We can sample if it's too large, but 
    # for a thorough baseline EDA we will count everything.
    for i in tqdm(range(len(dataset)), desc="Calculating distributions"):
        _, mask = dataset[i]
        mask_arr = np.array(mask)
        
        # Count pixels for each class
        for class_id in range(len(CLASS_NAMES)):
            class_pixel_counts[class_id] += np.sum(mask_arr == class_id)
            
        total_pixels += mask_arr.size
        
    print("\n--- EDA RESULTS: Class Distribution ---")
    results = []
    
    # Calculate frequencies and suggested weights (Inverse Frequency)
    for class_id, count in enumerate(class_pixel_counts):
        freq = count / total_pixels if total_pixels > 0 else 0
        
        # Avoid zero division for missing classes
        if count > 0:
            weight = total_pixels / (len(CLASS_NAMES) * count)
        else:
            weight = 0.0
            
        results.append({
            "Class ID": class_id,
            "Class Name": CLASS_NAMES[class_id],
            "Pixel Count": count,
            "Frequency (%)": freq * 100,
            "Suggested Weight": weight
        })
        
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Save to CSV
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data'), exist_ok=True)
    df.to_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'class_distribution.csv'), index=False)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df['Class Name'], df['Frequency (%)'], color='skyblue')
    plt.title('Pixel Class Frequency (%) in Training Set', fontsize=16)
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Frequency (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'data', 'class_distribution.png'))
    print("\nSaved distribution plot to data/class_distribution.png and CSV to data/class_distribution.csv")

if __name__ == "__main__":
    train_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'Offroad_Segmentation_Training_Dataset', 'Offroad_Segmentation_Training_Dataset', 'train'))
    calculate_class_distribution(train_dir)
