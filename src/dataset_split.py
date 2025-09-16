import os
import random
import shutil
from pathlib import Path

def split_dataset(image_dir: str, label_dir: str, output_dir: str,
                  train_ratio: float=0.7, val_ratio: float=0.15, test_ratio: float=0.15):
    """Copies images and their corresponding labels into train, validation and 
    test directories, each containing 'images' and 'labels' subdirectories."""
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    image_dir, label_dir, output_dir = Path(image_dir), Path(label_dir), Path(output_dir)

    for split in ['train', 'val', 'test']:
        os.makedirs(output_dir / split / 'images', exist_ok=True)
        os.makedirs(output_dir / split / 'labels', exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    print(f"Found {len(image_files)} images.")

    random.shuffle(image_files)

    num_images = len(image_files)
    last_train_index = int(num_images * train_ratio)
    last_val_index = last_train_index + int(num_images * val_ratio)

    file_splits = {
        'train': image_files[:last_train_index],
        'val': image_files[last_train_index:last_val_index],
        'test': image_files[last_val_index:]
    }

    for split, files in file_splits.items():
        for file in files:
            copy_image = output_dir / split / 'images' / file
            shutil.copy(image_dir / file, copy_image)

            label_file = find_label(file, label_dir)
            print(label_file)
            if label_file:
                shutil.copy(image_dir / file, copy_image)
                copy_label = output_dir / split / 'labels' / os.path.basename(label_file)
                shutil.copy(label_file, copy_label)
            else:
                print(f"Warning: No label found for image {file}")

    print("Dataset saved to ", output_dir)


def find_label(image_file:str, label_dir:Path) -> str:
    base_name = os.path.splitext(image_file)[0]
    for ext in ['.png', '.jpg', '.jpeg']:
        label_path = label_dir / f"{base_name}_mask{ext}"
        if label_path.exists():
            return str(label_path)
    return None


def main():
    IMAGE_DIR = "data/images"
    LABEL_DIR = "data/labels_binary"
    OUTPUT_DIR = "datasets"

    split_dataset(IMAGE_DIR, LABEL_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()