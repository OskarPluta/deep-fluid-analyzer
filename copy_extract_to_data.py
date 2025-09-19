import os
import shutil
import argparse
import uuid
import random
import string
from pathlib import Path
from tqdm import tqdm


def generate_random_prefix(length: int = 8) -> str:
    """Generate a random string prefix"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def copy_extracted_to_dataset(
    extracted_dir: str,
    output_dir: str,
    overwrite: bool = False,
    add_random_prefix: bool = True,
    prefix_length: int = 8
):
    """
    Copy extracted data from masks_generator.py to dataset format
    
    Args:
        extracted_dir: Directory containing 'images/' and 'labels_binary/' from masks_generator
        output_dir: Output directory (will create 'images/' and 'labels_binary/' subdirs)
        overwrite: Whether to overwrite existing files
        add_random_prefix: Whether to add random prefix to filenames
        prefix_length: Length of random prefix (default: 8)
    """
    
    extracted_path = Path(extracted_dir)
    output_path = Path(output_dir)
    
    # Check input directories exist
    images_input = extracted_path / 'images'
    labels_input = extracted_path / 'labels_binary'
    
    if not images_input.exists():
        raise ValueError(f"Images directory not found: {images_input}")
    
    if not labels_input.exists():
        raise ValueError(f"Labels directory not found: {labels_input}")
    
    # Create output directories
    images_output = output_path / 'images'
    labels_output = output_path / 'labels_binary'
    
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)
    
    print(f"Copying from {extracted_path} to {output_path}")
    if add_random_prefix:
        print(f"Adding random {prefix_length}-character prefix to filenames")
    
    # Get all image files
    image_files = sorted([
        f for f in os.listdir(images_input) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    print(f"Found {len(image_files)} image files")
    
    # Generate single random prefix for this batch
    if add_random_prefix:
        random_prefix = generate_random_prefix(prefix_length)
        print(f"Using prefix: {random_prefix}_")
    
    copied_images = 0
    copied_masks = 0
    skipped_files = 0
    
    for image_file in tqdm(image_files, desc="Copying files"):
        # Expected mask file name (from masks_generator.py)
        base_name = os.path.splitext(image_file)[0]
        extension = os.path.splitext(image_file)[1]
        mask_file = f"{base_name}_mask.png"
        
        # Source paths
        src_image = images_input / image_file
        src_mask = labels_input / mask_file
        
        # Generate new filenames with optional prefix
        if add_random_prefix:
            new_image_name = f"{random_prefix}_{image_file}"
            new_mask_name = f"{random_prefix}_{mask_file}"
        else:
            new_image_name = image_file
            new_mask_name = mask_file
        
        # Destination paths
        dst_image = images_output / new_image_name
        dst_mask = labels_output / new_mask_name
        
        # Check if mask exists
        if not src_mask.exists():
            print(f"Warning: Mask not found for {image_file}, skipping")
            skipped_files += 1
            continue
        
        # Check if files already exist
        if dst_image.exists() and not overwrite:
            print(f"Skipping {new_image_name} (already exists, use --overwrite to replace)")
            skipped_files += 1
            continue
        
        if dst_mask.exists() and not overwrite:
            print(f"Skipping {new_mask_name} (already exists, use --overwrite to replace)")
            skipped_files += 1
            continue
        
        # Copy files
        try:
            shutil.copy2(src_image, dst_image)
            shutil.copy2(src_mask, dst_mask)
            copied_images += 1
            copied_masks += 1
        except Exception as e:
            print(f"Error copying {image_file}: {e}")
            skipped_files += 1
    
    print(f"\nCopy complete!")
    print(f"Images copied: {copied_images}")
    print(f"Masks copied: {copied_masks}")
    print(f"Files skipped: {skipped_files}")
    print(f"\nOutput structure:")
    print(f"  {images_output} - {len(list(images_output.glob('*')))} files")
    print(f"  {labels_output} - {len(list(labels_output.glob('*')))} files")
    
    # Verify naming convention
    verify_naming_convention(output_path)


def verify_naming_convention(data_dir: str):
    """Verify that the naming convention matches what dataset expects"""
    data_path = Path(data_dir)
    images_dir = data_path / 'images'
    labels_dir = data_path / 'labels_binary'
    
    print(f"\nVerifying naming convention...")
    
    if not images_dir.exists() or not labels_dir.exists():
        print("Warning: Output directories don't exist yet")
        return
    
    image_files = sorted([
        f for f in os.listdir(images_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    valid_pairs = 0
    invalid_pairs = 0
    
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        expected_mask = labels_dir / f"{base_name}_mask.png"
        
        if expected_mask.exists():
            valid_pairs += 1
        else:
            print(f"Warning: No mask found for {img_file}")
            invalid_pairs += 1
    
    print(f"Valid image-mask pairs: {valid_pairs}")
    print(f"Invalid pairs: {invalid_pairs}")
    
    if invalid_pairs == 0:
        print("✓ All files follow the correct naming convention!")
        print("✓ Ready for use with dataset_split.py and dataset.py")
    else:
        print("⚠ Some files don't follow the expected naming convention")


def main():
    parser = argparse.ArgumentParser(description='Copy extracted video data to dataset format')
    parser.add_argument('--extracted-dir', type=str, required=True,
                       help='Directory containing extracted data (with images/ and labels_binary/ subdirs)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory (default: data)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing files')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify naming convention, don\'t copy files')
    parser.add_argument('--no-prefix', action='store_true',
                       help='Don\'t add random prefix to filenames')
    parser.add_argument('--prefix-length', type=int, default=8,
                       help='Length of random prefix (default: 8)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.extracted_dir):
        print(f"Error: Extracted directory not found: {args.extracted_dir}")
        return 1
    
    if args.verify_only:
        verify_naming_convention(args.output_dir)
        return 0
    
    try:
        copy_extracted_to_dataset(
            extracted_dir=args.extracted_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            add_random_prefix=not args.no_prefix,
            prefix_length=args.prefix_length
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())