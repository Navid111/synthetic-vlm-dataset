import json
import argparse
from PIL import Image
import random
import os


def visualize_samples(dataset_dir, num_samples=5):
    """Display sample images with their captions"""
    
    # Load dataset metadata
    with open(f"{dataset_dir}/dataset.json", 'r') as f:
        dataset = json.load(f)
    
    # Randomly sample images
    samples = random.sample(dataset, min(num_samples, len(dataset)))
    
    print(f"\n{'='*80}")
    print(f"DATASET VISUALIZATION - Showing {len(samples)} random samples")
    print(f"{'='*80}\n")
    
    for idx, sample in enumerate(samples, 1):
        print(f"Sample {idx}:")
        print(f"  Image: {sample['image_filename']}")
        print(f"  Annotations:")
        for ann in sample['annotations']:
            print(f"    - {ann['color']} {ann['shape']} (size: {ann['size']}) at {ann['position']}")
        print(f"  Captions:")
        for caption in sample['captions']:
            print(f"    - {caption}")
        print()
    
    # Statistics
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}\n")
    
    total_images = len(dataset)
    print(f"Total images: {total_images}")
    
    # Count shapes
    shape_counts = {}
    color_counts = {}
    position_counts = {}
    
    for item in dataset:
        for ann in item['annotations']:
            shape_counts[ann['shape']] = shape_counts.get(ann['shape'], 0) + 1
            color_counts[ann['color']] = color_counts.get(ann['color'], 0) + 1
            position_counts[ann['position']] = position_counts.get(ann['position'], 0) + 1
    
    print("\nShape distribution:")
    for shape, count in sorted(shape_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {shape}: {count}")
    
    print("\nColor distribution:")
    for color, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {color}: {count}")
    
    print("\nPosition distribution:")
    for position, count in sorted(position_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {position}: {count}")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize synthetic VLM dataset')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to display')
    args = parser.parse_args()
    
    if not os.path.exists(f"{args.dataset_dir}/dataset.json"):
        print(f"Error: dataset.json not found in {args.dataset_dir}")
        return
    
    visualize_samples(args.dataset_dir, args.num_samples)


if __name__ == "__main__":
    main()