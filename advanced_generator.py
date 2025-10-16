import numpy as np
from PIL import Image, ImageDraw
import random
import json
import os
import argparse
from tqdm import tqdm
import yaml
from generate_dataset import (
    SHAPES, COLORS, POSITIONS, 
    get_position_coords, draw_shape, load_config
)


def get_spatial_relationship(ann1, ann2):
    """Determine spatial relationship between two shapes"""
    x1, y1 = ann1['exact_coords']
    x2, y2 = ann2['exact_coords']
    
    threshold = 50  # pixels threshold for "near"
    
    if abs(y1 - y2) < threshold:
        if x1 < x2 - threshold:
            return f"The {ann1['color']} {ann1['shape']} is to the left of the {ann2['color']} {ann2['shape']}."
        elif x1 > x2 + threshold:
            return f"The {ann1['color']} {ann1['shape']} is to the right of the {ann2['color']} {ann2['shape']}."
    
    if abs(x1 - x2) < threshold:
        if y1 < y2 - threshold:
            return f"The {ann1['color']} {ann1['shape']} is above the {ann2['color']} {ann2['shape']}."
        elif y1 > y2 + threshold:
            return f"The {ann1['color']} {ann1['shape']} is below the {ann2['color']} {ann2['shape']}."
    
    return None


def generate_counting_captions(annotations):
    """Generate counting-based captions"""
    captions = []
    
    # Count by color
    color_counts = {}
    for ann in annotations:
        color = ann['color']
        color_counts[color] = color_counts.get(color, 0) + 1
    
    for color, count in color_counts.items():
        if count > 1:
            captions.append(f"Q: How many {color} shapes are there? A: There are {count} {color} shapes.")
    
    # Count by shape
    shape_counts = {}
    for ann in annotations:
        shape = ann['shape']
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    
    for shape, count in shape_counts.items():
        if count > 1:
            captions.append(f"Q: How many {shape}s are in the image? A: There are {count} {shape}s.")
    
    return captions


def generate_comparison_captions(annotations):
    """Generate size comparison captions"""
    captions = []
    
    if len(annotations) >= 2:
        ann1, ann2 = random.sample(annotations, 2)
        
        if ann1['size'] > ann2['size']:
            captions.append(
                f"Q: Is the {ann1['color']} {ann1['shape']} larger than the {ann2['color']} {ann2['shape']}? "
                f"A: Yes, the {ann1['color']} {ann1['shape']} is larger."
            )
        else:
            captions.append(
                f"Q: Is the {ann1['color']} {ann1['shape']} larger than the {ann2['color']} {ann2['shape']}? "
                f"A: No, the {ann2['color']} {ann2['shape']} is larger."
            )
    
    return captions


def generate_advanced_captions(annotations):
    """Generate advanced captions with spatial relationships, counting, and comparisons"""
    captions = []
    
    # Basic captions
    if len(annotations) == 1:
        ann = annotations[0]
        captions.append(f"A {ann['color']} {ann['shape']} in the {ann['position']}.")
    else:
        shape_descriptions = [f"a {ann['color']} {ann['shape']}" for ann in annotations]
        if len(shape_descriptions) > 1:
            captions.append(f"There are {', '.join(shape_descriptions[:-1])} and {shape_descriptions[-1]}.")
    
    # Spatial relationships
    if len(annotations) >= 2:
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                rel = get_spatial_relationship(annotations[i], annotations[j])
                if rel:
                    captions.append(rel)
                    break
            if len([c for c in captions if "left of" in c or "right of" in c or "above" in c or "below" in c]) > 0:
                break
    
    # Counting questions
    captions.extend(generate_counting_captions(annotations))
    
    # Comparison questions
    captions.extend(generate_comparison_captions(annotations))
    
    return captions


def generate_advanced_image_and_caption(image_size, shapes, colors, positions, size_range, shapes_per_image_range):
    """Generate image with advanced caption variations"""
    img = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(img)
    
    num_shapes = random.randint(shapes_per_image_range[0], shapes_per_image_range[1])
    
    annotations = []
    
    for _ in range(num_shapes):
        shape_type = random.choice(shapes)
        color_name = random.choice(list(colors.keys()))
        color_rgb = tuple(colors[color_name])
        position_name = random.choice(positions)
        size = random.randint(size_range[0], size_range[1])
        
        position_coords = get_position_coords(position_name, image_size, (size, size))
        position_coords = (
            position_coords[0] + random.randint(-20, 20),
            position_coords[1] + random.randint(-20, 20)
        )
        
        draw_shape(draw, shape_type, position_coords, size, color_rgb)
        
        annotations.append({
            'shape': shape_type,
            'color': color_name,
            'position': position_name,
            'size': size,
            'exact_coords': position_coords
        })
    
    captions = generate_advanced_captions(annotations)
    
    return img, annotations, captions


def main():
    parser = argparse.ArgumentParser(description='Generate advanced synthetic VLM dataset')
    parser.add_argument('--num-images', type=int, default=10000, help='Number of images to generate')
    parser.add_argument('--output-dir', type=str, default='advanced_vlm_dataset', help='Output directory')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        num_images = config.get('num_images', args.num_images)
        output_dir = config.get('output_dir', args.output_dir)
        image_size = tuple(config.get('image_size', [512, 512]))
        shapes = config.get('shapes', SHAPES)
        colors = config.get('colors', COLORS)
        positions = config.get('positions', POSITIONS)
        size_range = config.get('size_range', [30, 80])
        shapes_per_image = config.get('shapes_per_image', [1, 3])
        seed = config.get('random_seed', args.seed)
    else:
        num_images = args.num_images
        output_dir = args.output_dir
        image_size = (512, 512)
        shapes = SHAPES
        colors = COLORS
        positions = POSITIONS
        size_range = [30, 80]
        shapes_per_image = [1, 3]
        seed = args.seed
    
    random.seed(seed)
    np.random.seed(seed)
    
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    dataset = []
    
    print(f"Generating {num_images} advanced synthetic images...")
    for i in tqdm(range(num_images)):
        img, annotations, captions = generate_advanced_image_and_caption(
            image_size, shapes, colors, positions, size_range, shapes_per_image
        )
        
        img_filename = f"image_{i:06d}.png"
        img.save(f"{output_dir}/images/{img_filename}")
        
        dataset.append({
            'image_id': i,
            'image_filename': img_filename,
            'annotations': annotations,
            'captions': captions
        })
    
    with open(f"{output_dir}/dataset.json", 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ Advanced dataset generation complete!")
    print(f"  Images saved to: {output_dir}/images/")
    print(f"  Metadata saved to: {output_dir}/dataset.json")
    print(f"  Total images: {num_images}")


if __name__ == "__main__":
    main()