import numpy as np
from PIL import Image, ImageDraw
import random
import json
import os
import argparse
from tqdm import tqdm
import yaml
from typing import List, Dict

# Define shapes, colors, and positions
SHAPES = ['circle', 'rectangle', 'triangle', 'ellipse', 'polygon']
COLORS = {
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'green': (0, 255, 0),
    'yellow': (255, 255, 0),
    'purple': (128, 0, 128),
    'orange': (255, 165, 0),
    'pink': (255, 192, 203),
    'cyan': (0, 255, 255)
}
POSITIONS = ['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center']


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_position_coords(position, image_size, shape_size):
    """Convert position name to coordinates"""
    w, h = image_size
    sw, sh = shape_size
    
    positions = {
        'top-left': (sw + 50, sh + 50),
        'top-right': (w - sw - 50, sh + 50),
        'bottom-left': (sw + 50, h - sh - 50),
        'bottom-right': (w - sw - 50, h - sh - 50),
        'center': (w // 2, h // 2)
    }
    return positions[position]


def draw_shape(draw, shape_type, position, size, color):
    """Draw a specific shape at given position"""
    x, y = position
    
    if shape_type == 'circle':
        bbox = [x - size, y - size, x + size, y + size]
        draw.ellipse(bbox, fill=color)
    
    elif shape_type == 'rectangle':
        bbox = [x - size, y - size//2, x + size, y + size//2]
        draw.rectangle(bbox, fill=color)
    
    elif shape_type == 'triangle':
        points = [(x, y - size), (x - size, y + size), (x + size, y + size)]
        draw.polygon(points, fill=color)
    
    elif shape_type == 'ellipse':
        bbox = [x - size, y - size//2, x + size, y + size//2]
        draw.ellipse(bbox, fill=color)
    
    elif shape_type == 'polygon':
        # Pentagon
        points = []
        for i in range(5):
            angle = (2 * np.pi * i) / 5 - np.pi / 2
            px = x + size * np.cos(angle)
            py = y + size * np.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=color)

def generate_questions_from_annotations(annotations: List[Dict]) -> List[Dict]:
    """Generate questions and answers from annotations"""
    questions = []
    
    # 1. Description question
    if len(annotations) == 1:
        ann = annotations[0]
        desc = f"A {ann['color']} {ann['shape']}."
    else:
        shape_descs = [f"a {ann['color']} {ann['shape']}" for ann in annotations]
        desc = f"There are {', '.join(shape_descs[:-1])} and {shape_descs[-1]}."
    
    questions.append({
        "question": "Describe this image.",
        "answer": desc,
        "question_type": "description"
    })
    
    # 2. Color questions for each shape
    for ann in annotations:
        questions.append({
            "question": f"What color is the {ann['shape']}?",
            "answer": ann['color'],
            "question_type": "color"
        })
    
    # 3. Total count question
    questions.append({
        "question": "How many shapes are in this image?",
        "answer": str(len(annotations)),
        "question_type": "counting"
    })
    
    # 4. Color-specific counting
    color_counts = {}
    for ann in annotations:
        color_counts[ann['color']] = color_counts.get(ann['color'], 0) + 1
    
    for color, count in color_counts.items():
        questions.append({
            "question": f"How many {color} shapes are there?",
            "answer": str(count),
            "question_type": "counting"
        })
    
    # 5. Position questions
    for ann in annotations:
        questions.append({
            "question": f"Where is the {ann['color']} {ann['shape']}?",
            "answer": ann['position'],
            "question_type": "position"
        })
    
    return questions
'''
def generate_captions(annotations):
    """Generate multiple caption styles for VLM training"""
    captions = []
    
    # Simple description
    if len(annotations) == 1:
        ann = annotations[0]
        simple = f"A {ann['color']} {ann['shape']} in the {ann['position']}."
    else:
        shape_descriptions = [f"a {ann['color']} {ann['shape']}" for ann in annotations]
        if len(shape_descriptions) > 1:
            simple = f"There are {', '.join(shape_descriptions[:-1])} and {shape_descriptions[-1]}."
        else:
            simple = f"There is {shape_descriptions[0]}."
    
    captions.append(simple)
    
    # Detailed description
    detailed_parts = []
    for ann in annotations:
        detailed_parts.append(
            f"a {ann['size']}-pixel {ann['color']} {ann['shape']} located in the {ann['position']}"
        )
    detailed = f"The image contains {', '.join(detailed_parts)}."
    captions.append(detailed)
    
    # Question-answer format
    if len(annotations) > 0:
        ann = random.choice(annotations)
        qa = f"Q: What color is the {ann['shape']}? A: The {ann['shape']} is {ann['color']}."
        captions.append(qa)
    
    return captions
'''

def generate_image_and_caption(image_size, shapes, colors, positions, size_range, shapes_per_image_range):
    """Generate a single image with random shapes and corresponding caption"""
    # Create blank image with white background
    img = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Randomly decide how many shapes
    num_shapes = random.randint(shapes_per_image_range[0], shapes_per_image_range[1])
    
    annotations = []
    
    for _ in range(num_shapes):
        shape_type = random.choice(shapes)
        color_name = random.choice(list(colors.keys()))
        color_rgb = tuple(colors[color_name])
        position_name = random.choice(positions)
        size = random.randint(size_range[0], size_range[1])
        
        position_coords = get_position_coords(position_name, image_size, (size, size))
        # Add some randomness to exact position
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
    
    # Generate natural language captions
    captions = generate_questions_from_annotations(annotations)
    
    return img, annotations, captions


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic VLM dataset')
    parser.add_argument('--num-images', type=int, default=10000, help='Number of images to generate')
    parser.add_argument('--output-dir', type=str, default='synthetic_vlm_dataset', help='Output directory')
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
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    dataset = []
    
    print(f"Generating {num_images} synthetic images...")
    for i in tqdm(range(num_images)):
        img, annotations, captions = generate_image_and_caption(
            image_size, shapes, colors, positions, size_range, shapes_per_image
        )
        
        # Save image
        img_filename = f"image_{i:06d}.png"
        img.save(f"{output_dir}/images/{img_filename}")
        # Generate all questions upfront
        questions = generate_questions_from_annotations(annotations)
        
        # Store metadata
        dataset.append({
            'image_id': i,
            'image_filename': img_filename,
            'annotations': annotations,
            'captions': captions,
            'questions': questions
        })
    
    # Save dataset metadata
    with open(f"{output_dir}/dataset.json", 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ Dataset generation complete!")
    print(f"  Images saved to: {output_dir}/images/")
    print(f"  Metadata saved to: {output_dir}/dataset.json")
    print(f"  Total images: {num_images}")


if __name__ == "__main__":
    main()