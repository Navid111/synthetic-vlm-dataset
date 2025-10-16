import json
import argparse
from pathlib import Path


def convert_to_llava(dataset):
    """Convert to LLaVA conversational format"""
    llava_data = []
    
    for item in dataset:
        # Use the first (simple) caption for the conversation
        caption = item['captions'][0]
        
        llava_item = {
            "id": f"image_{item['image_id']:06d}",
            "image": f"images/{item['image_filename']}",
            "conversations": [
                {
                    "from": "human",
                    "value": "Describe this image."
                },
                {
                    "from": "gpt",
                    "value": caption
                }
            ]
        }
        llava_data.append(llava_item)
    
    return llava_data


def convert_to_coco(dataset):
    """Convert to COCO object detection format"""
    images = []
    annotations = []
    categories = []
    
    # Build categories from unique shapes
    shape_types = set()
    for item in dataset:
        for ann in item['annotations']:
            shape_types.add(ann['shape'])
    
    category_map = {}
    for idx, shape in enumerate(sorted(shape_types), 1):
        categories.append({"id": idx, "name": shape})
        category_map[shape] = idx
    
    annotation_id = 1
    for item in dataset:
        # Add image info
        images.append({
            "id": item['image_id'],
            "file_name": item['image_filename'],
            "height": 512,
            "width": 512
        })
        
        # Add annotations
        for ann in item['annotations']:
            x, y = ann['exact_coords']
            size = ann['size']
            
            annotations.append({
                "id": annotation_id,
                "image_id": item['image_id'],
                "category_id": category_map[ann['shape']],
                "bbox": [x - size, y - size, size * 2, size * 2],
                "area": (size * 2) ** 2
            })
            annotation_id += 1
    
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }


def convert_to_huggingface(dataset):
    """Convert to Hugging Face simple format"""
    hf_data = []
    
    for item in dataset:
        # Use the first (simple) caption
        caption = item['captions'][0]
        
        hf_item = {
            "image": f"images/{item['image_filename']}",
            "text": caption
        }
        hf_data.append(hf_item)
    
    return hf_data


def main():
    parser = argparse.ArgumentParser(description='Convert dataset to different formats')
    parser.add_argument('--input', type=str, required=True, help='Input dataset.json file')
    parser.add_argument('--format', type=str, required=True, choices=['llava', 'coco', 'huggingface'], 
                        help='Output format')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.input}...")
    with open(args.input, 'r') as f:
        dataset = json.load(f)
    
    # Convert to specified format
    print(f"Converting to {args.format} format...")
    if args.format == 'llava':
        converted_data = convert_to_llava(dataset)
    elif args.format == 'coco':
        converted_data = convert_to_coco(dataset)
    elif args.format == 'huggingface':
        converted_data = convert_to_huggingface(dataset)
    
    # Save converted dataset
    print(f"Saving to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"✓ Conversion complete! Saved to {args.output}")


if __name__ == "__main__":
    main()