import random
import json
import os
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# Small standalone generator for Data Flow Diagram (DFD) style images
# Produces: PNG images, `dataset.json` containing annotations and QA pairs


DEFAULT_COLORS = {
    'blue': (70, 130, 180),
    'yellow': (255, 215, 0),
    'red': (220, 20, 60),
    'green': (34, 139, 34),
    'purple': (138, 43, 226),
}


def draw_box(draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int], fill_color: Tuple[int, int, int], outline=(0, 0, 0), text: str = None):
    draw.rectangle(bbox, fill=fill_color, outline=outline, width=2)
    if text:
        font = ImageFont.load_default()
        try:
            tb = draw.textbbox((0, 0), text, font=font)
            text_w = tb[2] - tb[0]
            text_h = tb[3] - tb[1]
        except Exception:
            # fallback
            try:
                text_w, text_h = font.getsize(text)
            except Exception:
                text_w, text_h = (len(text) * 6, 10)

        x0, y0, x1, y1 = bbox
        tx = x0 + max(4, (x1 - x0 - text_w) // 2)
        ty = y0 + max(2, (y1 - y0 - text_h) // 2)
        draw.text((tx, ty), text, fill=(0, 0, 0), font=font)


def draw_arrow(draw: ImageDraw.ImageDraw, start: Tuple[int, int], end: Tuple[int, int], color=(0, 0, 0), width=2):
    # Line
    draw.line([start, end], fill=color, width=width)
    # Arrowhead
    import math
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_len = 10
    left = (end[0] - head_len * math.cos(angle - math.pi / 8), end[1] - head_len * math.sin(angle - math.pi / 8))
    right = (end[0] - head_len * math.cos(angle + math.pi / 8), end[1] - head_len * math.sin(angle + math.pi / 8))
    draw.polygon([end, left, right], fill=color)


def generate_questions_from_dfd(nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
    questions = []
    # Counting
    questions.append({
        'question': 'How many boxes are there?',
        'answer': str(len(nodes)),
        'question_type': 'counting'
    })
    questions.append({
        'question': 'How many arrows are there?',
        'answer': str(len(edges)),
        'question_type': 'counting'
    })

    # What does <color> box say?
    color_to_nodes = {}
    for n in nodes:
        color_to_nodes.setdefault(n['color'], []).append(n)

    for color in ['blue', 'yellow', 'red', 'green', 'purple']:
        if color in color_to_nodes:
            questions.append({
                'question': f'What does the {color} colored box say?',
                'answer': color_to_nodes[color][0]['label'],
                'question_type': 'content'
            })
            break

    # Where does the yellow box point to?
    if 'yellow' in color_to_nodes:
        yellow = color_to_nodes['yellow'][0]
        outs = [e for e in edges if e['from'] == yellow['id']]
        if outs:
            dest = next(n for n in nodes if n['id'] == outs[0]['to'])
            questions.append({
                'question': 'Where does the yellow colored box point to?',
                'answer': dest['label'],
                'question_type': 'position'
            })

    # Sample relation question
    if edges:
        e = edges[0]
        src = next(n for n in nodes if n['id'] == e['from'])
        dst = next(n for n in nodes if n['id'] == e['to'])
        questions.append({
            'question': f'What is the relation from {src["label"]} to {dst["label"]}?',
            'answer': e.get('label', 'data flow'),
            'question_type': 'relation'
        })

    return questions


def generate_dfd_image_and_annotation(image_size=(512, 512), colors=DEFAULT_COLORS, min_boxes=2, max_boxes=5, min_edges=1, max_edges=6) -> Tuple[Image.Image, List[Dict], List[Dict]]:
    w, h = image_size
    img = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(img)

    labels_pool = [
        'Process Orders', 'Validate Input', 'Store Data', 'Send Email',
        'Authenticate User', 'Generate Report', 'Receive Request', 'Transform Data'
    ]

    num_boxes = random.randint(min_boxes, max_boxes)
    nodes = []
    for i in range(num_boxes):
        bw = random.randint(120, 220)
        bh = random.randint(40, 90)
        x = random.randint(30, max(30, w - bw - 30))
        y = random.randint(30, max(30, h - bh - 30))
        color_name = random.choice(list(colors.keys()))
        color_rgb = tuple(colors[color_name])
        label = random.choice(labels_pool)
        bbox = (x, y, x + bw, y + bh)
        draw_box(draw, bbox, fill_color=color_rgb, outline=(0, 0, 0), text=label)
        nodes.append({'id': i, 'label': label, 'color': color_name, 'bbox': bbox, 'center': (x + bw // 2, y + bh // 2)})

    # Edges
    max_possible = max(0, num_boxes * (num_boxes - 1))
    num_edges = random.randint(min_edges, min(max_edges, max_possible)) if max_possible > 0 else 0
    edges = []
    pairs = set()
    attempts = 0
    while len(edges) < num_edges and attempts < num_edges * 5:
        a = random.randrange(num_boxes)
        b = random.randrange(num_boxes)
        attempts += 1
        if a == b or (a, b) in pairs:
            continue
        pairs.add((a, b))
        start = nodes[a]['center']
        end = nodes[b]['center']
        draw_arrow(draw, start, end, color=(0, 0, 0), width=2)
        edges.append({'from': a, 'to': b, 'label': 'data flow'})

    questions = generate_questions_from_dfd(nodes, edges)

    annotations = []
    for n in nodes:
        annotations.append({'id': n['id'], 'label': n['label'], 'color': n['color'], 'bbox': n['bbox'], 'center': n['center']})

    return img, annotations, questions


def main():
    parser = argparse.ArgumentParser(description='Generate DFD-style synthetic dataset')
    parser.add_argument('--num-images', type=int, default=100, help='Number of images to generate')
    parser.add_argument('--output-dir', type=str, default='dfd_dataset', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--min-boxes', type=int, default=2)
    parser.add_argument('--max-boxes', type=int, default=5)
    parser.add_argument('--min-edges', type=int, default=1)
    parser.add_argument('--max-edges', type=int, default=6)
    parser.add_argument('--image-size', type=int, nargs=2, default=[512, 512])
    args = parser.parse_args()

    random.seed(args.seed)

    os.makedirs(f"{args.output_dir}/images", exist_ok=True)

    dataset = []
    print(f"Generating {args.num_images} DFD images into '{args.output_dir}'...")
    for i in tqdm(range(args.num_images)):
        img, annotations, questions = generate_dfd_image_and_annotation(
            image_size=tuple(args.image_size),
            colors=DEFAULT_COLORS,
            min_boxes=args.min_boxes,
            max_boxes=args.max_boxes,
            min_edges=args.min_edges,
            max_edges=args.max_edges,
        )

        filename = f"image_{i:06d}.png"
        img.save(f"{args.output_dir}/images/{filename}")
        dataset.append({'image_id': i, 'image_filename': filename, 'annotations': annotations, 'questions': questions})

    with open(f"{args.output_dir}/dataset.json", 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✓ DFD dataset generation complete. Saved {args.num_images} images to {args.output_dir}/images and metadata to {args.output_dir}/dataset.json")


if __name__ == '__main__':
    main()
