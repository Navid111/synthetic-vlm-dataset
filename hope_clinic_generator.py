import random
import json
import os
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import math

# --- Configuration: Hope Clinic Specifics ---

# 1. Define the specific nodes exactly as they appear in the Hope Clinic DFD
HOPE_CLINIC_NODES = [
    # External Entities
    {'id': 'LISA', 'label': 'Lisa', 'type': 'entity', 'color': 'white'},
    {'id': 'CLARA', 'label': 'Clara', 'type': 'entity', 'color': 'white'},
    {'id': 'SUSAN', 'label': 'Susan', 'type': 'entity', 'color': 'white'},
    {'id': 'FRED', 'label': 'Fred', 'type': 'entity', 'color': 'white'},
    {'id': 'TOM', 'label': 'Tom', 'type': 'entity', 'color': 'white'},

    # Processes
    {'id': 'P1', 'label': '1. Appointment\nManagement', 'type': 'process', 'color': '#B0C4DE'},
    {'id': 'P2', 'label': '2. Expense/\nPayment Mgmt', 'type': 'process', 'color': '#B0C4DE'},
    {'id': 'P3', 'label': '3. Patient History\nManagement', 'type': 'process', 'color': '#B0C4DE'},
    {'id': 'P4', 'label': '4. Salary\nDisbursements', 'type': 'process', 'color': '#B0C4DE'},
    {'id': 'P5', 'label': '5. Accounting/\nReporting', 'type': 'process', 'color': '#B0C4DE'},

    # Data Stores
    {'id': 'D1', 'label': 'D1 | Appointments', 'type': 'datastore', 'color': '#E0E0E0'},
    {'id': 'D2', 'label': 'D2 | Expense', 'type': 'datastore', 'color': '#E0E0E0'},
    {'id': 'D3', 'label': 'D3 | Patient History', 'type': 'datastore', 'color': '#E0E0E0'},
    {'id': 'D4', 'label': 'D4 | Salary Disb.', 'type': 'datastore', 'color': '#E0E0E0'},
]

# 2. Define the exact connections (Topology)
HOPE_CLINIC_EDGES = [
    {'from': 'LISA', 'to': 'P1', 'label': ''},
    {'from': 'P1', 'to': 'D1', 'label': 'Appointment'},
    {'from': 'D1', 'to': 'P1', 'label': 'Availability'},
    {'from': 'D1', 'to': 'P2', 'label': 'Patient Info'},
    {'from': 'CLARA', 'to': 'P2', 'label': ''},
    {'from': 'P2', 'to': 'D2', 'label': 'Payment'},
    {'from': 'D1', 'to': 'P3', 'label': 'Patient Info'},
    {'from': 'SUSAN', 'to': 'P3', 'label': ''},
    {'from': 'P3', 'to': 'D3', 'label': ''},
    {'from': 'FRED', 'to': 'P4', 'label': ''},
    {'from': 'P4', 'to': 'D4', 'label': ''},
    {'from': 'D2', 'to': 'P5', 'label': 'Expenses'},
    {'from': 'D4', 'to': 'P5', 'label': 'Disbursements'},
    {'from': 'P5', 'to': 'TOM', 'label': 'Report'},
    {'from': 'P5', 'to': 'D4', 'label': 'Update'}, # Based on interpretation
]

# --- Drawing Utilities ---

def check_overlap(new_box, existing_boxes, buffer=10):
    """Checks if new_box overlaps with any existing_boxes with a buffer."""
    nx0, ny0, nx1, ny1 = new_box
    for box in existing_boxes:
        ex0, ey0, ex1, ey1 = box
        # Check for intersection
        if not (nx1 + buffer < ex0 or nx0 - buffer > ex1 or ny1 + buffer < ey0 or ny0 - buffer > ey1):
            return True
    return False

def get_random_layout(image_size, nodes):
    """
    Randomly places nodes on the canvas ensuring no overlaps.
    Returns a dict mapping node_id -> bbox (x0, y0, x1, y1)
    """
    w, h = image_size
    layout = {}
    existing_boxes = []
    
    # Sort nodes by size preference (Process/Stores usually larger) to place them first? 
    # Or just random shuffle to ensure true layout variance.
    shuffled_nodes = nodes.copy()
    random.shuffle(shuffled_nodes)

    for node in shuffled_nodes:
        # Define approximate sizes based on type
        if node['type'] == 'process':
            bw, bh = 140, 80
        elif node['type'] == 'datastore':
            bw, bh = 160, 50
        else: # Entity
            bw, bh = 100, 50
        
        placed = False
        attempts = 0
        while not placed and attempts < 200:
            x = random.randint(20, w - bw - 20)
            y = random.randint(20, h - bh - 20)
            bbox = (x, y, x + bw, y + bh)
            
            if not check_overlap(bbox, existing_boxes):
                layout[node['id']] = bbox
                existing_boxes.append(bbox)
                placed = True
            attempts += 1
            
        if not placed:
            # Fallback: Just place it somewhere central if we fail (rare with 512x512)
            layout[node['id']] = (0, 0, bw, bh) 
            
    return layout

def draw_dfd_node(draw, node_info, bbox):
    """Draws the node based on its specific DFD type."""
    x0, y0, x1, y1 = bbox
    color = node_info['color']
    label = node_info['label']
    ntype = node_info['type']
    
    # Draw Shape
    if ntype == 'process':
        # Rounded Rectangle
        draw.rounded_rectangle(bbox, radius=15, fill=color, outline='black', width=2)
        # Add a line for the "ID" part of the process if desired, but simple text is fine
        
    elif ntype == 'datastore':
        # Open rectangle (or rect with double side line)
        draw.rectangle(bbox, fill=color, outline='black', width=2)
        # Distinctive double line on left
        draw.line([(x0 + 10, y0), (x0 + 10, y1)], fill='black', width=2)
        
    else: # Entity
        # Sharp Rectangle
        draw.rectangle(bbox, fill=color, outline='black', width=2)

    # Draw Text
    font = ImageFont.load_default()
    # Simple centered text wrapper
    lines = label.split('\n')
    
    # Calculate total text height
    total_h = len(lines) * 12 # approx line height
    start_y = y0 + (y1 - y0 - total_h) / 2
    
    for i, line in enumerate(lines):
        # We use textbbox to center
        try:
            tb = draw.textbbox((0, 0), line, font=font)
            line_w = tb[2] - tb[0]
            line_x = x0 + (x1 - x0 - line_w) / 2
            draw.text((line_x, start_y + i * 14), line, fill='black', font=font)
        except AttributeError:
             # Fallback for older PIL
            draw.text((x0 + 5, start_y + i * 14), line, fill='black', font=font)

def draw_connection(draw, start_bbox, end_bbox, label):
    """Draws an arrow between the centers of two bboxes."""
    sx0, sy0, sx1, sy1 = start_bbox
    ex0, ey0, ex1, ey1 = end_bbox
    
    sc = ((sx0 + sx1) / 2, (sy0 + sy1) / 2)
    ec = ((ex0 + ex1) / 2, (ey0 + ey1) / 2)
    
    # Draw Line
    draw.line([sc, ec], fill='black', width=2)
    
    # Draw Arrowhead at 'ec'
    angle = math.atan2(ec[1] - sc[1], ec[0] - sc[0])
    head_len = 10
    # Stop the arrow slightly before the center so it doesn't overlap text
    # We ideally want to stop at the bbox edge, but center-to-center is acceptable for basic VLM training
    # Refinement: Clip to bbox would be better, but keeping it simple for now.
    
    arrow_tip = (ec[0] - 20 * math.cos(angle), ec[1] - 20 * math.sin(angle)) # Back off a bit
    
    left = (arrow_tip[0] - head_len * math.cos(angle - math.pi / 6), arrow_tip[1] - head_len * math.sin(angle - math.pi / 6))
    right = (arrow_tip[0] - head_len * math.cos(angle + math.pi / 6), arrow_tip[1] - head_len * math.sin(angle + math.pi / 6))
    
    draw.polygon([arrow_tip, left, right], fill='black')
    
    # Draw Label (Midpoint)
    if label:
        mx, my = (sc[0] + ec[0]) / 2, (sc[1] + ec[1]) / 2
        font = ImageFont.load_default()
        draw.rectangle((mx-2, my-2, mx+40, my+10), fill='white') # Tiny background for legibility
        draw.text((mx, my), label, fill='black', font=font)


# --- QA Generation ---

def generate_qa_for_image(layout):
    """
    Generates QA pairs focused on entity identification.
    """
    questions = []
    
    # 1. Count entities question
    questions.append({
        'question': 'How many entities are there?',
        'answer': '5',
        'type': 'counting'
    })
    
    # 2. Name each entity question
    entity_names = ['Lisa', 'Clara', 'Susan', 'Fred', 'Tom']
    questions.append({
        'question': 'Name each entity.',
        'answer': ', '.join(entity_names),
        'type': 'naming'
    })

    return questions

# --- Main Generator Loop ---

def generate_dataset(num_images, output_dir, seed=42):
    random.seed(seed)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    dataset = []
    image_size = (800, 600) # Slightly larger to fit everything nicely
    
    print(f"Generating {num_images} Randomized Hope Clinic DFDs...")
    
    for i in tqdm(range(num_images)):
        # 1. Create Image
        img = Image.new('RGB', image_size, 'white')
        draw = ImageDraw.Draw(img)
        
        # 2. Randomize Layout
        layout = get_random_layout(image_size, HOPE_CLINIC_NODES)
        
        # 3. Draw Edges FIRST (so they are behind nodes if overlap occurs)
        for edge in HOPE_CLINIC_EDGES:
            start_box = layout.get(edge['from'])
            end_box = layout.get(edge['to'])
            if start_box and end_box:
                draw_connection(draw, start_box, end_box, edge['label'])
                
        # 4. Draw Nodes
        for node in HOPE_CLINIC_NODES:
            bbox = layout.get(node['id'])
            draw_dfd_node(draw, node, bbox)
            
        # 5. Generate QA
        qa = generate_qa_for_image(layout)
        
        # 6. Save
        filename = f"hope_clinic_{i:04d}.png"
        img.save(os.path.join(output_dir, "images", filename))
        
        # 7. Metadata
        dataset.append({
            'image_id': i,
            'file_name': filename,
            'nodes': [{'id': k, 'bbox': v} for k,v in layout.items()],
            'qa_pairs': qa
        })
        
    # Save JSON
    with open(os.path.join(output_dir, "dataset.json"), 'w') as f:
        json.dump(dataset, f, indent=2)
        
    print(f"Done! Dataset saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-images', type=int, default=10, help='Number of images to generate')
    parser.add_argument('--output-dir', type=str, default='hope_clinic_dataset', help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    generate_dataset(args.num_images, args.output_dir, args.seed)