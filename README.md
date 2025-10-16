# Synthetic VLM Dataset Generator

A Python tool for generating synthetic datasets to train and fine-tune Vision-Language Models (VLMs). Creates images with random shapes, colors, and positions along with corresponding natural language descriptions.

## Features

- 🎨 Generate images with multiple shapes (circles, rectangles, triangles, ellipses, polygons)
- 🌈 8 color options with full RGB control
- 📍 Positional awareness (top-left, top-right, bottom-left, bottom-right, center)
- 💬 Multiple caption styles (simple, detailed, Q&A format)
- 🔄 Export to multiple VLM formats (LLaVA, COCO, Hugging Face)
- ⚙️ YAML-based configuration system
- 📊 Built-in visualization tools
- 🚀 Production-ready with progress tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/Navid111/synthetic-vlm-dataset.git
cd synthetic-vlm-dataset

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Generate 1000 images with default settings
python generate_dataset.py --num-images 1000 --output-dir ./my_dataset

# Visualize some samples
python visualize_dataset.py --dataset-dir ./my_dataset --num-samples 5

# Convert to LLaVA format
python convert_format.py --input ./my_dataset/dataset.json --format llava --output ./my_dataset/llava_format.json
```

## Usage Examples

### Basic Generation

```bash
python generate_dataset.py --num-images 10000 --output-dir ./dataset --seed 42
```

### Using Configuration File

```bash
python generate_dataset.py --config configs/dataset_config.yaml
```

### Advanced Generation with Spatial Relationships

```bash
python advanced_generator.py --num-images 5000 --output-dir ./advanced_dataset
```

### Format Conversion

```bash
# Convert to LLaVA format
python convert_format.py --input dataset.json --format llava --output llava_dataset.json

# Convert to COCO format
python convert_format.py --input dataset.json --format coco --output coco_dataset.json

# Convert to Hugging Face format
python convert_format.py --input dataset.json --format huggingface --output hf_dataset.json
```

## Configuration

Edit `configs/dataset_config.yaml` to customize generation:

```yaml
num_images: 10000
image_size: [512, 512]
output_dir: "synthetic_vlm_dataset"

shapes: ["circle", "rectangle", "triangle", "ellipse", "polygon"]
colors:
  red: [255, 0, 0]
  blue: [0, 0, 255]
  green: [0, 255, 0]
  yellow: [255, 255, 0]
  purple: [128, 0, 128]
  orange: [255, 165, 0]
  pink: [255, 192, 203]
  cyan: [0, 255, 255]

positions: ["top-left", "top-right", "bottom-left", "bottom-right", "center"]
size_range: [30, 80]
shapes_per_image: [1, 3]
random_seed: 42
```

## Dataset Formats

### LLaVA Format (Conversational)
```json
[
  {
    "id": "image_000001",
    "image": "images/image_000001.png",
    "conversations": [
      {
        "from": "human",
        "value": "Describe this image."
      },
      {
        "from": "gpt",
        "value": "A red circle in the top-left and a blue rectangle in the center."
      }
    ]
  }
]
```
## Model Evaluation

Test Vision-Language Models (VLMs) on your synthetic dataset and measure their accuracy!

### Supported Models

- **BLIP** (Salesforce/blip-vqa-base) - Fast, accurate VQA model
- **BLIP-2** (Salesforce/blip2-opt-2.7b) - Improved version with better reasoning
- **InstructBLIP** (Salesforce/instructblip-vicuna-7b) - Instruction-tuned model

### Installation

Install evaluation dependencies:

```bash
pip install -r requirements.txt
```

### Quick Start

```bash
# Evaluate BLIP on your dataset
python evaluate_model.py --dataset-dir ./my_dataset --model blip

# Evaluate on 100 samples only
python evaluate_model.py --dataset-dir ./my_dataset --model blip --num-samples 100

# Use specific GPU
python evaluate_model.py --dataset-dir ./my_dataset --model blip --device cuda
```

### Output

The evaluation generates:

1. **CSV Results** (`evaluation_results.csv`):
   - Every question asked
   - Expected vs actual answers
   - Correctness and similarity scores
   - Question types

2. **JSON Summary** (`evaluation_results_summary.json`):
   - Overall accuracy
   - Accuracy by question type
   - Statistics and metrics

### Analyze Results

Generate visualizations and analysis:

```bash
python analyze_results.py --results evaluation_results.csv --output-dir ./analysis
```

This creates:
- Accuracy charts by question type
- Similarity score distributions
- Example predictions (correct/incorrect)
- Detailed analysis report

### Compare Multiple Models

```bash
python compare_models.py --dataset-dir ./my_dataset --models blip blip2 --output comparison.csv
```

### Example Output

```
==========================================================
EVALUATION SUMMARY
==========================================================
Model: blip
Overall Accuracy: 87.3%
Total Questions: 3000
Correct Answers: 2619

Accuracy by Question Type:
  color          : 95.2% (800 questions)
  counting       : 82.1% (600 questions)
  position       : 88.4% (700 questions)
  shape          : 91.3% (500 questions)
  description    : 75.8% (400 questions)
==========================================================
```

### Tips

- Start with BLIP (fastest and requires less memory)
- Use `--num-samples` to test on a subset first
- BLIP-2 and InstructBLIP require more GPU memory
- Use CPU mode if you don't have a GPU (slower but works)

### COCO Format (Object Detection)
```json
{
  "images": [{"id": 1, "file_name": "image_000001.png", "height": 512, "width": 512}],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 100, 80, 80]
    }
  ],
  "categories": [{"id": 1, "name": "circle"}]
}
```

### Hugging Face Format (Simple)
```json
[
  {
    "image": "images/image_000001.png",
    "text": "A red circle in the top-left."
  }
]
```

## Training Tasks

This dataset supports various VLM training tasks:

1. **Object Recognition**: "What shapes are in this image?"
2. **Color Identification**: "What color is the circle?"
3. **Spatial Reasoning**: "Which shape is in the top-right?"
4. **Counting**: "How many red shapes are there?"
5. **Size Comparison**: "Is the circle larger than the square?"
6. **Spatial Relationships**: "What is to the left of the triangle?"

## Tips for VLM Fine-tuning

- **Start small**: Begin with 1,000-5,000 images to test your pipeline
- **Balance your data**: Ensure good distribution of shapes, colors, and positions
- **Vary caption styles**: Include different question types for robust learning
- **Augment strategically**: Add noise, rotations, or occlusions for robustness
- **Validate regularly**: Hold out 10-20% for validation
- **Scale gradually**: Increase dataset size based on model performance

## Customization

### Adding New Shapes

Edit `generate_dataset.py` and add to the `draw_shape()` function:

```python
elif shape_type == 'hexagon':
    points = []
    for i in range(6):
        angle = (2 * np.pi * i) / 6
        px = x + size * np.cos(angle)
        py = y + size * np.sin(angle)
        points.append((px, py))
    draw.polygon(points, fill=color)
```

### Adding New Colors

Update `configs/dataset_config.yaml`:

```yaml
colors:
  brown: [165, 42, 42]
  gray: [128, 128, 128]
```

## Example Outputs

Sample images and captions can be found in the `examples/` directory after generation.

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Feel free to open issues or submit pull requests.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{synthetic_vlm_dataset,
  author = {Navid111},
  title = {Synthetic VLM Dataset Generator},
  year = {2025},
  url = {https://github.com/Navid111/synthetic-vlm-dataset}
}
```