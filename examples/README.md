# Example Outputs

This directory contains example outputs from the synthetic VLM dataset generator.

## Sample Images

After running the generator, you'll see images like:

- **Simple scene**: A single red circle in the center
- **Multi-shape scene**: A blue rectangle in the top-left, a yellow triangle in the bottom-right
- **Complex scene**: Three shapes with various colors and positions

## Sample Captions

### Simple Description
```
A red circle in the top-left.
```

### Detailed Description
```
The image contains a 45-pixel red circle located in the top-left, a 60-pixel blue rectangle located in the center.
```

### Question-Answer Format
```
Q: What color is the circle? A: The circle is red.
```

### Spatial Relationships (Advanced)
```
The red circle is to the left of the blue rectangle.
```

### Counting Questions (Advanced)
```
Q: How many red shapes are there? A: There are 2 red shapes.
```

### Size Comparisons (Advanced)
```
Q: Is the red circle larger than the blue rectangle? A: No, the blue rectangle is larger.
```

## Running Examples

```bash
# Generate 100 sample images
python generate_dataset.py --num-images 100 --output-dir examples/basic_samples

# Generate advanced samples
python advanced_generator.py --num-images 100 --output-dir examples/advanced_samples

# Visualize the samples
python visualize_dataset.py --dataset-dir examples/basic_samples --num-samples 10
```