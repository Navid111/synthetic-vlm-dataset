import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import re
from fuzzywuzzy import fuzz
from models import create_model


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    # Convert to lowercase and strip whitespace
    answer = answer.lower().strip()
    # Remove punctuation
    answer = re.sub(r'[^\w\s]', '', answer)
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    return answer


def extract_key_info(answer: str) -> set:
    """Extract key information from answer (colors, shapes, numbers)"""
    colors = {'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan'}
    shapes = {'circle', 'rectangle', 'triangle', 'ellipse', 'polygon', 'square'}
    
    words = set(normalize_answer(answer).split())
    return words.intersection(colors.union(shapes))


def compare_answers(expected: str, actual: str) -> Tuple[bool, float]:
    """Compare two answers and return (is_correct, similarity_score)"""
    expected_norm = normalize_answer(expected)
    actual_norm = normalize_answer(actual)
    
    # Exact match
    if expected_norm == actual_norm:
        return True, 1.0
    
    # Check for number synonyms (2 vs two)
    number_map = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3',
        'four': '4', 'five': '5', 'six': '6', 'seven': '7',
        'eight': '8', 'nine': '9', 'ten': '10'
    }
    
    expected_with_nums = expected_norm
    actual_with_nums = actual_norm
    for word, num in number_map.items():
        expected_with_nums = expected_with_nums.replace(word, num)
        actual_with_nums = actual_with_nums.replace(word, num)
    
    if expected_with_nums == actual_with_nums:
        return True, 0.95
    
    # Fuzzy string matching
    similarity = fuzz.ratio(expected_norm, actual_norm) / 100.0
    
    # Check if key information matches (colors, shapes, numbers)
    expected_keys = extract_key_info(expected)
    actual_keys = extract_key_info(actual)
    
    if expected_keys and actual_keys:
        key_overlap = len(expected_keys.intersection(actual_keys)) / len(expected_keys)
        similarity = max(similarity, key_overlap)
    
    # Consider correct if similarity > 0.8
    is_correct = similarity > 0.8
    
    return is_correct, similarity


def extract_qa_pairs(captions: List[str]) -> List[Tuple[str, str]]:
    """Extract Q&A pairs from captions"""
    qa_pairs = []
    
    for caption in captions:
        # Look for Q: ... A: ... pattern
        match = re.search(r'Q:\s*(.+?)\s*A:\s*(.+?)(?:\.|$)', caption)
        if match:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            qa_pairs.append((question, answer))
    
    return qa_pairs


def generate_questions_from_annotations(annotations: List[Dict]) -> List[Tuple[str, str]]:
    """Generate additional questions from annotations"""
    questions = []
    
    if not annotations:
        return questions
    
    # Describe the image
    if len(annotations) == 1:
        ann = annotations[0]
        desc = f"A {ann['color']} {ann['shape']}."
    else:
        shape_descs = [f"a {ann['color']} {ann['shape']}" for ann in annotations]
        desc = f"There are {', '.join(shape_descs[:-1])} and {shape_descs[-1]}."
    
    questions.append(("Describe this image.", desc))
    
    # Color questions for each shape
    for ann in annotations:
        questions.append((
            f"What color is the {ann['shape']}?",
            ann['color']
        ))
    
    # Count questions
    total_shapes = len(annotations)
    questions.append((
        "How many shapes are in this image?",
        str(total_shapes)
    ))
    
    # Color-specific counting
    color_counts = {}
    for ann in annotations:
        color_counts[ann['color']] = color_counts.get(ann['color'], 0) + 1
    
    for color, count in color_counts.items():
        questions.append((
            f"How many {color} shapes are there?",
            str(count)
        ))
    
    return questions


def detect_question_type(question: str) -> str:
    """Detect the type of question"""
    question_lower = question.lower()
    
    if 'color' in question_lower or 'what color' in question_lower:
        return 'color'
    elif 'how many' in question_lower:
        return 'counting'
    elif any(word in question_lower for word in ['where', 'position', 'top', 'bottom', 'left', 'right']):
        return 'position'
    elif 'shape' in question_lower:
        return 'shape'
    elif 'describe' in question_lower:
        return 'description'
    else:
        return 'other'


def load_dataset(dataset_dir: str) -> List[Dict]:
    """Load dataset from JSON file"""
    dataset_path = Path(dataset_dir) / 'dataset.json'
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    return dataset


def evaluate_model(dataset_dir: str, model_name: str, output_path: str, 
                   num_samples: int = None, device: str = 'auto') -> Dict:
    """Evaluate a VLM model on the dataset"""
    
    # Load dataset
    print(f"Loading dataset from {dataset_dir}...")
    dataset = load_dataset(dataset_dir)
    
    if num_samples:
        dataset = dataset[:num_samples]
    
    print(f"Evaluating on {len(dataset)} images")
    
    # Load model
    model = create_model(model_name, device)
    
    # Prepare results
    results = []
    stats_by_type = {}
    
    # Process each image
    images_dir = Path(dataset_dir) / 'images'
    for item in tqdm(dataset, desc="Evaluating"):
        image_path = images_dir / item['image_filename']
    
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load {image_path}: {e}")
            continue
    
    for qa in item['questions']:
        question = qa['question']
        expected_answer = qa['answer']
        question_type = qa['question_type']
        
        try:
            model_answer = model.answer_question(image, question)
            is_correct, similarity = compare_answers(expected_answer, model_answer)
            
            results.append({
                'image_id': item['image_id'],
                'image_filename': item['image_filename'],
                'question': question,
                'expected_answer': expected_answer,
                'model_answer': model_answer,
                'correct': is_correct,
                'similarity_score': similarity,
                'question_type': question_type
            })
            
            # Update stats
            if question_type not in stats_by_type:
                stats_by_type[question_type] = {'correct': 0, 'total': 0}
            
            stats_by_type[question_type]['total'] += 1
            if is_correct:
                stats_by_type[question_type]['correct'] += 1
                
        except Exception as e:
            print(f"Warning: Error processing question '{question}': {e}")
            continue
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    
    # Calculate summary statistics
    total_questions = len(results)
    correct_answers = sum(1 for r in results if r['correct'])
    overall_accuracy = correct_answers / total_questions if total_questions > 0 else 0
    
    by_question_type = {}
    for qtype, stats in stats_by_type.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        by_question_type[qtype] = {
            'accuracy': accuracy,
            'count': stats['total']
        }
    
    summary = {
        'overall_accuracy': overall_accuracy,
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'by_question_type': by_question_type,
        'model': model_name,
        'dataset_dir': dataset_dir,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    # Save summary
    summary_path = output_path.replace('.csv', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Summary saved to {summary_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_answers}")
    print(f"\nAccuracy by Question Type:")
    for qtype, stats in sorted(by_question_type.items()):
        print(f"  {qtype:15s}: {stats['accuracy']:.2%} ({stats['count']} questions)")
    print(f"{'='*60}\n")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate VLM models on synthetic dataset')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model', type=str, default='blip', 
                       choices=['blip', 'blip2'],
                       help='Model to evaluate')
    parser.add_argument('--output', type=str, default='evaluation_results.csv',
                       help='Output CSV file path')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    evaluate_model(
        dataset_dir=args.dataset_dir,
        model_name=args.model,
        output_path=args.output,
        num_samples=args.num_samples,
        device=args.device
    )


if __name__ == "__main__":
    main()