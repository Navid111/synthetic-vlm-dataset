import argparse
import pandas as pd
from pathlib import Path
from evaluate_model import evaluate_model


def compare_models(dataset_dir: str, models: list, output_path: str, 
                   num_samples: int = None, device: str = 'auto'):
    """Compare multiple models on the same dataset"""
    
    results = []
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.upper()}")
        print(f"{'='*60}\n")
        
        # Create temporary output file for this model
        temp_output = f"temp_{model_name}_results.csv"
        
        # Evaluate model
        summary = evaluate_model(
            dataset_dir=dataset_dir,
            model_name=model_name,
            output_path=temp_output,
            num_samples=num_samples,
            device=device
        )
        
        # Store summary
        results.append({
            'model': model_name,
            'overall_accuracy': summary['overall_accuracy'],
            'total_questions': summary['total_questions'],
            **{f"{qtype}_accuracy": stats['accuracy'] 
               for qtype, stats in summary['by_question_type'].items()}
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON")
    print(f"{'='*60}\n")
    print(comparison_df.to_string(index=False))
    print(f"\n✓ Comparison saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare multiple VLM models')
    parser.add_argument('--dataset-dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                       choices=['blip', 'blip2'],
                       help='Models to compare')
    parser.add_argument('--output', type=str, default='model_comparison.csv',
                       help='Output comparison CSV file')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    compare_models(
        dataset_dir=args.dataset_dir,
        models=args.models,
        output_path=args.output,
        num_samples=args.num_samples,
        device=args.device
    )


if __name__ == "__main__":
    main()