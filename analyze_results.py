import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import json


def analyze_results(results_path: str, output_dir: str = './analysis'):
    """Analyze evaluation results and generate visualizations"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load results
    print(f"Loading results from {results_path}...")
    df = pd.read_csv(results_path)
    
    print(f"Loaded {len(df)} results")
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Accuracy by question type
    print("\n1. Generating accuracy by question type chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    type_accuracy = df.groupby('question_type')['correct'].agg(['mean', 'count'])
    type_accuracy = type_accuracy.sort_values('mean', ascending=False)
    
    bars = ax.bar(range(len(type_accuracy)), type_accuracy['mean'])
    ax.set_xticks(range(len(type_accuracy)))
    ax.set_xticklabels(type_accuracy.index, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Question Type')
    ax.set_title('Accuracy by Question Type')
    ax.set_ylim([0, 1])
    
    # Add count labels on bars
    for i, (idx, row) in enumerate(type_accuracy.iterrows()):
        ax.text(i, row['mean'] + 0.02, f"n={row['count']}", 
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_by_type.png", dpi=300)
    print(f"  ✓ Saved to {output_dir}/accuracy_by_type.png")
    
    # 2. Similarity score distribution
    print("\n2. Generating similarity score distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(df['similarity_score'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(0.8, color='red', linestyle='--', label='Threshold (0.8)')
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Similarity Scores')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/similarity_distribution.png", dpi=300)
    print(f"  ✓ Saved to {output_dir}/similarity_distribution.png")
    
    # 3. Examples of correct vs incorrect
    print("\n3. Generating example predictions...")
    
    correct_examples = df[df['correct'] == True].head(5)
    incorrect_examples = df[df['correct'] == False].head(5)
    
    with open(f"{output_dir}/examples.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("CORRECT PREDICTIONS (Sample)\n")
        f.write("="*80 + "\n\n")
        
        for _, row in correct_examples.iterrows():
            f.write(f"Image: {row['image_filename']}\n")
            f.write(f"Question: {row['question']}\n")
            f.write(f"Expected: {row['expected_answer']}\n")
            f.write(f"Model: {row['model_answer']}\n")
            f.write(f"Similarity: {row['similarity_score']:.3f}\n")
            f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("INCORRECT PREDICTIONS (Sample)\n")
        f.write("="*80 + "\n\n")
        
        for _, row in incorrect_examples.iterrows():
            f.write(f"Image: {row['image_filename']}\n")
            f.write(f"Question: {row['question']}\n")
            f.write(f"Expected: {row['expected_answer']}\n")
            f.write(f"Model: {row['model_answer']}\n")
            f.write(f"Similarity: {row['similarity_score']:.3f}\n")
            f.write("\n")
    
    print(f"  ✓ Saved to {output_dir}/examples.txt")
    
    # 4. Summary report
    print("\n4. Generating summary report...")
    
    overall_accuracy = df['correct'].mean()
    avg_similarity = df['similarity_score'].mean()
    
    with open(f"{output_dir}/report.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write("EVALUATION ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Overall Accuracy: {overall_accuracy:.2%}\n")
        f.write(f"Average Similarity Score: {avg_similarity:.3f}\n")
        f.write(f"Total Questions: {len(df)}\n")
        f.write(f"Correct Answers: {df['correct'].sum()}\n")
        f.write(f"Incorrect Answers: {(~df['correct']).sum()}\n\n")
        
        f.write("Accuracy by Question Type:\n")
        f.write("-" * 40 + "\n")
        for qtype, group in df.groupby('question_type'):
            acc = group['correct'].mean()
            count = len(group)
            f.write(f"  {qtype:15s}: {acc:.2%} ({count} questions)\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"  ✓ Saved to {output_dir}/report.txt")
    
    print(f"\n✓ Analysis complete! Results saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Analyze VLM evaluation results')
    parser.add_argument('--results', type=str, required=True, 
                       help='Path to evaluation results CSV')
    parser.add_argument('--output-dir', type=str, default='./analysis',
                       help='Output directory for analysis')
    
    args = parser.parse_args()
    
    analyze_results(args.results, args.output_dir)


if __name__ == "__main__":
    main()