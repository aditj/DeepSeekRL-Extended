"""
Script to analyze multi-environment evaluation results from training.

Usage:
    python analyze_multi_env_results.py --output_dir path/to/output --rounds 0 100 200 300
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_combined_metrics(output_dir, round_num):
    """Load combined metrics for all environments at a given round."""
    metrics_file = os.path.join(output_dir, 'eval_logs', f'metrics_{round_num}_all_envs.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None


def collect_metrics_across_rounds(output_dir, rounds=None):
    """
    Collect metrics across all evaluation rounds.
    
    Args:
        output_dir: Path to output directory
        rounds: List of specific rounds to analyze, or None for all
        
    Returns:
        DataFrame with columns: round, environment, accuracy, reward, ...
    """
    eval_log_dir = os.path.join(output_dir, 'eval_logs')
    
    if not os.path.exists(eval_log_dir):
        print(f"Error: {eval_log_dir} does not exist")
        return None
    
    # Find all combined metrics files if rounds not specified
    if rounds is None:
        rounds = []
        for file in os.listdir(eval_log_dir):
            if file.endswith('_all_envs.json'):
                round_num = int(file.split('_')[1])
                rounds.append(round_num)
        rounds = sorted(rounds)
    
    data = []
    for round_num in rounds:
        combined_metrics = load_combined_metrics(output_dir, round_num)
        if combined_metrics:
            for env_name, env_data in combined_metrics.items():
                row = {
                    'round': round_num,
                    'environment': env_name,
                    'accuracy': env_data.get('accuracy', 0.0),
                    'reward': env_data['metrics'].get('reward', 0.0)
                }
                # Add individual reward components
                for metric_name, metric_value in env_data['metrics'].items():
                    if metric_name != 'reward':
                        row[metric_name] = metric_value
                data.append(row)
    
    if not data:
        print("No metrics data found")
        return None
    
    df = pd.DataFrame(data)
    return df


def plot_accuracy_trends(df, output_path=None):
    """Plot accuracy trends across rounds for each environment."""
    plt.figure(figsize=(12, 6))
    
    for env in df['environment'].unique():
        env_data = df[df['environment'] == env]
        plt.plot(env_data['round'], env_data['accuracy'], marker='o', label=env, linewidth=2)
    
    plt.xlabel('Training Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Evaluation Accuracy Across Environments', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved accuracy plot to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_reward_trends(df, output_path=None):
    """Plot reward trends across rounds for each environment."""
    plt.figure(figsize=(12, 6))
    
    for env in df['environment'].unique():
        env_data = df[df['environment'] == env]
        plt.plot(env_data['round'], env_data['reward'], marker='s', label=env, linewidth=2)
    
    plt.xlabel('Training Round', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Reward Trends Across Environments', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved reward plot to {output_path}")
    else:
        plt.show()
    plt.close()


def create_summary_table(df):
    """Create a summary table showing latest metrics for each environment."""
    latest_round = df['round'].max()
    latest_data = df[df['round'] == latest_round]
    
    print(f"\n{'='*80}")
    print(f"Summary at Round {latest_round}")
    print(f"{'='*80}")
    print(f"{'Environment':<30} {'Accuracy':<15} {'Reward':<15}")
    print(f"{'-'*80}")
    
    for _, row in latest_data.iterrows():
        print(f"{row['environment']:<30} {row['accuracy']:>6.2f}%        {row['reward']:>6.4f}")
    
    print(f"{'='*80}\n")


def create_comparison_heatmap(df, output_path=None):
    """Create a heatmap comparing all environments across rounds."""
    pivot_accuracy = df.pivot(index='environment', columns='round', values='accuracy')
    
    plt.figure(figsize=(14, max(6, len(pivot_accuracy) * 0.5)))
    plt.imshow(pivot_accuracy.values, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    plt.colorbar(label='Accuracy (%)')
    plt.yticks(range(len(pivot_accuracy.index)), pivot_accuracy.index)
    plt.xticks(range(len(pivot_accuracy.columns)), pivot_accuracy.columns, rotation=45)
    plt.xlabel('Training Round')
    plt.ylabel('Environment')
    plt.title('Accuracy Heatmap Across Environments and Rounds', fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {output_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze multi-environment evaluation results')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to training output directory')
    parser.add_argument('--rounds', type=int, nargs='+', default=None, help='Specific rounds to analyze (default: all)')
    parser.add_argument('--plot_dir', type=str, default=None, help='Directory to save plots (default: output_dir/analysis_plots)')
    parser.add_argument('--no_plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Set up plot directory
    if args.plot_dir is None:
        args.plot_dir = os.path.join(args.output_dir, 'analysis_plots')
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Load and process data
    print(f"Loading metrics from {args.output_dir}...")
    df = collect_metrics_across_rounds(args.output_dir, args.rounds)
    
    if df is None or df.empty:
        print("No data to analyze. Make sure evaluation has run and produced metrics files.")
        return
    
    print(f"Loaded {len(df)} data points across {df['environment'].nunique()} environments and {df['round'].nunique()} rounds")
    
    # Create summary table
    create_summary_table(df)
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        plot_accuracy_trends(df, os.path.join(args.plot_dir, 'accuracy_trends.png'))
        plot_reward_trends(df, os.path.join(args.plot_dir, 'reward_trends.png'))
        create_comparison_heatmap(df, os.path.join(args.plot_dir, 'accuracy_heatmap.png'))
        print(f"\nAll plots saved to {args.plot_dir}")
    
    # Save processed data
    csv_path = os.path.join(args.plot_dir, 'combined_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved combined metrics to {csv_path}")


if __name__ == '__main__':
    main()
