"""
Script to compare results across multiple experiments.

Usage:
    python compare_experiments.py --experiment_dirs output_single_multi_env output_mixture_multi_env
    python compare_experiments.py --experiment_dirs output_single_multi_env --output comparison_results.csv
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_final_metrics(experiment_dir):
    """Load the final (last) all_envs metrics file from an experiment."""
    eval_log_dir = os.path.join(experiment_dir, 'eval_logs')
    
    if not os.path.exists(eval_log_dir):
        return None
    
    # Find all combined metrics files
    all_env_files = sorted([
        f for f in os.listdir(eval_log_dir) 
        if f.endswith('_all_envs.json')
    ], key=lambda x: int(x.split('_')[1]))
    
    if not all_env_files:
        return None
    
    # Load the last one
    last_file = os.path.join(eval_log_dir, all_env_files[-1])
    with open(last_file, 'r') as f:
        data = json.load(f)
    
    round_num = int(all_env_files[-1].split('_')[1])
    return data, round_num


def extract_experiment_info(dir_name):
    """Extract experiment information from directory name."""
    # Example: grpo_acre_single_1000steps_eval50_chains10_seed420394_temp0.9
    parts = dir_name.replace('grpo_', '').split('_')
    
    info = {
        'full_name': dir_name,
        'task': parts[0] if len(parts) > 0 else 'unknown',
        'method': 'unknown'
    }
    
    # Detect method
    if 'single' in dir_name or 'vanilla' in dir_name:
        info['method'] = 'vanilla'
    elif 'different_tokens' in dir_name or 'different' in dir_name:
        info['method'] = 'different_tokens'
    elif 'dirichlet' in dir_name:
        info['method'] = 'dirichlet'
    
    # Extract other parameters
    for part in parts:
        if 'steps' in part:
            try:
                info['steps'] = int(part.replace('steps', ''))
            except:
                pass
        elif 'chains' in part:
            try:
                info['chains'] = int(part.replace('chains', ''))
            except:
                pass
        elif 'seed' in part:
            try:
                info['seed'] = int(part.replace('seed', ''))
            except:
                pass
        elif part.startswith('k') and part[1:].isdigit():
            info['mixture_k'] = int(part[1:])
    
    return info


def collect_all_experiments(base_dirs):
    """Collect results from all experiment directories."""
    all_results = []
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Warning: Directory {base_dir} does not exist")
            continue
        
        # Find all experiment subdirectories
        for exp_dir_name in os.listdir(base_dir):
            exp_path = os.path.join(base_dir, exp_dir_name)
            if not os.path.isdir(exp_path):
                continue
            
            # Load metrics
            result = load_final_metrics(exp_path)
            if result is None:
                print(f"Warning: No metrics found in {exp_path}")
                continue
            
            metrics_data, final_round = result
            exp_info = extract_experiment_info(exp_dir_name)
            
            # Extract accuracies for each evaluation environment
            for env_name, env_data in metrics_data.items():
                row = {
                    'experiment_dir': exp_path,
                    'experiment_name': exp_dir_name,
                    'training_task': exp_info['task'],
                    'method': exp_info['method'],
                    'final_round': final_round,
                    'eval_environment': env_name.replace('reasoning_gym.', ''),
                    'accuracy': env_data.get('accuracy', 0.0),
                    'reward': env_data['metrics'].get('reward', 0.0)
                }
                
                # Add other info
                for key in ['steps', 'chains', 'seed', 'mixture_k']:
                    if key in exp_info:
                        row[key] = exp_info[key]
                
                all_results.append(row)
    
    return pd.DataFrame(all_results)


def create_comparison_plots(df, output_dir):
    """Create comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Accuracy heatmap (training task vs eval environment)
    if 'training_task' in df.columns and 'eval_environment' in df.columns:
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            pivot = method_df.pivot_table(
                values='accuracy',
                index='training_task',
                columns='eval_environment',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(14, 8))
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100)
            plt.title(f'Accuracy Heatmap - {method.capitalize()} Method\n(Training Task vs Evaluation Environment)', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Evaluation Environment', fontsize=12)
            plt.ylabel('Training Task', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'heatmap_{method}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot 2: Method comparison across environments
    if len(df['method'].unique()) > 1:
        plt.figure(figsize=(14, 6))
        method_perf = df.groupby(['method', 'eval_environment'])['accuracy'].mean().reset_index()
        
        for method in df['method'].unique():
            method_data = method_perf[method_perf['method'] == method]
            plt.plot(method_data['eval_environment'], method_data['accuracy'], 
                    marker='o', linewidth=2, label=method, markersize=8)
        
        plt.xlabel('Evaluation Environment', fontsize=12)
        plt.ylabel('Average Accuracy (%)', fontsize=12)
        plt.title('Method Comparison Across Evaluation Environments', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Transfer learning analysis (train vs eval)
    plt.figure(figsize=(12, 10))
    
    # Calculate performance for same-task and cross-task evaluation
    df['is_same_task'] = df['training_task'] == df['eval_environment']
    same_task = df[df['is_same_task']].groupby('method')['accuracy'].mean()
    cross_task = df[~df['is_same_task']].groupby('method')['accuracy'].mean()
    
    comparison_df = pd.DataFrame({
        'Same Task': same_task,
        'Cross Task': cross_task
    })
    
    ax = comparison_df.plot(kind='bar', figsize=(10, 6), width=0.7)
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Average Accuracy (%)', fontsize=12)
    plt.title('Transfer Learning: Same-Task vs Cross-Task Evaluation', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transfer_learning.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def print_summary(df):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\nTotal experiments: {df['experiment_name'].nunique()}")
    print(f"Methods: {', '.join(df['method'].unique())}")
    print(f"Training tasks: {', '.join(df['training_task'].unique())}")
    print(f"Evaluation environments: {', '.join(df['eval_environment'].unique())}")
    
    print("\n" + "-"*80)
    print("AVERAGE ACCURACY BY METHOD")
    print("-"*80)
    method_summary = df.groupby('method')['accuracy'].agg(['mean', 'std', 'min', 'max'])
    print(method_summary.to_string())
    
    print("\n" + "-"*80)
    print("BEST PERFORMING EXPERIMENTS (Top 10)")
    print("-"*80)
    top_experiments = df.nlargest(10, 'accuracy')[
        ['experiment_name', 'training_task', 'eval_environment', 'method', 'accuracy']
    ]
    print(top_experiments.to_string(index=False))
    
    print("\n" + "-"*80)
    print("AVERAGE ACCURACY: TRAINING TASK vs EVALUATION ENVIRONMENT")
    print("-"*80)
    pivot = df.pivot_table(
        values='accuracy',
        index='training_task',
        columns='eval_environment',
        aggfunc='mean'
    )
    print(pivot.to_string())
    
    # Transfer learning analysis
    df['is_same_task'] = df['training_task'] == df['eval_environment']
    same_task_acc = df[df['is_same_task']]['accuracy'].mean()
    cross_task_acc = df[~df['is_same_task']]['accuracy'].mean()
    
    print("\n" + "-"*80)
    print("TRANSFER LEARNING ANALYSIS")
    print("-"*80)
    print(f"Same-task evaluation accuracy:  {same_task_acc:.2f}%")
    print(f"Cross-task evaluation accuracy: {cross_task_acc:.2f}%")
    print(f"Transfer gap:                   {same_task_acc - cross_task_acc:.2f}%")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Compare results across multiple experiments')
    parser.add_argument('--experiment_dirs', type=str, nargs='+', required=True,
                       help='Directories containing experiment results')
    parser.add_argument('--output', type=str, default='experiment_comparison.csv',
                       help='Output CSV file path')
    parser.add_argument('--plot_dir', type=str, default='comparison_plots',
                       help='Directory to save comparison plots')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    print("Collecting experiment results...")
    df = collect_all_experiments(args.experiment_dirs)
    
    if df.empty:
        print("No experiment results found!")
        return
    
    print(f"Found {len(df)} result entries across {df['experiment_name'].nunique()} experiments")
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    
    # Print summary
    print_summary(df)
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating comparison plots...")
        create_comparison_plots(df, args.plot_dir)


if __name__ == '__main__':
    main()
