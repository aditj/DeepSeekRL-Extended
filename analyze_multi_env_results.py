"""
Script to analyze multi-environment evaluation results from training.

Usage:
    # Standard analysis (single directory)
    python analyze_multi_env_results.py --output_dir path/to/output --rounds 0 100 200 300
    
    # Generate summary PDFs from multiple subdirectories
    python analyze_multi_env_results.py --output_dir path/to/parent_dir --generate_pdfs
"""

import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


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


def extract_env_name_from_dir(dirname):
    """Extract environment name from directory name like 'grpo_acre_single_...'"""
    parts = dirname.split('_')
    # Find 'grpo' and get the next part(s) until 'single' or 'multi'
    try:
        grpo_idx = parts.index('grpo')
        env_parts = []
        for i in range(grpo_idx + 1, len(parts)):
            if parts[i] in ['single', 'multi']:
                break
            env_parts.append(parts[i])
        return '_'.join(env_parts) if env_parts else dirname
    except (ValueError, IndexError):
        return dirname


def generate_reward_trends_pdf(output_dir, pdf_path):
    """Generate PDF with reward trends for each subdirectory."""
    subdirs = [d for d in os.listdir(output_dir) 
               if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('grpo_')]
    
    if not subdirs:
        print("No subdirectories found starting with 'grpo_'")
        return
    
    with PdfPages(pdf_path) as pdf:
        for subdir in sorted(subdirs):
            subdir_path = os.path.join(output_dir, subdir)
            env_name = extract_env_name_from_dir(subdir)
            
            # Collect metrics for this subdirectory
            df = collect_metrics_across_rounds(subdir_path, rounds=None)
            
            if df is None or df.empty:
                print(f"No data found for {subdir}, skipping...")
                continue
            
            # Create reward trend plot
            fig = plt.figure(figsize=(12, 8))
            
            for env in df['environment'].unique():
                env_data = df[df['environment'] == env]
                plt.plot(env_data['round'], env_data['reward'], 
                        marker='s', label=env, linewidth=2, markersize=6)
            
            plt.xlabel('Training Round', fontsize=14)
            plt.ylabel('Average Reward', fontsize=14)
            plt.title(f'Reward Trends - Trained on: {env_name}', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print(f"Added page for {env_name}")
    
    print(f"\nSaved reward trends PDF to {pdf_path}")


def generate_accuracy_trends_pdf(output_dir, pdf_path):
    """Generate PDF with accuracy trends for each subdirectory."""
    subdirs = [d for d in os.listdir(output_dir) 
               if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('grpo_')]
    
    if not subdirs:
        print("No subdirectories found starting with 'grpo_'")
        return
    
    with PdfPages(pdf_path) as pdf:
        for subdir in sorted(subdirs):
            subdir_path = os.path.join(output_dir, subdir)
            env_name = extract_env_name_from_dir(subdir)
            
            # Collect metrics for this subdirectory
            df = collect_metrics_across_rounds(subdir_path, rounds=None)
            
            if df is None or df.empty:
                print(f"No data found for {subdir}, skipping...")
                continue
            
            # Create accuracy trend plot
            fig = plt.figure(figsize=(12, 8))
            
            for env in df['environment'].unique():
                env_data = df[df['environment'] == env]
                plt.plot(env_data['round'], env_data['accuracy'], 
                        marker='o', label=env, linewidth=2, markersize=6)
            
            plt.xlabel('Training Round', fontsize=14)
            plt.ylabel('Accuracy (%)', fontsize=14)
            plt.title(f'Accuracy Trends - Trained on: {env_name}', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print(f"Added page for {env_name}")
    
    print(f"\nSaved accuracy trends PDF to {pdf_path}")


def generate_reward_components_pdf(output_dir, pdf_path):
    """Generate PDF with detailed reward component trends for each subdirectory."""
    subdirs = [d for d in os.listdir(output_dir) 
               if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('grpo_')]
    
    if not subdirs:
        print("No subdirectories found starting with 'grpo_'")
        return
    
    # Define reward components to plot
    reward_components = [
        ('rewards/correctness_reward_func', 'Correctness Reward', 'o'),
        ('rewards/format_reward_func', 'Format Reward', 's'),
        ('rewards/word_count_reward_func', 'Word Count Reward', '^'),
        ('rewards/xml_count_reward_func', 'XML Count Reward', 'v')
    ]
    
    with PdfPages(pdf_path) as pdf:
        for subdir in sorted(subdirs):
            subdir_path = os.path.join(output_dir, subdir)
            env_name = extract_env_name_from_dir(subdir)
            
            # Collect metrics for this subdirectory
            df = collect_metrics_across_rounds(subdir_path, rounds=None)
            
            if df is None or df.empty:
                print(f"No data found for {subdir}, skipping...")
                continue
            
            # Create a 2x2 subplot for the four reward components + accuracy
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))
            fig.suptitle(f'Reward Components Analysis - Trained on: {env_name}', 
                        fontsize=16, fontweight='bold', y=0.995)
            
            # Plot each reward component
            for idx, (component_key, component_name, marker) in enumerate(reward_components):
                row = idx // 2
                col = idx % 2
                ax = axes[row, col]
                
                for env in df['environment'].unique():
                    env_data = df[df['environment'] == env]
                    if component_key in env_data.columns:
                        ax.plot(env_data['round'], env_data[component_key], 
                               marker=marker, label=env, linewidth=2, markersize=5)
                
                ax.set_xlabel('Training Round', fontsize=11)
                ax.set_ylabel(component_name, fontsize=11)
                ax.set_title(component_name, fontsize=12, fontweight='bold')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Plot total reward in bottom left
            ax = axes[2, 0]
            for env in df['environment'].unique():
                env_data = df[df['environment'] == env]
                ax.plot(env_data['round'], env_data['reward'], 
                       marker='*', label=env, linewidth=2, markersize=6)
            ax.set_xlabel('Training Round', fontsize=11)
            ax.set_ylabel('Total Reward', fontsize=11)
            ax.set_title('Total Reward', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Plot accuracy in bottom right
            ax = axes[2, 1]
            for env in df['environment'].unique():
                env_data = df[df['environment'] == env]
                ax.plot(env_data['round'], env_data['accuracy'], 
                       marker='o', label=env, linewidth=2, markersize=5)
            ax.set_xlabel('Training Round', fontsize=11)
            ax.set_ylabel('Accuracy (%)', fontsize=11)
            ax.set_title('Accuracy', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print(f"Added reward components page for {env_name}")
    
    print(f"\nSaved reward components PDF to {pdf_path}")


def generate_transfer_efficiency_matrix_pdf(output_dir, pdf_path):
    """Generate PDF with transfer efficiency matrices."""
    # Get all subdirectories
    subdirs = [d for d in os.listdir(output_dir) 
               if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('grpo_')]
    
    if not subdirs:
        print("No subdirectories found starting with 'grpo_'")
        return
    
    # Collect all data
    all_data = {}
    all_rounds = set()
    
    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        env_name = extract_env_name_from_dir(subdir)
        df = collect_metrics_across_rounds(subdir_path, rounds=None)
        
        if df is None or df.empty:
            continue
        
        all_data[env_name] = df
        all_rounds.update(df['round'].unique())
    
    if not all_data:
        print("No data found in any subdirectory")
        return
    
    all_rounds = sorted(list(all_rounds))
    trained_envs = sorted(all_data.keys())
    
    # Calculate baseline performance (when trained on own environment)
    baseline_performance = {}
    for trained_env, df in all_data.items():
        # Get performance on own environment across rounds
        own_env_data = df[df['environment'].str.contains(trained_env, case=False, na=False)]
        if not own_env_data.empty:
            baseline_performance[trained_env] = own_env_data.groupby('round')['accuracy'].mean().to_dict()
    
    with PdfPages(pdf_path) as pdf:
        # Create averaged transfer efficiency matrix
        print("  Creating averaged transfer efficiency matrix...")
        evaluated_envs = set()
        for trained_env, df in all_data.items():
            evaluated_envs.update(df['environment'].unique())
        evaluated_envs = sorted(list(evaluated_envs))
        
        # Calculate average efficiency across all rounds
        avg_efficiency = np.full((len(trained_envs), len(evaluated_envs)), np.nan)
        
        for i, trained_env in enumerate(trained_envs):
            df = all_data[trained_env]
            
            for j, eval_env in enumerate(evaluated_envs):
                eval_data = df[df['environment'] == eval_env]
                if not eval_data.empty:
                    # Get baseline (performance when trained on eval_env)
                    baseline_key = None
                    for key in baseline_performance.keys():
                        if key in eval_env or eval_env in key:
                            baseline_key = key
                            break
                    
                    if baseline_key and baseline_performance[baseline_key]:
                        # Calculate efficiency for each round and average
                        efficiencies = []
                        for _, row in eval_data.iterrows():
                            round_num = row['round']
                            if round_num in baseline_performance[baseline_key]:
                                baseline = baseline_performance[baseline_key][round_num]
                                if baseline > 0:
                                    efficiency = row['accuracy'] / baseline
                                    efficiencies.append(efficiency)
                        
                        if efficiencies:
                            avg_efficiency[i, j] = np.mean(efficiencies)
        
        # Plot averaged transfer efficiency matrix
        fig = plt.figure(figsize=(max(12, len(evaluated_envs) * 0.8), 
                                 max(8, len(trained_envs) * 0.6)))
        
        masked_matrix = np.ma.masked_invalid(avg_efficiency)
        im = plt.imshow(masked_matrix, aspect='auto', cmap='RdYlGn', 
                      interpolation='nearest', vmin=0, vmax=1.5)
        
        cbar = plt.colorbar(im, label='Transfer Efficiency (1.0 = same as trained on target)')
        
        plt.yticks(range(len(trained_envs)), trained_envs, fontsize=10)
        plt.xticks(range(len(evaluated_envs)), evaluated_envs, 
                  rotation=45, ha='right', fontsize=10)
        
        # Add text annotations
        for i in range(len(trained_envs)):
            for j in range(len(evaluated_envs)):
                if not np.isnan(avg_efficiency[i, j]):
                    value = avg_efficiency[i, j]
                    text_color = 'white' if value < 0.75 else 'black'
                    plt.text(j, i, f'{value:.2f}', 
                           ha='center', va='center', 
                           color=text_color, fontsize=8, fontweight='bold')
        
        plt.xlabel('Evaluated On', fontsize=12, fontweight='bold')
        plt.ylabel('Trained On', fontsize=12, fontweight='bold')
        plt.title('Transfer Efficiency Matrix (Averaged Across All Rounds)', 
                 fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Create per-round transfer efficiency matrices
        for round_num in all_rounds:
            print(f"  Creating transfer efficiency matrix for round {round_num}...")
            
            # Calculate efficiency for this round
            efficiency = np.full((len(trained_envs), len(evaluated_envs)), np.nan)
            
            for i, trained_env in enumerate(trained_envs):
                df = all_data[trained_env]
                round_data = df[df['round'] == round_num]
                
                for j, eval_env in enumerate(evaluated_envs):
                    eval_data = round_data[round_data['environment'] == eval_env]
                    if not eval_data.empty:
                        # Get baseline (performance when trained on eval_env)
                        baseline_key = None
                        for key in baseline_performance.keys():
                            if key in eval_env or eval_env in key:
                                baseline_key = key
                                break
                        
                        if baseline_key and round_num in baseline_performance[baseline_key]:
                            baseline = baseline_performance[baseline_key][round_num]
                            if baseline > 0:
                                efficiency[i, j] = eval_data['accuracy'].values[0] / baseline
            
            # Plot
            fig = plt.figure(figsize=(max(12, len(evaluated_envs) * 0.8), 
                                     max(8, len(trained_envs) * 0.6)))
            
            masked_matrix = np.ma.masked_invalid(efficiency)
            im = plt.imshow(masked_matrix, aspect='auto', cmap='RdYlGn', 
                          interpolation='nearest', vmin=0, vmax=1.5)
            
            cbar = plt.colorbar(im, label='Transfer Efficiency')
            
            plt.yticks(range(len(trained_envs)), trained_envs, fontsize=10)
            plt.xticks(range(len(evaluated_envs)), evaluated_envs, 
                      rotation=45, ha='right', fontsize=10)
            
            # Add text annotations
            for i in range(len(trained_envs)):
                for j in range(len(evaluated_envs)):
                    if not np.isnan(efficiency[i, j]):
                        value = efficiency[i, j]
                        text_color = 'white' if value < 0.75 else 'black'
                        plt.text(j, i, f'{value:.2f}', 
                               ha='center', va='center', 
                               color=text_color, fontsize=8, fontweight='bold')
            
            plt.xlabel('Evaluated On', fontsize=12, fontweight='bold')
            plt.ylabel('Trained On', fontsize=12, fontweight='bold')
            plt.title(f'Transfer Efficiency Matrix - Round {round_num}', 
                     fontsize=14, fontweight='bold', pad=15)
            plt.tight_layout()
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    print(f"\nSaved transfer efficiency PDF to {pdf_path}")


def generate_confusion_matrix_pdf(output_dir, pdf_path):
    """Generate PDF with confusion matrices (trained vs evaluated) for each round."""
    # Get all subdirectories
    subdirs = [d for d in os.listdir(output_dir) 
               if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('grpo_')]
    
    if not subdirs:
        print("No subdirectories found starting with 'grpo_'")
        return
    
    # Collect all data
    all_data = {}
    all_rounds = set()
    
    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        env_name = extract_env_name_from_dir(subdir)
        df = collect_metrics_across_rounds(subdir_path, rounds=None)
        
        if df is None or df.empty:
            continue
        
        all_data[env_name] = df
        all_rounds.update(df['round'].unique())
    
    if not all_data:
        print("No data found in any subdirectory")
        return
    
    all_rounds = sorted(list(all_rounds))
    
    with PdfPages(pdf_path) as pdf:
        for round_num in all_rounds:
            # Build confusion matrix for this round
            trained_envs = sorted(all_data.keys())
            evaluated_envs = set()
            
            # First pass: collect all evaluated environments
            for trained_env, df in all_data.items():
                round_data = df[df['round'] == round_num]
                evaluated_envs.update(round_data['environment'].unique())
            
            evaluated_envs = sorted(list(evaluated_envs))
            
            # Create matrix
            matrix = np.full((len(trained_envs), len(evaluated_envs)), np.nan)
            
            for i, trained_env in enumerate(trained_envs):
                df = all_data[trained_env]
                round_data = df[df['round'] == round_num]
                
                for j, eval_env in enumerate(evaluated_envs):
                    eval_data = round_data[round_data['environment'] == eval_env]
                    if not eval_data.empty:
                        matrix[i, j] = eval_data['accuracy'].values[0]
            
            # Plot heatmap
            fig = plt.figure(figsize=(max(12, len(evaluated_envs) * 0.8), 
                                     max(8, len(trained_envs) * 0.6)))
            
            # Create masked array to handle NaN values
            masked_matrix = np.ma.masked_invalid(matrix)
            
            im = plt.imshow(masked_matrix, aspect='auto', cmap='RdYlGn', 
                          interpolation='nearest', vmin=0, vmax=1)
            
            # Add colorbar
            cbar = plt.colorbar(im, label='Accuracy (%)')
            
            # Set ticks and labels
            plt.yticks(range(len(trained_envs)), trained_envs, fontsize=10)
            plt.xticks(range(len(evaluated_envs)), evaluated_envs, 
                      rotation=45, ha='right', fontsize=10)
            
            # Add text annotations
            for i in range(len(trained_envs)):
                for j in range(len(evaluated_envs)):
                    if not np.isnan(matrix[i, j]):
                        text_color = 'white' if matrix[i, j] < 50 else 'black'
                        plt.text(j, i, f'{matrix[i, j]:.1f}', 
                               ha='center', va='center', 
                               color=text_color, fontsize=8, fontweight='bold')
            
            plt.xlabel('Evaluated On', fontsize=12, fontweight='bold')
            plt.ylabel('Trained On', fontsize=12, fontweight='bold')
            plt.title(f'Environment Confusion Matrix - Training Round {round_num}', 
                     fontsize=14, fontweight='bold', pad=15)
            plt.tight_layout()
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print(f"Added confusion matrix for round {round_num}")
    
    print(f"\nSaved confusion matrix PDF to {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze multi-environment evaluation results')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to training output directory')
    parser.add_argument('--rounds', type=int, nargs='+', default=None, help='Specific rounds to analyze (default: all)')
    parser.add_argument('--plot_dir', type=str, default=None, help='Directory to save plots (default: output_dir/analysis_plots)')
    parser.add_argument('--no_plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--generate_pdfs', action='store_true', help='Generate comprehensive summary PDFs (reward/accuracy trends, reward components breakdown, confusion matrices, and transfer efficiency) and exit')
    
    args = parser.parse_args()
    
    # Set up plot directory
    if args.plot_dir is None:
        args.plot_dir = os.path.join(args.output_dir, 'analysis_plots')
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Handle PDF generation mode
    if args.generate_pdfs:
        print("Generating summary PDFs...")
        print("="*80)
        
        # Generate reward trends PDF
        reward_pdf = os.path.join(args.plot_dir, 'reward_trends_summary.pdf')
        print(f"\n1. Generating reward trends PDF...")
        generate_reward_trends_pdf(args.output_dir, reward_pdf)
        
        # Generate accuracy trends PDF
        accuracy_pdf = os.path.join(args.plot_dir, 'accuracy_trends_summary.pdf')
        print(f"\n2. Generating accuracy trends PDF...")
        generate_accuracy_trends_pdf(args.output_dir, accuracy_pdf)
        
        # Generate reward components PDF
        components_pdf = os.path.join(args.plot_dir, 'reward_components_summary.pdf')
        print(f"\n3. Generating reward components breakdown PDF...")
        generate_reward_components_pdf(args.output_dir, components_pdf)
        
        # Generate confusion matrix PDF
        confusion_pdf = os.path.join(args.plot_dir, 'confusion_matrix_summary.pdf')
        print(f"\n4. Generating confusion matrix PDF...")
        generate_confusion_matrix_pdf(args.output_dir, confusion_pdf)
        
        # Generate transfer efficiency matrix PDF
        transfer_pdf = os.path.join(args.plot_dir, 'transfer_efficiency_summary.pdf')
        print(f"\n5. Generating transfer efficiency matrix PDF...")
        generate_transfer_efficiency_matrix_pdf(args.output_dir, transfer_pdf)
        
        print("\n" + "="*80)
        print("PDF generation complete!")
        print(f"  - Reward trends: {reward_pdf}")
        print(f"  - Accuracy trends: {accuracy_pdf}")
        print(f"  - Reward components breakdown: {components_pdf}")
        print(f"  - Confusion matrices: {confusion_pdf}")
        print(f"  - Transfer efficiency: {transfer_pdf}")
        return
    
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





