"""
Given a model (opened-source or closed-source), a config file for what to evaluate on (dataset name, what parameters to vary etc.), evaluate the model, and save the results
"""
import argparse
import yaml
import torch
import random
import numpy as np
import os
import json
import statistics
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from typing import Dict, List
from itertools import product
from collections import defaultdict
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Repo specific imports
import llms
import evaluator
import rldatasets
from main import generate_completions

# Set seed
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def load_model(model_name: str, device: str):
    model, tokenizer = llms.get_llm_tokenizer(model_name, device)
    return model, tokenizer

def generate_hyperparam_sets(dataset_config: Dict) -> List[Dict]:
    """
    Args:
        dataset_config: Dict, each key is the name of the hyperparam, each value is a list of [start, end, step]

    Returns:
    List of dicts, each dict is a set of hyperparmeters to run (hyperparam_name: value)
        For parameters in pair (min, max), we will make sure that the max values are always greater than the min values

    Examples input:
    {
        "min_family_size": [2, 12, 2],
        "max_family_size": [6, 16, 2]  
    }
    Example output:
    [
        {"min_family_size": 2, "max_family_size": 6},
        ...,
        {"min_family_size": 2, "max_family_size": 14},
        {"min_family_size": 4, "max_family_size": 6},
        ...,
        {"min_family_size": 4, "max_family_size": 14},
        {"min_family_size": 6, "max_family_size": 8},
        ...,
        {"min_family_size": 6, "max_family_size": 14},
        ...
    ]
    """
    # Extract parameter ranges from config
    param_ranges = {}
    for param_name, param_config in dataset_config.items():
        if param_name != "name":
            start, end, step = param_config
            param_ranges[param_name] = list(np.arange(start, end, step))
    
    # Generate all combinations of parameters
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    # Get all combinations using itertools.product
    all_combinations = list(product(*param_values))
    
    # Convert to list of dictionaries
    hyperparam_sets = []
    for combination in all_combinations:
        param_dict = dict(zip(param_names, combination))
        
        # Filter out invalid combinations where min >= max for paired parameters
        valid = True
        for param_name in param_names:
            if param_name.startswith('min_'):
                max_param_name = param_name.replace('min_', 'max_')
                if max_param_name in param_dict:
                    if dataset_name in ["graph_color"]:
                        # graph_color: min_num_vertices and max_num_vertices can be the same
                        if param_dict[param_name] > param_dict[max_param_name]:
                            valid = False
                            break
                    else:
                        if param_dict[param_name] >= param_dict[max_param_name]:
                            valid = False
                            break
        
        if valid:
            hyperparam_sets.append(param_dict)

    print(hyperparam_sets)
    print(f"Total number of hyperparam sets: {len(hyperparam_sets)}")
    input("Press Enter to continue...")
    
    return hyperparam_sets

def eval_one_dataset(
    model_name: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset_evaluator: evaluator.RewardEvaluator,
    dataset_config: Dict,
    model_generation_config: Dict,
    num_problem_per_hyperparam: int,
    output_dir: str,
    device: str,
):
    """
    Args:
        model_name: Name of the model to evaluate
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        dataset_evaluator: Evaluator to use for the dataset
        dataset_config: Configuration for the dataset
        num_problem_per_hyperparam: Number of problems to evaluate per hyperparam set
        output_dir: Directory to save the results
        device: Device to run evaluation on
        model_generation_config: Configuration for model generation parameters

    Effects:
        For each hyperparam set:
            - Save individual results as a YAML file in the output path (a list of results): 
            - id: match num_examples
              question: 
              response: 
              ground_truth: 
              metrics: 
              total_score: 
              generation_log:
                phase_transitions: []
                final_sequences:
                  sequence_lengths: []
        - Save the aggregated results (average/standard error for each hyperparam set) as a JSON file in the output path
            {
                "hyperparam_set_1": {
                    "metric_1": {
                        "average": value,
                        "standard_error": value,
                    },
                    "metric_2": {
                        "average": value,
                        "standard_error": value,
                    }
                }
                "hyperparam_set_2": {
                    "metric_1": {
                        "average": value,
                        "standard_error": value,
                    },
                    "metric_2": {
                        "average": value,
                        "standard_error": value,
                    }
                }
                ...
            }
    """
    # Determine output path
    dataset_name = dataset_config["name"]
    output_path = os.path.join(output_dir, f"{dataset_name}_s={SEED}", model_name.split("/")[-1])
    os.makedirs(output_path, exist_ok=True)

    # Determine the sets of hyperparmeters to eval on
    hyperparam_sets = generate_hyperparam_sets(dataset_config)

    aggregated_results_file = os.path.join(output_path, "_aggregated_results.json")

    if not os.path.exists(aggregated_results_file):
    
        # Initialize aggregated results collection
        all_aggregated_results = {}

        for i in range(len(hyperparam_sets)):
            hyperparam_set = hyperparam_sets[i]
            hyperparam_set_path = os.path.join(output_path, f"{'+'.join([f'{k}={v}' for k, v in hyperparam_set.items()])}")
            os.makedirs(hyperparam_set_path, exist_ok=True)

            if not os.path.exists(os.path.join(hyperparam_set_path, "individual_results.yaml")):
                _, test_loader = rldatasets.build_reasoning_gym_dataloaders(dataset_name, predefined_test_size=config["num_problem_per_hyperparam"], **hyperparam_set)

                results = []

                for question, answer, entry in tqdm(test_loader, desc=f"Eval set {i+1}/{len(hyperparam_sets)}"):
                    # Create a mock args object with parameters from config file
                    mock_args = argparse.Namespace()
                    for key, value in model_generation_config.items():
                        setattr(mock_args, key, value)
                    
                    # Generate completions
                    if mock_args.normal_generation:
                        prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text = generate_completions(
                            model, tokenizer, question, device, mock_args
                        )
                        generation_log = None
                    else:
                        prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text, generation_log, token_embeddings_list, mixture_selected_tokens = generate_completions(
                            model, tokenizer, question, device, mock_args
                        )
                    
                    # Score completions using evaluator
                    mock_prompts = [[{'content': question}]] * len(completions_text)
                    mock_completions = [[{'content': completion}] for completion in completions_text]
                    # Make answer array same length as completions
                    answers = [answer] * len(completions_text)
                    rewards_per_func, metrics = dataset_evaluator.compute_rewards(
                        prompts=mock_prompts,
                        completions=mock_completions, 
                        answer=answers,
                        device=device,
                        entry=entry
                    )
                    
                    # Create result entry
                    result_entry = {
                        "id": len(results),
                        "question": question,
                        "response": completions_text[0] if completions_text else "",
                        "ground_truth": answer,
                        "metrics": metrics,
                        "total_score": rewards_per_func.sum().item() if rewards_per_func is not None else 0.0,
                        "generation_log": {
                            "phase_transitions": generation_log.get('phase_transitions', []) if generation_log else [],
                            "final_sequences": {
                                "sequence_lengths": generation_log.get('final_sequences', {}).get('sequence_lengths', []) if generation_log else []
                            }
                        }
                    }
                    
                    results.append(result_entry)
                    
                    # Limit number of problems per hyperparam set
                    if len(results) >= num_problem_per_hyperparam:
                        break

                # Save individual results as YAML file
                individual_results_file = os.path.join(hyperparam_set_path, "individual_results.yaml")
                with open(individual_results_file, 'w') as f:
                    yaml.dump(results, f, default_flow_style=False, sort_keys=False)
            else:
                print(f"Individual results already exist for hyperparam set: {hyperparam_set_path}")
                print(f"Loading individual results from: {os.path.join(hyperparam_set_path, 'individual_results.yaml')}")
                with open(os.path.join(hyperparam_set_path, 'individual_results.yaml'), 'r') as f:
                    results = yaml.load(f, Loader=yaml.FullLoader)
                
            # Calculate aggregated results for this hyperparameter set
            hyperparam_key = '+'.join([f'{k}={v}' for k, v in hyperparam_set.items()])
            
            # Collect all metrics from results
            all_metrics = defaultdict(list)
            all_total_scores = []
            
            for result in results:
                all_total_scores.append(result["total_score"])
                for metric_name, metric_value in result["metrics"].items():
                    if metric_name != "correctness":  # Skip correctness as it's not a numeric metric
                        all_metrics[metric_name].append(metric_value)
            
            # Calculate averages and standard errors for each metric
            hyperparam_results = {}
            
            # Add total_score metrics
            if all_total_scores:
                hyperparam_results["total_score"] = {
                    "average": statistics.mean(all_total_scores),
                    "standard_error": statistics.stdev(all_total_scores) / (len(all_total_scores) ** 0.5) if len(all_total_scores) > 1 else 0.0
                }
            
            # Add other metrics
            for metric_name, values in all_metrics.items():
                if values:
                    hyperparam_results[metric_name] = {
                        "average": statistics.mean(values),
                        "standard_error": statistics.stdev(values) / (len(values) ** 0.5) if len(values) > 1 else 0.0
                    }
            
            # Store results for later aggregation
            all_aggregated_results[hyperparam_key] = hyperparam_results
            
            print(f"Completed evaluation for hyperparam set: {hyperparam_key}")
            print(f"Processed {len(results)} problems")
            print(f"Results saved to: {hyperparam_set_path}")
        
        # Save aggregated results for all hyperparameter sets in the main output path
        with open(aggregated_results_file, 'w') as f:
            json.dump(all_aggregated_results, f, indent=4)
        
        print(f"All aggregated results saved to: {aggregated_results_file}")
    else:
        print(f"Aggregated results already exist for {dataset_name}, loading from: {aggregated_results_file}")
        with open(aggregated_results_file, 'r') as f:
            all_aggregated_results = json.load(f)

    return all_aggregated_results

def plot_aggregated_results(aggregated_results: Dict, img_save_path: str, html_save_path: str = "_output/reasoning_gym/_consolidated_html/"):
    """ Plot parallel coordinates plot of the aggregated results
    Args:
        aggregated_results: Dict, each key is the name of the hyperparam set, each value is a dict of the aggregated results
        img_save_path: Directory path where to save individual PNG files for each metric
        html_save_path: Base path where to save individual HTML files for each metric
    
    Effects:
        - Save individual PNG files for each metric in img_save_path/{metric_name}.png
        - Save separate html files for each metric in html_save_path/{metric_name}/{metric_name}_{dataset_name}_s={SEED}.html
    """
    
    # Parse hyperparameter sets and extract data
    data_rows = []
    
    for hyperparam_key, metrics in aggregated_results.items():
        # Parse hyperparameters from key (format: "param1=value1+param2=value2+...")
        hyperparams = {}
        for param_pair in hyperparam_key.split('+'):
            if '=' in param_pair:
                param_name, param_value = param_pair.split('=', 1)
                try:
                    # Try to convert to float, fallback to string
                    hyperparams[param_name] = float(param_value) if '.' in param_value else int(param_value)
                except ValueError:
                    hyperparams[param_name] = param_value
        
        # Extract metrics (use average values)
        row_data = hyperparams.copy()
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict) and 'average' in metric_data:
                row_data[metric_name] = metric_data['average']
        
        data_rows.append(row_data)
    
    if not data_rows:
        print("No data to plot")
        return
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Identify hyperparameter columns vs metric columns
    hyperparam_cols = []
    metric_cols = []
    
    for col in df.columns:
        if col in ['min_rows', 'max_rows', 'min_cols', 'max_cols', 'min_family_size', 'max_family_size', 
                   'min_num_vertices', 'max_num_vertices', 'min_edges', 'max_edges']:
            hyperparam_cols.append(col)
        else:
            metric_cols.append(col)
    
    if not hyperparam_cols or not metric_cols:
        print(f"Warning: Found {len(hyperparam_cols)} hyperparameter columns and {len(metric_cols)} metric columns")
        return
    
    n_metrics = len(metric_cols)
    if n_metrics == 0:
        print("No metrics found to plot")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(img_save_path, exist_ok=True)
    
    # Create individual HTML and PNG files for each metric
    for metric in metric_cols:
        # Create dimensions for this specific metric: hyperparameters + this metric only
        metric_dimensions = hyperparam_cols + [metric]
        
        # Create parallel coordinates plot using go.Parcoords for more control
        fig_metric = go.Figure()
        
        # Prepare data for parallel coordinates
        parcoords_data = []
        for dim_name in metric_dimensions:
            parcoords_data.append(df[dim_name].values)
        
        # Create custom hover text for each line
        hover_texts = []
        for idx, row in df.iterrows():
            hover_parts = []
            for dim_name in metric_dimensions:
                value = row[dim_name]
                if isinstance(value, float):
                    hover_parts.append(f"<b>{dim_name}:</b> {value:.4f}")
                else:
                    hover_parts.append(f"<b>{dim_name}:</b> {value}")
            hover_texts.append("<br>".join(hover_parts))
        
        # Add parallel coordinates trace
        parcoords_trace = go.Parcoords(
            line=dict(
                color=df[metric].values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=metric)
            ),
            dimensions=list([
                dict(
                    label=dim_name,
                    values=df[dim_name].values,
                    range=[df[dim_name].min(), df[dim_name].max()] if dim_name not in hyperparam_cols else [df[dim_name].max(), df[dim_name].min()]
                ) for dim_name in metric_dimensions
            ]),
            customdata=hover_texts
        )
        
        fig_metric.add_trace(parcoords_trace)
        
        # Update layout for better appearance
        fig_metric.update_layout(
            title=f"Interactive Parallel Coordinates: Hyperparameters → {metric}",
            title_x=0.5,
            height=600,
            font=dict(size=12),
            # Add hover mode to highlight lines
            hovermode='closest',
            # Configure hover behavior
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="black",
                font_size=12,
                font_family="Arial"
            ),
            # Add some padding and styling
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Try to make lines more visible by updating the line properties
        fig_metric.update_traces(
            line=dict(
                color=df[metric].values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=metric)
            )
        )
        
        # Create metric-specific directory for HTML
        metric_dir = os.path.join(html_save_path, metric)
        os.makedirs(metric_dir, exist_ok=True)
        
        # Extract dataset name from img_save_path if possible
        dataset_name = "unknown"
        if "reasoning_gym" in img_save_path:
            path_parts = img_save_path.split("/")
            for part in path_parts:
                if "_s=" in part:
                    dataset_name = part.split("_s=")[0]
                    break
        
        # Create HTML filename and save
        html_filename = f"{dataset_name}_{'-'.join(metric.split('/'))}_s={SEED}.html"
        html_filepath = os.path.join(metric_dir, html_filename)
        
        # Generate HTML with custom CSS for thicker lines
        html_content = fig_metric.to_html(include_plotlyjs=True, full_html=True)
        
        # Add custom CSS to make lines thicker
        custom_css = """
        <style>
        .parcoords .line {
            stroke-width: 3px !important;
        }
        .parcoords .line:hover {
            stroke-width: 5px !important;
        }
        .parcoords .line.selected {
            stroke-width: 4px !important;
        }
        </style>
        """
        
        # Insert CSS before closing head tag
        if '</head>' in html_content:
            html_content = html_content.replace('</head>', custom_css + '</head>')
        else:
            # If no head tag, add CSS at the beginning of body
            html_content = html_content.replace('<body>', '<body>' + custom_css)
        
        # Write the modified HTML
        with open(html_filepath, 'w') as f:
            f.write(html_content)
        
        print(f"Interactive plot for {metric} saved to: {html_filepath}")
        
        # Create PNG filename and save
        png_filename = f"{'-'.join(metric.split('/'))}.png"
        png_filepath = os.path.join(img_save_path, png_filename)
        fig_metric.write_image(png_filepath, width=1200, height=600)
        print(f"PNG plot for {metric} saved to: {png_filepath}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/eval_varying_hyperparam.yaml", help="Path to the config file. Contain (1) # of questions to eval per settting (2) General model generation config (3) List of datasets to eval on")
    parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Path to the model to evaluate on")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="Directory to save the evaluation results")
    parser.add_argument("-p", "--plot_eval", action="store_true", default=False, help="Whether to plot the evaluation results")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Detect the device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load agents to evaluate on
    model, tokenizer = load_model(args.model, device)

    output_dir = os.path.join(args.output_dir, "reasoning_gym")

    for dataset_config in config["datasets"]:
        # Load evaluator
        dataset_evaluator = evaluator.get_evaluator(f"{dataset_config['name']}.reasoning_gym")  # Need to add .reasoning_gym to be compatible with the evaluator

        # Extract the dataset name and merge params into the config
        dataset_name = dataset_config["name"]
        dataset_params = dataset_config.get("params", {})
        full_dataset_config = {"name": dataset_name, **dataset_params}
        
        # Evaluate model on this dataset configuration
        all_aggregated_results = eval_one_dataset(
            model_name=args.model,
            model=model, 
            tokenizer=tokenizer,
            dataset_evaluator=dataset_evaluator,
            dataset_config=full_dataset_config,
            model_generation_config=config["model_generation_config"],
            num_problem_per_hyperparam=config["num_problem_per_hyperparam"],
            output_dir=output_dir,
            device=device
        )
        
        if args.plot_eval:
            # Generate interactive parallel coordinates plot
            img_save_path = os.path.join(output_dir, f"{dataset_name}_s={SEED}", args.model.split("/")[-1])
            html_save_path = os.path.join(output_dir, "_consolidated_html")
            plot_aggregated_results(all_aggregated_results, img_save_path, html_save_path)



    

