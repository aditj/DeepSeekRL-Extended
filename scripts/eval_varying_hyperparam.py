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
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from tqdm import tqdm
from typing import Dict, List, Callable
from itertools import product
from collections import defaultdict
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Repo specific imports
import llms
import evaluator
import rldatasets
from main_single import generate_completions

# Set seed
SEED = 42

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def validate_model_path(model_path: str) -> bool:
    """
    Validate if a model path exists (for local paths) or is accessible (for HuggingFace models).
    
    Args:
        model_path: Path to the model (local or HuggingFace)
    
    Returns:
        True if path is valid, False otherwise
    """
    # Check if it's a local path (contains 'models/' or starts with './' or is an absolute path)
    if (model_path.startswith('models/') or 
        model_path.startswith('./') or 
        model_path.startswith('/') or 
        (os.path.exists(model_path) and os.path.isdir(model_path))):
        # Check if local path exists
        if os.path.exists(model_path):
            return True
        else:
            print(f"Warning: Local model path does not exist: {model_path}")
            return False
    else:
        # Assume it's a HuggingFace model name - we'll let the loading function handle validation
        return True

def load_model(model_name: str, device: str):
    """
    Load a model and tokenizer from either HuggingFace or local path.
    
    Args:
        model_name: Model name (HuggingFace) or local path
        device: Device to load on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Validate the model path first
    if not validate_model_path(model_name):
        raise ValueError(f"Invalid model path: {model_name}")
    
    try:
        model, tokenizer = llms.get_llm_tokenizer(model_name, device)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise

def generate_hyperparam_sets(dataset_config: Dict, eval_mode: str) -> List[Dict]:
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
        if param_name != "name" and param_name != "train_range":
            start, end, step = param_config
            param_ranges[param_name] = np.arange(start, end, step).tolist()
    
    if eval_mode == "train":
        hyperparam_sets = _generate_hyperparam_sets_train(param_ranges, dataset_config["name"])
    elif eval_mode == "test":
        hyperparam_sets = _generate_hyperparam_sets_test(param_ranges, dataset_config["name"])
    else:
        raise ValueError(f"Invalid eval mode: {eval_mode}")

    print(hyperparam_sets)
    print(f"Total number of hyperparam sets: {len(hyperparam_sets)}")
    # input("Press Enter to continue...")
    
    return hyperparam_sets


def _generate_hyperparam_sets_train(param_ranges: Dict, dataset_name: str) -> List[Dict]:
    """Generate all combinations of parameters for training parameter swipe

    Most importantly: For a hyperparameter pair (min, max), min and max can have different values
    """
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
                    if param_dict[param_name] > param_dict[max_param_name]:
                        valid = False
                        break
        
        if valid:
            hyperparam_sets.append(param_dict)

    return hyperparam_sets

def _generate_hyperparam_sets_test(param_ranges: Dict, dataset_name: str) -> List[Dict]:
    """Generate all combinations of parameters for testing parameter swipe

    Note:
        - Most importantly: min and max MUST have the same value for all hyperparameters
        - Some hyperparameters have multiple ranges (e.g., min_length_part_1 and min_length_part_2): We will consolidate them into a single range
    """
    # First, consolidate parameters with _part_1, _part_2, ..., _part_10 suffixes
    consolidated_ranges = {}
    processed_params = set()
    
    for param_name, param_values in param_ranges.items():

            
        # Check if this is a multi-part parameter
        if '_part_' in param_name:
            # Extract base name and part number
            parts = param_name.split('_part_')
            if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) == 1:
                # Only consolidate the results once (when part_num == 1)
                base_name = parts[0]
                part_num = int(parts[1])
                
                # Collect all parts for this base parameter
                all_parts_values = []
                for i in range(1, 11):  # Support up to 10 parts
                    part_param_name = f"{base_name}_part_{i}"
                    if part_param_name in param_ranges:
                        all_parts_values.extend(param_ranges[part_param_name])
                        processed_params.add(part_param_name)
                    else:
                        break  # No more parts found
                
                # Only consolidate if we found multiple parts
                if len(all_parts_values) > 0:
                    consolidated_ranges[base_name] = all_parts_values
                else:
                    # Single part, use as is
                    consolidated_ranges[param_name] = param_values
        else:
            # Regular parameter, use as is
            consolidated_ranges[param_name] = param_values
    
    # Generate all combinations of consolidated parameters
    param_names = list(consolidated_ranges.keys())
    param_values = list(consolidated_ranges.values())
    
    # Get all combinations using itertools.product
    all_combinations = list(product(*param_values))
    
    # Convert to list of dictionaries
    hyperparam_sets = []
    for combination in all_combinations:
        param_dict = dict(zip(param_names, combination))
        
        # For test mode, we need to ensure min and max parameters have the same values
        # and handle special cases
        processed_dict = {}
        
        for param_name, param_value in param_dict.items():
            processed_dict[param_name] = param_value
            
            if param_name.startswith('min_'):
                max_param_name = param_name.replace('min_', 'max_')

                if max_param_name not in param_dict:
                    # max parameter needs to be set to the same value as min parameter
                    if dataset_name == "number_sequence" and param_name == "min_value":
                        # Special case for number_sequence
                        # because this specifies the range of value to each term, we set max to be abs(min)
                        processed_dict[max_param_name] = abs(param_value)
                    else:
                        # For other datasets, we set max to be the same as min
                        processed_dict[max_param_name] = param_value
        
        # Post processing to check for invalid combinations
        # Filter out invalid combinations where min >= max for paired parameters
        valid = True
        completed_param_names = list(processed_dict.keys())
        for param_name in completed_param_names:
            if param_name.startswith('min_'):
                max_param_name = param_name.replace('min_', 'max_')
                if max_param_name in processed_dict:
                    if processed_dict[param_name] > processed_dict[max_param_name]:
                        print(f"Invalid combination: {param_name} = {processed_dict[param_name]} and {max_param_name} = {processed_dict[max_param_name]}")
                        valid = False

        if dataset_name == "palindrome_partitioning":
            # Requires Maximum substring palindrome length must be less than or equal to maximum string length
            valid = valid and (processed_dict["max_substring_palindrome_len"] <= processed_dict["max_string_len"])
        
        if valid:
            print(processed_dict)
            hyperparam_sets.append(processed_dict)
    
    return hyperparam_sets
    

def eval_one_dataset(
    eval_mode: str,
    model_name_to_save: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset_evaluator: evaluator.RewardEvaluator,
    dataset_config: Dict,
    model_generation_config: Dict,
    num_problem_per_hyperparam: int,
    output_dir: str,
    device: str,
    get_eval_data: bool = False,
    use_aggregated_results: bool = False,
):
    """
    Args:
        eval_mode: Mode of hyperparameter evaluation (train or test)
            Affects how hyperparameters are generated and loaded
        model_name_to_save: Name of the model to save the results to
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
    output_path = os.path.join(output_dir, f"{dataset_name}_s={SEED}", model_name_to_save)
    os.makedirs(output_path, exist_ok=True)

    # Determine the sets of hyperparmeters to eval on
    hyperparam_sets = generate_hyperparam_sets(dataset_config, eval_mode)

    aggregated_results_file = os.path.join(output_path, "_aggregated_results.json")

    if use_aggregated_results and os.path.exists(aggregated_results_file):
        print(f"\033[33mLoading aggregated results from: {aggregated_results_file}\033[0m")
        with open(aggregated_results_file, 'r') as f:
            all_aggregated_results = json.load(f)

        return all_aggregated_results
    else:
        all_aggregated_results = {}

    for i in range(len(hyperparam_sets)):
        hyperparam_set = hyperparam_sets[i]
        print(hyperparam_set)
        hyperparam_set_path = os.path.join(output_path, f"{'+'.join([f'{k}={v}' for k, v in hyperparam_set.items()])}")
        os.makedirs(hyperparam_set_path, exist_ok=True)
        individual_results_file = os.path.join(hyperparam_set_path, "_individual_results.yaml")

        if not os.path.exists(individual_results_file):
            if get_eval_data:
                raise ValueError(f"{individual_results_file} should exists, but it doesn't")

            set_seed(SEED)
            _, test_loader = rldatasets.build_reasoning_gym_dataloaders(dataset_name, predefined_test_size=num_problem_per_hyperparam, seed=SEED, **hyperparam_set)

            results = []

            for question, answer, entry in tqdm(test_loader, desc=f"Eval set {i+1}/{len(hyperparam_sets)}"):
                # Create a mock args object with parameters from config file
                mock_args = argparse.Namespace()
                for key, value in model_generation_config.items():
                    setattr(mock_args, key, value)
                
                # Generate completions
                # Note: (10/23) normal_generation deprecated in main.py
                prompt_completion_ids, prompt_ids, completion_ids, attention_mask, completions_text, prompt_text = generate_completions(
                    model=model, tokenizer=tokenizer, question=question, device=device, args=mock_args
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
                    "total_score": rewards_per_func.sum().item() if rewards_per_func is not None else 0.0
                }
            
                # Save individual question file in txt format (similar to main.py)
                question_id = len(results)
                question_file = os.path.join(hyperparam_set_path, f"{question_id}.md")
                with open(question_file, 'w') as f:
                    f.write(f"# Question id={question_id}\n")
                    f.write(f"## Question:\n{question}\n\n")
                    f.write(f"## Response:\n{completions_text[0] if completions_text else ''}\n\n")
                    f.write(f"## Ground Truth:\n{answer}\n")
                    f.write("## Metrics:\n")
                    for metric, value in metrics.items():
                        f.write(f"- {metric}: {value}\n")
                    f.write(f"Total Score: {rewards_per_func.sum().item() if rewards_per_func is not None else 0.0}\n")

                results.append(result_entry)
                
                # Limit number of problems per hyperparam set
                if len(results) >= num_problem_per_hyperparam:
                    break

            # Save individual results as YAML file
            individual_results_file = os.path.join(hyperparam_set_path, "_individual_results.yaml")
            with open(individual_results_file, 'w') as f:
                yaml.dump(results, f, default_flow_style=False, sort_keys=False)
        else:
            print(f"Loading individual results from: {individual_results_file}")
            with open(individual_results_file, 'r') as f:
                results = yaml.load(f, Loader=yaml.FullLoader)

            # Filter the results to only include the first num_problem_per_hyperparam results
            results = results[:num_problem_per_hyperparam]
                
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

    return all_aggregated_results

def plot_detailed_sweep_results(dataset_config: Dict, aggregated_results: Dict, img_save_path: str, html_save_path: str = "_output/reasoning_gym/_consolidated_html/"):
    """ Plot parallel coordinates plot of the aggregated results
    Args:
        dataset_config: Dict containing dataset configuration including hyperparameter names
        aggregated_results: Dict, each key is the name of the hyperparam set, each value is a dict of the aggregated results
        img_save_path: Directory path where to save individual PNG files for each metric
        html_save_path: Base path where to save individual HTML files for each metric
        dataset_config: Dict containing dataset configuration including hyperparameter names
    
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
    
    # Get hyperparameter names from dataset config if available
    if dataset_config and 'params' in dataset_config:
        expected_hyperparams = set(dataset_config['params'].keys())
    else:
        # Fallback to hardcoded list if config not available
        expected_hyperparams = {'min_rows', 'max_rows', 'min_cols', 'max_cols', 'min_family_size', 'max_family_size', 
                               'min_num_vertices', 'max_num_vertices', 'min_edges', 'max_edges'}
    
    for col in df.columns:
        if col in expected_hyperparams:
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
        
        # Create HTML filename and save
        html_filename = f"{dataset_config['name']}_{'-'.join(metric.split('/'))}_s={SEED}.html"
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

def get_model_name_to_plot(model_name_to_results: Dict) -> Dict:
    """
    Get the model name to plot for the results. We only keep track of the high-level model names.
    """
    model_names_to_plot = {}
    for model_name in model_name_to_results.keys():
        if "-checkpoint" in model_name:
            relevant_model_name = model_name.split("-checkpoint")[0]
        elif "Qwen" in model_name:
            relevant_model_name = "Qwen"
        else:
            relevant_model_name = model_name
        if relevant_model_name not in model_names_to_plot:
            model_names_to_plot[relevant_model_name] = 1
        else:
            model_names_to_plot[relevant_model_name] += 1
    
    return "-".join([f"{model_name}={count}" for model_name, count in sorted(model_names_to_plot.items())])

def get_metric_sorting_function(dataset_name: str) -> Callable:
    """
    Get the sorting function for a dataset based on its primary hyperparameter.
    Returns None if no sorting function is defined for the dataset.

    The function takes the hyperparameter values, returns the value to sort by (a number) and the value to display in the plot (a string).
    """
    if dataset_name == "family_relationships":
        # Sort by min_family_size
        return lambda x: (x["min_family_size"], str(x["min_family_size"])), "Family Size"
    elif dataset_name == "shortest_path":
        # Sort by the area of the grid (min_rows * min_cols)
        return lambda x: (x["min_rows"] * x["min_cols"], f"r={x["min_rows"]}_c={x["min_cols"]}"), "Rows by Columns"
    elif dataset_name == "graph_color":
        # Sort by 2 * min_num_vertices + num_colors
        #   Assume that min_num_vertices make the problem harder, so we give it more weight
        return lambda x: (2 * x["min_num_vertices"] + x["num_colors"], f"v={x["min_num_vertices"]}_c={x["num_colors"]}"), "# Vertices by # Colors"
    elif dataset_name == "number_sequence":
        # Sort by 10 * number of terms + value / 100 + max_complexity
        #   Assume that number of terms makes the problem harder, so we give it more weight
        #   min_value is in the 100s so we rescale it
        return lambda x: (10 * x["min_terms"] + abs(x["min_value"]) / 100 + x["max_complexity"], f"t={x["min_terms"]}_v={x["min_value"]}_c={x["max_complexity"]}"), "# Terms by Term Value by Complexity"
    elif dataset_name == "palindrome_generation":
        # Sort by min_string_len
        return lambda x: (x["min_length"], str(x["min_length"])),"String Length"
    elif dataset_name == "palindrome_partitioning":
        # Sort by 10 * min_string_len + (max_substring_palindrome_len - min_substring_palindrome_len)
        #   Assume that min_string_len makes the problem harder, so we give it more weight
        #   The range of substring_palindrome_len is secondary factor
        return lambda x: (10 * x["min_string_len"] + (x["max_substring_palindrome_len"] - x["min_substring_palindrome_len"]), f"len={x["min_string_len"]}_substr_l=[{x["min_substring_palindrome_len"]},{x["max_substring_palindrome_len"]}]"), "String Length by Substring Palindrome Length"
    else:
        # No sorting function defined for this dataset
        return None, None

def plot_aggregated_line_plot(dataset_name: str, model_name_to_results: Dict, img_save_folder: str, train_range: str = ""):
    """
    Args:
        dataset_name: Name of the dataset
        model_name_to_results: Dict, each key is the name of the model, each value is a dict of the aggregated results
        img_save_folder: Directory path where to save the aggregated line plot
        train_range: Dict containing the training range parameters to display below the title
    """
    # Get the metric sorting function for the dataset
    metric_sorting_function, metric_sorting_label = get_metric_sorting_function(dataset_name)
    
    if metric_sorting_function is None:
        print(f"No sorting function defined for dataset: {dataset_name}")
        return

    # Parse hyperparameter keys and organize data by metrics
    metric_to_model_data = defaultdict(lambda: defaultdict(list))  # metric_name -> model_name -> [(sort_value, display_value, y_value, error)]

    model_name_to_plot = get_model_name_to_plot(model_name_to_results)

    for model_name, aggregated_results in model_name_to_results.items():
        for hyperparam_key, hyperparam_value in aggregated_results.items():
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
            
            # Get the sorting value and display value for this hyperparameter set
            try:
                sort_value, display_value = metric_sorting_function(hyperparams)
            except KeyError as e:
                print(f"Warning: Missing hyperparameter for sorting: {e}")
                continue
            
            # Extract metrics for this hyperparameter set
            for metric_name, metric_data in hyperparam_value.items():
                if isinstance(metric_data, dict) and 'average' in metric_data and 'standard_error' in metric_data:
                    y_value = metric_data['average']
                    error_value = metric_data['standard_error']
                    metric_to_model_data[metric_name][model_name].append((sort_value, display_value, y_value, error_value))
    
    # Create output directory
    for metric_name in metric_to_model_data.keys():
        os.makedirs(os.path.join(img_save_folder, metric_name), exist_ok=True)
    
    # Create line plots for each metric using matplotlib
    for metric_name, model_data in metric_to_model_data.items():
        if not model_data:
            continue
            
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Define colors for different models
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_data)))
        
        # Plot lines for each model
        for i, (model_name, data_points) in enumerate(model_data.items()):
            if not data_points:
                continue
                
            # Sort data points by sort_value (first element)
            data_points.sort(key=lambda x: x[0])
            sort_values = [point[0] for point in data_points]
            display_values = [point[1] for point in data_points]
            y_values = [point[2] for point in data_points]
            error_values = [point[3] for point in data_points]
            
            # Calculate upper and lower bounds for shaded region
            y_upper = [y + err for y, err in zip(y_values, error_values)]
            y_lower = [y - err for y, err in zip(y_values, error_values)]
            
            # Plot the main line using display values for x-axis
            plt.plot(
                display_values, y_values,
                label=model_name, color=colors[i],
                linewidth=2, marker='o', markersize=6
            )
            
            # Add shaded region for error bounds using display values
            plt.fill_between(
                display_values, y_lower, y_upper,
                color=colors[i], alpha=0.2
            )
        
        # Customize plot
        plt.title(f"{dataset_name}: {metric_name} vs {metric_sorting_label}", fontsize=14, fontweight='bold', pad=20)
        
        # Add training range information below the title if available
        if train_range:
            # Position the text below the title using axes coordinates
            ax = plt.gca()
            ax.text(0.5, 1.15, f"Training Range: {train_range}", ha='center', va='bottom', 
                   fontsize=10, style='italic', transform=ax.transAxes)
        
        plt.xlabel(metric_sorting_label, fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels 45 degrees
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Add extra top margin to create more space between title and plot
        plt.subplots_adjust(top=0.85)
        
        # Save plot
        safe_metric_name = metric_name.replace('/', '_').replace(' ', '_')
        plot_filename = f"{model_name_to_plot}_{dataset_name}_{safe_metric_name}_line_plot.png"
        plot_path = os.path.join(img_save_folder, metric_name, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Line plot for {metric_name} saved to: {plot_path}")
        
        # Clear the figure to avoid overlapping plots
        plt.close()

    
def eval_sweep(models, config, generation_config, output_dir, device, get_eval_data, use_aggregated_results):
    """
    Args:
        models: List of model names to evaluate
        config: Configuration for the evaluation
        generation_config: Configuration for the model generation
        output_dir: Directory to save the evaluation results
        device: Device to run evaluation on
        plot_eval: Whether to plot the evaluation results
    """
    aggreated_results_dict = {dataset_config["name"]: {get_model_name_to_save(model_name): {} for model_name in models} for dataset_config in config["datasets"]}  # dataset_name -> model_name -> aggregated_results

    for model_name in models:
        print(f"\033[32mLoading model: {model_name}\033[0m")
        model_name_to_save = get_model_name_to_save(model_name)
        print(f"\033[32mModel name to save: {model_name_to_save}\033[0m")

        if get_eval_data:
            # Assume that we are just getting the evaluation data, so we don't need to load the model (for speed)
            model = None
            tokenizer = None
        else:
            model, tokenizer = load_model(model_name, device)
            model.eval()
            
            # Verify model size and basic config
            try:
                total_params = model.num_parameters()
            except Exception:
                total_params = sum(p.numel() for p in model.parameters())
            try:
                mem_bytes = model.get_memory_footprint()
            except Exception:
                mem_bytes = None
            def _fmt_count(n):
                return f"{n/1e9:.2f}B" if n >= 1e9 else (f"{n/1e6:.2f}M" if n >= 1e6 else (f"{n/1e3:.2f}K" if n >= 1e3 else str(n)))
            if mem_bytes is not None:
                print(f"\033[36mModel params: {_fmt_count(total_params)}, memory footprint: {mem_bytes/1024/1024/1024:.2f} GiB\033[0m")
            else:
                print(f"\033[36mModel params: {_fmt_count(total_params)}\033[0m")
            hidden_size = getattr(model.config, "hidden_size", None)
            num_layers = getattr(model.config, "num_hidden_layers", None)
            if hidden_size is not None and num_layers is not None:
                print(f"\033[36mConfig: hidden_size={hidden_size}, num_layers={num_layers}\033[0m")

        for dataset_config in config["datasets"]:
            # Load evaluator
            if get_eval_data:
                dataset_evaluator = None
            else:
                dataset_evaluator = evaluator.get_evaluator(f"{dataset_config['name']}.reasoning_gym")  # Need to add .reasoning_gym to be compatible with the evaluator

            # Extract the dataset name and merge params into the config
            dataset_name = dataset_config["name"]
            dataset_params = dataset_config.get("params", {})
            full_dataset_config = {"name": dataset_name, **dataset_params}
            
            # Evaluate model on this dataset configuration
            aggregated_results = eval_one_dataset(
                eval_mode=mode,
                model_name_to_save=model_name_to_save,
                model=model, 
                tokenizer=tokenizer,
                dataset_evaluator=dataset_evaluator,
                dataset_config=full_dataset_config,
                model_generation_config=generation_config,
                num_problem_per_hyperparam=config["num_problem_per_hyperparam"],
                output_dir=output_dir,
                device=device,
                get_eval_data=get_eval_data,
                use_aggregated_results=use_aggregated_results
            )
            aggreated_results_dict[dataset_name][model_name_to_save] = aggregated_results

    return aggreated_results_dict


def plot_sweep(aggregated_results_dict, output_dir, plot_detailed_sweep, config):
    """
    Args:
        aggregated_results_dict: Dict, each key is the name of the dataset, each value is a dict of the aggregated results
        output_dir: Directory to save the evaluation results
        plot_detailed_sweep: Whether to plot the detailed sweep of the evaluation results
        config: Configuration dict containing dataset configs with training ranges
    """
    for dataset_name, model_name_to_results in aggregated_results_dict.items():
        # Extract training range from config if available
        train_range = None
        for dataset_config in config["datasets"]:
            if dataset_config.get("name") == dataset_name and "train_range" in dataset_config:
                train_range_dict = dataset_config["train_range"]
                for param_name, param_value in train_range_dict.items():
                    train_range = "+".join([f"{param_name}={param_value}" for param_name, param_value in train_range_dict.items()])
                break  # Stop after finding the first matching dataset config
        
        plot_aggregated_line_plot(dataset_name, model_name_to_results, img_save_folder=os.path.join(output_dir, "_consolidated_line_plots", f"{dataset_name}_s={SEED}"), train_range=train_range)
   
    if plot_detailed_sweep:
        for dataset_name, model_name_to_results in aggregated_results_dict.items():
            for model_name_to_save, aggregated_results in model_name_to_results.items():
                # Generate interactive parallel coordinates plot
                img_save_path = os.path.join(output_dir, f"{dataset_name}_s={SEED}", model_name_to_save)
                html_save_path = os.path.join(output_dir, "_consolidated_html", model_name_to_save)
                
                # Create a minimal dataset config for the plotting function
                dataset_config = {"name": dataset_name}
                plot_detailed_sweep_results(dataset_config, aggregated_results, img_save_path, html_save_path)

def get_model_name_to_save(model_name: str) -> str:
    """
    Get the model name to save the results to
    """
    if "multi_task_rl_llms-" in model_name:
        model_name_to_save = model_name.split("multi_task_rl_llms-")[-1]
        model_name_to_save = "-".join(model_name_to_save.split("/"))
        return model_name_to_save
    else:
        return model_name.split("/")[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/eval_varying_train_hyperparam.yaml", help="Path to the config file. Contain (1) # of questions to eval per settting (2) List of datasets to eval on")
    parser.add_argument("-gc", "--generation_config", type=str, default="configs/eval_generation_config.yaml", help="Path to the generation config file")
    parser.add_argument("-m", "--models", nargs="+", default=["Qwen/Qwen2.5-7B-Instruct"], help="List of model paths to evaluate on. Can be HuggingFace model names (e.g., 'Qwen/Qwen2.5-7B-Instruct') or local paths to downloaded models (e.g., 'models/multi_task_rl_llms-family_relationships/multi_task_rl_llms-family_relationships/checkpoint_1000')")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="Directory to save the evaluation results")
    parser.add_argument("-a", "--use_aggregated_results", action="store_true", default=False, help="Whether to use the aggregated results instead of loading individual results's yaml and/or re-evaluating the model")
    parser.add_argument("-p", "--plot_eval", action="store_true", default=False, help="Whether to plot the evaluation results")
    parser.add_argument("--plot_detailed_sweep", action="store_true", default=False, help="Whether to plot the detailed sweep of the evaluation results")
    args = parser.parse_args()

    set_seed(SEED)

    # Detect the mode of hyperparameter evaluation
    mode = "train" if "train" in args.config else "test"
    print(f"\033[32mRunning {mode} mode\033[0m")

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open(args.generation_config, "r") as f:
        generation_config = yaml.load(f, Loader=yaml.FullLoader)

    # Detect the device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = os.path.join(args.output_dir, "reasoning_gym")

    aggregated_results_dict = eval_sweep(args.models, config, generation_config, output_dir, device, args.plot_eval, args.use_aggregated_results)

    if args.plot_eval:
        plot_sweep(aggregated_results_dict, output_dir, args.plot_detailed_sweep, config)