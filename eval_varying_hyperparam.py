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
            param_ranges[param_name] = list(range(start, end, step))
    
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

    print(json.dumps(hyperparam_sets, indent=4))
    print(f"Total number of hyperparam sets: {len(hyperparam_sets)}")
    input("Press Enter to continue...")
    
    return hyperparam_sets

def eval_one_dataset(
    model_name: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    evaluator: evaluator.RewardEvaluator,
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
        evaluator: Evaluator to use
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
    
    # Initialize aggregated results collection
    all_aggregated_results = {}

    for i in range(len(hyperparam_sets)):
        hyperparam_set = hyperparam_sets[i]
        hyperparam_set_path = os.path.join(output_path, f"{'_'.join([f'{k}={v}' for k, v in hyperparam_set.items()])}")
        os.makedirs(hyperparam_set_path, exist_ok=True)

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
            rewards_per_func, metrics = evaluator.compute_rewards(
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
        
        # Calculate aggregated results for this hyperparameter set
        hyperparam_key = '_'.join([f'{k}={v}' for k, v in hyperparam_set.items()])
        
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
    aggregated_results_file = os.path.join(output_path, "aggregated_results.json")
    with open(aggregated_results_file, 'w') as f:
        json.dump(all_aggregated_results, f, indent=4)
    
    print(f"All aggregated results saved to: {aggregated_results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/eval_varying_hyperparam.yaml")
    parser.add_argument("-m", "--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("-o", "--output_dir", type=str, default="output")
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
        evaluator = evaluator.get_evaluator(f"{dataset_config['name']}.reasoning_gym")  # Need to add .reasoning_gym to be compatible with the evaluator

        # Extract the dataset name and merge params into the config
        dataset_name = dataset_config["name"]
        dataset_params = dataset_config.get("params", {})
        full_dataset_config = {"name": dataset_name, **dataset_params}
        
        # Evaluate model on this dataset configuration
        eval_one_dataset(
            model_name=args.model,
            model=model, 
            tokenizer=tokenizer,
            evaluator=evaluator,
            dataset_config=full_dataset_config,
            model_generation_config=config["model_generation_config"],
            num_problem_per_hyperparam=config["num_problem_per_hyperparam"],
            output_dir=output_dir,
            device=device
        )

