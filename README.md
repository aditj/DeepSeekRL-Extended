
# DeepSeek R1 Implementation

## Motivation
I wanted to recreate DeepSeek R1's results at a smaller scale, focusing on understanding the core mechanics by implementing everything from scratch. This repository trains Qwen1.5B on various reasoning datasets including the [grade school math dataset](https://github.com/openai/grade-school-math), with extensive enhancements for advanced token generation strategies and multi-dataset support.

This implementation heavily borrows from [Will Brown's  work](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb) ([@willccbb](https://x.com/willccbb)), but restructures the code into a format optimized for learning and experimentation.

The key difference in my implementation is computing the GRPO loss function directly rather than using external RL libraries, and reformatting into a multi script repo.

The implementation now includes several major enhancements:
- **Advanced Token Generation**: Mixture-of-token generation with phase transitions, supporting various sampling strategies (nucleus, Dirichlet, element-wise max)
- **Multi-Dataset Support**: Extended beyond GSM8K to include Reasoning Gym datasets, MBPP, LeetCode, and Math500
- **Safe Code Execution**: Subprocess-based code execution for programming tasks with crash protection
- **Enhanced Evaluation**: LLM-based answer evaluation and sophisticated reward functions
- **Comprehensive Logging**: Detailed generation logs with phase transition tracking and token selection analysis

I hope this might help other people understand things better, and maybe provide an easier way to try out smaller scale ideas etc.

## Installation
### Using uv
Inside the repository, run:
```bash
uv sync
```

Then, we install flash attention separately (Note: this is only tested on G2)
```bash
source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
```

For eval_varying_hyperparam.py, you need to install chrome so that we can render the html as plots as well:
```bash
plotly_get_chrome
kaleido_get_chrome
```

G2 Specific notes: gpt has prebuilt nvcc that needs to be pointed in the path, so add the following to your bashrc:
```bash
export CUDA_VERSION="cuda-12.1"
export PATH="/usr/local/$CUDA_VERSION/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/$CUDA_VERSION/lib64:$LD_LIBRARY_PATH
```

### Deprecated
```
pip install -r requirements.txt
```

Required environment variables:
```
export HUGGINGFACE_TOKEN="your-token-here"
huggingface-cli login
```

## Implementation Details

The system consists of several key modules:

### main.py
Contains the core training loop implementing GRPO (Generalized Reward-Powered Optimization). Features advanced generation capabilities including:
- Mixture-of-token generation with phase transitions
- Multiple sampling strategies (nucleus, Dirichlet, element-wise max)
- Comprehensive logging of generation steps and token selection
- Enhanced loss computation with embedding-based token selection

### llms.py
Manages model loading and configuration with enhanced features:
- Support for LLaMA and Qwen models through Hugging Face's transformers library
- Conditional flash attention support based on model type
- Optimized model loading for different architectures

### rldatasets.py
Handles dataset loading and preprocessing with expanded support:
- GSM8K, Math500, MBPP, and LeetCode datasets
- Full Reasoning Gym integration with task-specific configurations
- Flexible data loaders with custom preprocessing for different data formats

### evaluator.py
Contains evaluation metrics and reward functions with major enhancements:
- LLM-based answer evaluation using OpenAI API
- Safe subprocess execution for programming tasks with crash protection
- Reasoning Gym dataset scoring with custom reward functions
- Timeout handling and robust error management

### utils.py
Utility functions supporting advanced token processing:
- Memory-efficient selective log softmax operations
- Multiple token embedding and log probability computation methods
- Enhanced generation logging with dataset-specific formatting

### token_analysis.py
Provides detailed analysis of token probability distributions during single inference traces. Analyzes entropy patterns, token consistency, and probability landscapes to understand model behavior during generation.

### eval_varying_hyperparam.py
#### Motivation
Although reasoning_gym provides a set of hyperparameters for "easy" and "hard" problems, it is unclear whether those training/testing hyperparameters are reasonable for a larger 7B model. 

#### Overview
Evaluates a open-source model on a set of datasets with varying hyperparameters, and plots the results.

There are two types of varying hyperparameters: train v.s. test
- If the config file has `train` in the name, we assume that there is a range to sweep for EVERY hyperparameter (i.e., different ranges for min v.s. max)
- If the config file has `test` in the name, we assume that we are setting MOST of parameters that have min_ and max_ to be the same. Specifically, we only specify the min parameter, and the max parameter is set automatically. 
More on the config file in a subsection below. 

The general process is: 
1. Edit the config file (e.g., `configs/eval_train_varying_hyperparam.yaml`) as needed.
2. Run the script for evaluate the model.
3. Plot the results (by adding at least the `-p` flag)

#### 1. Edit the config file
##### Train mode

For training hyperparameter evaluation, see `configs/eval_varying_train_hyperparam.yaml` as an example. 

This config sweeps across ranges for **ALL hyperparameters**, allowing you to find optimal training parameters by testing different combinations of min/max values.

For example, the parameter ranges are specified as a list [start, end, step], where end is exclusive. For example, `min_family_size: [4, 17, 2]` means that we will sweep from 4 to 17 (exclusive) with a step of 2.

##### Test mode
For testing hyperparameter evaluation, see `configs/eval_varying_test_hyperparam.yaml` as an example. `configs/eval_varying_test_finetuned_model_hyperparam.yaml` is another example for testing a finetuned model on the same datasets, but with a smaller set of hyperparameters.

This config sets min and max values to be the same for most parameters, providing controlled variance for testing specific difficulty levels.

- **Basic**: Specify the range for the min parameter, and the max parameter is set automatically. 
  - For example, `min_family_size: [4, 17, 2]` means that we will sweep from 4 to 17 (exclusive) with a step of 2. max_family_size is always set to be the same as min_family_size.
- **Multiple ranges for min**: If you want to specify multiple ranges for the min parameter, you can do so by adding a `_part_1` and `_part_2` suffix to the parameter name. 
  - For example, `min_family_size_part_1: [4, 17, 2]` and `min_family_size_part_2: [8, 22, 2]` means that the `min_family_size` parameter will be swept from 4 to 17 (exclusive) with a step of 2, and from 8 to 22 (exclusive) with a step of 2.
- **Specify a max parameter**: If you want to specify a max parameter, you can do so by adding a `max_` prefix to the parameter name. 
  - For example, `max_family_size: [8, 22, 2]` means that we will sweep from 8 to 22 (exclusive) with a step of 2. The max hyperparameter value ignores `min_family_size`.


#### 2. Run the script to evaluate the model
Here are examples of how to run the script
```bash
python eval_varying_hyperparam.py
  -c <path_to_config_file>
  -m <huggingface_model_name> <or local_model_path>
  -o <output_dir>
```

For example, the script below will evaluate the Qwen-7B model and the model trained on the family_relationships dataset for 10000 steps. The config used is `configs/eval_varying_test_finetuned_model_hyperparam.yaml`.
```bash
python scripts/eval_varying_hyperparam.py 
  -c configs/eval_varying_test_finetuned_model_hyperparam.yaml
  -m Qwen/Qwen2.5-7B-Instruct models/multi_task_rl_llms-family_relationships/checkpoint_10000
```

#### 3. Plot the results
The plots are saved in output_dir/reasoning_gym/_consolidated_line_plots/ and output_dir/reasoning_gym/_consolidated_html/.
```bash
python eval_varying_hyperparam.py
  -c <path_to_config_file>
  -m <huggingface_model_name> <or local_model_path>
  -o <output_dir>
  -p # Most importantly, add this flag
  -a # Optionally: Add this flag to make plotting faster. It directly read the _aggregated_results.json, which has results for all the hyperparameter sets.
  --plot_detailed_sweep # Optionally: Add this flag to plot the detailed sweep of the evaluation results. Used mostly for the train mode.
```

For example, the script below will plot the results for the Qwen-7B model and the model trained on the family_relationships dataset for 10000 steps. The config used is `configs/eval_varying_test_finetuned_model_hyperparam.yaml`.
```bash
python scripts/eval_varying_hyperparam.py
  -c configs/eval_varying_test_finetuned_model_hyperparam.yaml
  -m Qwen/Qwen2.5-7B-Instruct models/multi_task_rl_llms-family_relationships/checkpoint_10000
  -p -a
```

## Token Probability Analysis
Explore model behavior during inference:
```bash
# Analyze token distributions for a single reasoning trace
python token_analysis.py --output_dir "analysis_results"

# Or use the convenience script
./run_token_analysis.sh
```

This generates comprehensive visualizations including:
- Entropy evolution during generation
- Top-k token probability heatmaps  
- Token consistency analysis
- Probability landscape visualization
- Individual token PDFs (one plot per generated token showing full probability distribution)

For faster analysis, skip the individual PDFs:
```bash
python token_analysis.py --output_dir "fast_analysis" --skip_individual_pdfs
```

## Soft Thinking Mode 🧠

This implementation includes an experimental "soft thinking" feature inspired by recent research on preserving information during token generation. Instead of always sampling a single token from the probability distribution (which can lose information), soft thinking:

1. **Samples top-k tokens** with their probabilities
2. **Creates weighted embeddings** by mixing the top-k token embeddings based on their probabilities  
3. **Feeds mixed embeddings** to the next layer, preserving the superposition of likely tokens
4. **Exits to normal generation** when `</reasoning>` becomes the most likely token

### Usage

Enable soft thinking mode:
```bash
# Basic soft thinking (top-2 tokens)
python main.py --soft_thinking

# Customize parameters
python main.py --soft_thinking --soft_thinking_k 3 --soft_thinking_temperature 1.2
```

### Parameters

- `--soft_thinking`: Enable soft thinking mode
- `--soft_thinking_k`: Number of top tokens to mix (default: 2)
- `--soft_thinking_temperature`: Temperature for probability mixing (default: 1.0)

### Testing

Test both generation modes:
```bash
python test_soft_thinking.py
```

This will verify that both normal and soft thinking generation work correctly and show the differences in token handling.

## Results
Training was conducted on a single H100 GPU. After ~400 training steps:

![Training Results](plots/train_score.png)

And results on the validation set - this shows a clearer sign of learning:
![Eval Results](plots/eval_score.png)

## New Features

### Mixture-of-Token Generation
The implementation now supports advanced token generation strategies:
- **Phase Transitions**: Automatic switching between mixture and normal generation modes using `</think>` token triggers
- **Multiple Sampling Strategies**:
  - Standard mixture sampling with top-k token selection
  - Nucleus sampling for dynamic token selection
  - Dirichlet sampling for probabilistic mixture weights
  - Element-wise max for deterministic selection
- **Token Embedding Tracking**: Detailed logging of token embeddings and selection probabilities

### Multi-Dataset Support
Extended beyond GSM8K to support:
- **GSM8K**: Original grade school math problems
- **Math500**: More challenging math problems
- **MBPP**: Python programming problems with safe execution
- **LeetCode**: Competitive programming problems
- **Reasoning Gym**: Various reasoning tasks (shortest path, family relationships, number sequences, maze, sokoban)

### Advanced Configuration
The training script supports numerous experimental configurations:
- `--experiment_name`: Configure different token generation strategies
- `--mixture_k`: Control number of tokens in mixture
- `--loss_on_all_tokens`: Toggle loss computation strategy
- `--normal_generation`: Fallback to standard generation for comparison

## Future Directions
I'm really pleased to see how well the key mechanics work even in this simplified implementation. Building on this, I am very excited about several directions:

1. **Self-play capabilities** where agents compete and learn from each other using relative rewards. This would create a more dynamic training environment where the reward signal comes from agent interactions rather than fixed metrics.

2. **Implementing soft reward structures**, particularly for complex reasoning tasks. I've written a framework for AI debate that I'm excited to try out.

3. **Expanding into vision-language models (VLMs)** to improve world modeling capabilities. I have an idea about using R1-style training to enhance how VLMs build and maintain internal world models that I'm really excited to explore. (Really excited about this idea - if anyone else is interested I would love to talk.)

4. **I'd like to do all this experimentation in this framework**, so I need to make things faster, and support multi-gpu training.

5. **Enhanced mixture-of-token strategies** - exploring more sophisticated token selection mechanisms and phase transition triggers.

6. **Multi-model architectures** - extending beyond Qwen to support other model families and hybrid architectures.

7. **Advanced evaluation frameworks** - developing more sophisticated reward functions and evaluation metrics for different task domains.

8. **Production optimization** - implementing model distillation, quantization, and other techniques for deployment.



