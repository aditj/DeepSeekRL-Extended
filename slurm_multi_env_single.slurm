#!/bin/bash
#SBATCH --job-name=grpo_single_multi_env
#SBATCH --gres=gpu:h100:3    # Request 1× H100 GPU
#SBATCH --time=48:00:00       # HH:MM:SS
#SBATCH --array=0-5       # 9 environments (0-8)
#SBATCH --output=std/%x_%A_%a.out   # Stdout/stderr per-array-task

# Activate environment

# Define the 9 environments to train on
tasks=(
    "acre"
    "prime_factorization"
    "number_sequence"
    "graph_color"
    "color_cube_rotation"
    "shortest_path"
    "circuit_logic"
    "family_relationships"
    "syllogism"
    "zebra_puzzles"
)
tasks_hard=(
    "family_relationships.reasoning_gym.hard",
    "shortest_path.reasoning_gym.hard",
    "graph_color.reasoning_gym.hard",
    "number_sequence.reasoning_gym.hard",
    "palindrome_generation.reasoning_gym.hard",
    "palindrome_partitioning.reasoning_gym.hard",
)

# Get current task based on array index
task=${tasks_hard[$SLURM_ARRAY_TASK_ID]}
dataset="${task}"

# Define evaluation environments (evaluate on all 9 tasks)
eval_envs=(
    "acre.reasoning_gym"
    "prime_factorization.reasoning_gym"
    "number_sequence.reasoning_gym"
    "graph_color.reasoning_gym"
    "color_cube_rotation.reasoning_gym"
    "shortest_path.reasoning_gym"
    "circuit_logic.reasoning_gym"
    "family_relationships.reasoning_gym"
    "syllogism.reasoning_gym"
    "zebra_puzzles.reasoning_gym"
)
eval_envs_hard=(
    "acre.reasoning_gym"
    "graph_color.reasoning_gym"
    "number_sequence.reasoning_gym.hard"
    "zebra_puzzles.reasoning_gym"
)

# Training parameters
num_steps=10000
num_chains=5
eval_iterations=0
max_prompt_length=1024
max_completion_length=2048
temperature=0.9
learning_rate=5e-6
randomseed=420394
model_name="Qwen/Qwen2.5-7B-Instruct"

# Experiment naming
experiment_name="${task}.reasoning_gym.hard_7b_single_${num_steps}steps_eval${eval_iterations}_chains${num_chains}_seed${randomseed}_temp${temperature}"

echo "[$(date)] Starting task $SLURM_ARRAY_TASK_ID: ${task}"
echo "Training on: ${dataset}"
echo "Evaluating on: ${eval_envs[@]} ${eval_envs_hard[@]}"
echo "Experiment: ${experiment_name}"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run training with multi-environment evaluation
uv run scripts/main_single.py \
  --output_dir ../output_single_multi_env/grpo_${experiment_name} \
  --model_name $model_name \
  --dataset $dataset \
  --eval_environments ${eval_envs_hard[@]} \
  --num_chains $num_chains \
  --num_train_iters $num_steps \
  --eval_iterations $eval_iterations \
  --learning_rate $learning_rate \
  --temperature $temperature \
  --max_prompt_length $max_prompt_length \
  --max_completion_length $max_completion_length \
  --seed $randomseed \
  --checkpoint_frequency 1000 \

echo "[$(date)] Finished task $SLURM_ARRAY_TASK_ID: ${task}"

# Optionally analyze results after completion
if [ $? -eq 0 ]; then
    echo "[$(date)] Analyzing results..."
    python analyze_multi_env_results.py \
      --output_dir output_single_multi_env/grpo_${experiment_name} \
      --plot_dir output_single_multi_env/grpo_${experiment_name}/analysis_plots
    echo "[$(date)] Analysis complete"
fi
