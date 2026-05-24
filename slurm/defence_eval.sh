#!/bin/bash
#SBATCH --job-name=defense-eval
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --array=0-2
#SBATCH --output=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal/slurm-%A_%a.out
#SBATCH --chdir=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal

MODEL_PATHS=(
    "/home/h24/baga0553/models/Qwen2.5-7B-Instruct"
    "/home/h24/baga0553/models/Llama-3.1-8B-Instruct"
    "/home/h24/baga0553/models/gemma-2-9b-it"
)
MODEL_ALIASES=(
    "Qwen2.5-7B-Instruct"
    "Meta-Llama-3.1-8B-Instruct"
    "gemma-2-9b-it"
)

MODEL_PATH=${MODEL_PATHS[$SLURM_ARRAY_TASK_ID]}
MODEL_ALIAS=${MODEL_ALIASES[$SLURM_ARRAY_TASK_ID]}

echo "Model: $MODEL_ALIAS  Start: $(date)"

source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:$PYTHONPATH

python scripts/defense_evaluation.py \
    --model_path   "$MODEL_PATH" \
    --model_alias  "$MODEL_ALIAS" \
    --vector_dir   "output/jailbreak_analysis/$MODEL_ALIAS" \
    --baseline_dir "output/ja_vector_sweep" \
    --exp_id       "20250519-232436/1" \
    --output_dir   "output/defense/$MODEL_ALIAS" \
    --alpha        20.0 \
    --beta         1.0 \
    --layer        -1 \
    --batch_size   8 \
    --max_new_tokens 200

echo "Done: $(date)"
