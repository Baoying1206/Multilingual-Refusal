#!/bin/bash
#SBATCH --job-name=defense-eval
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/defense_%A_%a.out

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
# Gemma needs smaller alpha — alpha=20 over-steers at its very late k* (layer 39/41)
ALPHAS=(
    "20.0"
    "20.0"
    "5.0"
)

MODEL_PATH=${MODEL_PATHS[$SLURM_ARRAY_TASK_ID]}
MODEL_ALIAS=${MODEL_ALIASES[$SLURM_ARRAY_TASK_ID]}
ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}

echo "Model: $MODEL_ALIAS  Alpha: $ALPHA  Start: $(date)"

cd ~/thesis_experiment/Multilingual-Refusal
source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:$PYTHONPATH

python scripts/defense_evaluation.py \
    --model_path      "$MODEL_PATH" \
    --model_alias     "$MODEL_ALIAS" \
    --vector_dir      "output/jailbreak_analysis/$MODEL_ALIAS" \
    --baseline_dir    "output/ja_vector_sweep" \
    --exp_id          "20250519-232436/1" \
    --output_dir      "output/defense_v2/$MODEL_ALIAS" \
    --alpha           "$ALPHA" \
    --beta            1.0 \
    --layer           -1 \
    --batch_size      8 \
    --max_new_tokens  200 \
    --harmless_n_test 128

echo "Done: $(date)"
