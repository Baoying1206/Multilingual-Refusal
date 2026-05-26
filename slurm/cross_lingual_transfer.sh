#!/bin/bash
#SBATCH --job-name=xling-transfer
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/transfer_%A_%a.out

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
# Source languages that have extractable jailbreak vectors per model:
#   Qwen:  yo,ko,ja  (all 14 available; pick highest-bypass sources)
#   LLaMA: yo,ko,ja  (sufficient data for these)
#   Gemma: yo,ko,ja  (only ja/ko/th/yo/ar available; yo most extreme)
SOURCE_LANGS=(
    "yo,ko,ja"
    "yo,ko,ja"
    "yo,ko,ja"
)
# Target: high-resource languages (high baseline refusal, low bypass)
TARGET_LANGS=(
    "en,de,es,fr,it"
    "en,de,es,fr,it"
    "en,de,es,fr,it"
)
# Gemma uses alpha=5 to avoid catastrophic generation (same as defense_eval)
ALPHAS=(
    "20.0"
    "20.0"
    "5.0"
)

MODEL_PATH=${MODEL_PATHS[$SLURM_ARRAY_TASK_ID]}
MODEL_ALIAS=${MODEL_ALIASES[$SLURM_ARRAY_TASK_ID]}
SRC=${SOURCE_LANGS[$SLURM_ARRAY_TASK_ID]}
TGT=${TARGET_LANGS[$SLURM_ARRAY_TASK_ID]}
ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}

echo "Model: $MODEL_ALIAS  src=$SRC  tgt=$TGT  alpha=$ALPHA  Start: $(date)"

cd ~/thesis_experiment/Multilingual-Refusal
mkdir -p slurm/logs
source venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:$PYTHONPATH

python scripts/cross_lingual_transfer.py \
    --model_path      "$MODEL_PATH" \
    --model_alias     "$MODEL_ALIAS" \
    --vector_dir      "output/jailbreak_analysis/$MODEL_ALIAS" \
    --baseline_dir    "output/ja_vector_sweep" \
    --exp_id          "20250519-232436/1" \
    --output_dir      "output/transfer/$MODEL_ALIAS" \
    --source_langs    "$SRC" \
    --target_langs    "$TGT" \
    --alpha           "$ALPHA" \
    --batch_size      8 \
    --max_new_tokens  200

echo "Done: $(date)"
