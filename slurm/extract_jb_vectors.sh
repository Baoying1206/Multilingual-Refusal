#!/bin/bash
#SBATCH --job-name=extract-jb-vec
#SBATCH --partition=gpu
#SBATCH --account=slurm-students 
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

python scripts/extract_jailbreak_vectors.py \
    --model_path   "$MODEL_PATH" \
    --model_alias  "$MODEL_ALIAS" \
    --baseline_dir "output/ja_vector_sweep" \
    --exp_id       "20250519-232436/1" \
    --output_dir   "output/jailbreak_analysis/$MODEL_ALIAS" \
    --n_train      128 \
    --batch_size   8
  
echo "Done: $(date)"
