#!/bin/bash
#SBATCH --job-name=download-models
#SBATCH --partition=gpu
#SBATCH --account=slurm-students

source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
cd ~/thesis_experiment/Multilingual-Refusal

huggingface-cli login -- HUGGINGFACE

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir ~/models/Qwen2.5-7B-Instruct

huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir ~/models/Llama-3.1-8B-Instruct

echo "Done!"
