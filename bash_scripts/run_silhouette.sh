#!/bin/bash
#SBATCH --job-name=silhouette
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal/slurm-%j.out
#SBATCH --chdir=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal

source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:$PYTHONPATH

python3 scripts/compute_silhouette.py \
    --config pipeline/runs/Qwen2.5-7B-Instruct/zh/zh_local.yaml \
    --layer 15

echo "Done!"
