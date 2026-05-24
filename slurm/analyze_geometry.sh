#!/bin/bash
#SBATCH --job-name=analyze-geometry
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal/slurm-%j.out
#SBATCH --chdir=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal

echo "Start: $(date)"

source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:$PYTHONPATH

python scripts/analyze_geometry.py \
    --results_dir output/jailbreak_analysis \
    --output_dir  output/figures

echo "Done: $(date)"
