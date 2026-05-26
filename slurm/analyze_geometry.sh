#!/bin/bash
#SBATCH --job-name=analyze-geometry
#SBATCH --partition=cpu
#SBATCH --account=slurm-students
#SBATCH --output=slurm/logs/geometry_%j.out

cd ~/thesis_experiment/Multilingual-Refusal
echo "Start: $(date)"

source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:$PYTHONPATH

python scripts/analyze_geometry.py \
    --results_dir  output/jailbreak_analysis \
    --transfer_dir output/transfer \
    --output_dir   output/figures

echo "Done: $(date)"
