#!/bin/bash
#SBATCH --job-name=dri-sweep2
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal/slurm-%j.out
#SBATCH --chdir=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal

source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:$PYTHONPATH

for alpha in 0.001 0.005 0.01 0.05; do
    echo "=== DRI alpha=$alpha harmful ==="
    python3 scripts/run_dri.py \
        --config pipeline/runs/Qwen2.5-7B-Instruct/zh/zh_local.yaml \
        --lang zh --layer 15 --alpha $alpha

    echo "=== DRI alpha=$alpha harmless ==="
    python3 scripts/run_dri.py \
        --config pipeline/runs/Qwen2.5-7B-Instruct/zh/zh_local.yaml \
        --lang zh --layer 15 --alpha $alpha --harmless
done

echo "Done!"
