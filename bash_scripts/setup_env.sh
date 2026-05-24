#!/bin/bash
#SBATCH --job-name=setup-env
#SBATCH --partition=gpu
#SBATCH --account=slurm-students

cd ~/thesis_experiment/Multilingual-Refusal

python3 -m venv venv
source venv/bin/activate

grep -v "pyairports" requirements.txt > requirements_clean.txt
pip install -r requirements_clean.txt

echo "Done!"
