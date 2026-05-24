#!/bin/bash
#SBATCH --job-name=test-env
#SBATCH --partition=gpu
#SBATCH --account=slurm-students

source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate

python3 -c "import torch; print(torch.__version__)"
python3 -c "import transformers; print(transformers.__version__)"
python3 -c "import torch; print('GPU available:', torch.cuda.is_available())"
python3 -c "import torch; print('GPU name:', torch.cuda.get_device_name(0))"

