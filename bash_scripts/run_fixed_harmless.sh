#!/bin/bash
#SBATCH --job-name=fixed-harmless
#SBATCH --partition=gpu
#SBATCH --account=slurm-students
#SBATCH --output=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal/slurm-%j.out
#SBATCH --chdir=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal

source ~/thesis_experiment/Multilingual-Refusal/venv/bin/activate
export PYTHONPATH=/home/h24/baga0553/thesis_experiment/Multilingual-Refusal:$PYTHONPATH

for coeff in 1.0 5.0 10.0 20.0; do
    echo "=== Fixed Addition coeff=$coeff (harmless) ==="
    python3 -c "
import torch, json, os, random, mmengine, sys
sys.path.insert(0, '.')
from dataset.load_dataset import load_dataset_split
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import get_mean_activations
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak

cfg = mmengine.Config.fromfile('pipeline/runs/Qwen2.5-7B-Instruct/zh/zh_local.yaml')
model_base = construct_model_base(cfg.model_path, 'en')

random.seed(42)
harmful_en = random.sample(load_dataset_split(harmtype='harmful', split='train', lang='en', instructions_only=True), 128)
harmless_en = random.sample(load_dataset_split(harmtype='harmless', split='train', lang='en', instructions_only=True), 128)

def get_acts(instructions):
    return get_mean_activations(system=None, model=model_base.model, tokenizer=model_base.tokenizer,
        instructions=instructions, tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules, positions=[-1])

harmful_acts = get_acts(harmful_en)
harmless_acts = get_acts(harmless_en)
direction = harmful_acts[0, 15, :] - harmless_acts[0, 15, :]
direction = direction / (direction.norm() + 1e-8)

coeff = $coeff
hook = get_activation_addition_input_pre_hook(vector=direction, coeff=coeff)
fwd_pre_hooks = [(model_base.model_block_modules[15], hook)]

random.seed(42)
test_data = load_dataset_split(harmtype='harmless', split='val', lang='zh')
completions = model_base.generate_completions(test_data, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=[], max_new_tokens=cfg.max_new_tokens, batch_size=cfg.batch_size, system=None)

os.makedirs('output/fixed_addition/zh', exist_ok=True)
evaluation = evaluate_jailbreak(completions=completions, methodologies=['substring_matching'], evaluation_path=f'output/fixed_addition/zh/eval_harmless_coeff_{coeff}.json')
print(f'coeff={coeff}, False Refusal Rate={evaluation[\"substring_matching_success_rate\"]:.4f}')
"
done
echo "Done!"
