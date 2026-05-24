import torch
import sys
import random

from dataset.load_dataset import load_dataset_split
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import get_mean_activations

model_base = construct_model_base('/home/h24/baga0553/models/Qwen2.5-7B-Instruct', 'en')

random.seed(42)
harmful_en = random.sample(load_dataset_split(harmtype='harmful', split='train', lang='en', instructions_only=True), 128)
harmless_en = random.sample(load_dataset_split(harmtype='harmless', split='train', lang='en', instructions_only=True), 128)
harmless_zh = random.sample(load_dataset_split(harmtype='harmless', split='val', lang='zh', instructions_only=True), 32)
harmful_zh = random.sample(load_dataset_split(harmtype='harmful', split='train', lang='zh', instructions_only=True), 32)

def get_acts(instructions):
    return get_mean_activations(
        system=None, model=model_base.model, tokenizer=model_base.tokenizer,
        instructions=instructions, tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules, positions=[-1]
    )

harmful_acts = get_acts(harmful_en)
harmless_acts = get_acts(harmless_en)
direction = harmful_acts[0, 15, :] - harmless_acts[0, 15, :]
direction = direction / (direction.norm() + 1e-8)

harmless_zh_acts = get_acts(harmless_zh)
harmful_zh_acts = get_acts(harmful_zh)

p_harmless_zh = (harmless_zh_acts[0, 15, :] @ direction).item()
p_harmful_zh = (harmful_zh_acts[0, 15, :] @ direction).item()
p_harmless_en = (harmless_acts[0, 15, :] @ direction).item()
p_harmful_en = (harmful_acts[0, 15, :] @ direction).item()

print(f"en harmful p:  {p_harmful_en:.4f}")
print(f"en harmless p: {p_harmless_en:.4f}")
print(f"zh harmful p:  {p_harmful_zh:.4f}")
print(f"zh harmless p: {p_harmless_zh:.4f}")
