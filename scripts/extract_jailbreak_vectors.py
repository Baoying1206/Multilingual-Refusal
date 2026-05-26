"""
Extract jailbreak vectors and refusal directions, then compute their geometric relationship.

Jailbreak vector (per language) = mean(bypassed_activations) - mean(refused_activations)
  - bypassed/refused labels come from WildGuard evaluation (already computed)
  - "bypassed" = model complied with harmful request (wildguard.refusal == 0)
  - "refused"  = model refused the harmful request  (wildguard.refusal == 1)

Refusal direction (per language) = mean(harmful_activations) - mean(harmless_activations)
  - extracted from PolyRefuse train split

For each language pair (jailbreak_lang, refusal_lang), compute layer-wise cosine similarity.

Usage:
  python scripts/extract_jailbreak_vectors.py \
      --model_path /path/to/Qwen2.5-7B-Instruct \
      --model_alias Qwen2.5-7B-Instruct \
      --baseline_dir output/ja_vector_sweep \
      --exp_id 20250519-232436/1 \
      --output_dir output/jailbreak_analysis \
      --batch_size 8
"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset.load_dataset import load_dataset_split
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import get_mean_activations


LANGS = ['en', 'de', 'ja', 'ko', 'zh', 'ru', 'th', 'yo', 'ar', 'es', 'fr', 'it', 'nl', 'pl']
MIN_SAMPLES = 20  # minimum bypassed or refused samples to compute a reliable vector


def load_bypassed_refused(eval_path):
    """Load instructions split by WildGuard refusal label."""
    with open(eval_path) as f:
        ev = json.load(f)
    completions = ev['completions']

    bypassed = [c['instruction'] for c in completions
                if c.get('wildguard', {}).get('refusal', 1) == 0]
    refused  = [c['instruction'] for c in completions
                if c.get('wildguard', {}).get('refusal', 1) == 1]
    return bypassed, refused


def extract_mean_activations(model_base, instructions, batch_size, desc=""):
    """Extract mean activations across all layers at the last token position."""
    if len(instructions) == 0:
        return None
    return get_mean_activations(
        system=None,
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        batch_size=batch_size,
        positions=[-1],   # last token position
    ).squeeze(0)  # [n_layers, d_model]


def extract_jailbreak_vector(model_base, bypassed, refused, batch_size):
    """
    Jailbreak vector = mean(bypassed activations) - mean(refused activations).
    Returns None if either group is too small.
    """
    if len(bypassed) < MIN_SAMPLES or len(refused) < MIN_SAMPLES:
        return None, len(bypassed), len(refused)

    act_bypassed = extract_mean_activations(model_base, bypassed, batch_size, "bypassed")
    act_refused  = extract_mean_activations(model_base, refused,  batch_size, "refused")

    jb_vec = act_bypassed - act_refused  # [n_layers, d_model]
    return jb_vec, len(bypassed), len(refused)


def extract_refusal_direction(model_base, lang, n_train, random_seed, batch_size):
    """
    Refusal direction = mean(harmful activations) - mean(harmless activations).
    Uses PolyRefuse train split.
    """
    random.seed(random_seed)
    harmful  = random.sample(
        load_dataset_split('harmful',  'train', lang=lang, instructions_only=True), n_train)
    harmless = random.sample(
        load_dataset_split('harmless', 'train', lang=lang, instructions_only=True), n_train)

    act_harmful  = extract_mean_activations(model_base, harmful,  batch_size)
    act_harmless = extract_mean_activations(model_base, harmless, batch_size)

    refusal_dir = act_harmful - act_harmless  # [n_layers, d_model]
    return refusal_dir


def extract_harmfulness_direction(model_base, bypassed, lang, n_train, random_seed, batch_size):
    """
    Harmfulness direction = mean(bypassed_harmful) - mean(harmless).
    Captures pure harm signal without refusal component:
    bypassed samples are harmful inputs the model complied with, so they carry
    harmfulness encoding but NOT the refusal signal.
    """
    random.seed(random_seed + 99)
    harmless = random.sample(
        load_dataset_split('harmless', 'train', lang=lang, instructions_only=True), n_train)

    sample = bypassed[:n_train] if len(bypassed) > n_train else bypassed
    act_bypassed = extract_mean_activations(model_base, sample,   batch_size)
    act_harmless = extract_mean_activations(model_base, harmless, batch_size)

    return act_bypassed - act_harmless  # [n_layers, d_model]


def compute_cosine_similarity(vec_a, vec_b):
    """Layer-wise cosine similarity between two [n_layers, d_model] tensors."""
    return F.cosine_similarity(
        vec_a.float(), vec_b.float(), dim=-1
    ).tolist()  # [n_layers]


def _find_model_dir(baseline_dir, model_alias):
    """
    Search for model_alias in baseline_dir which has structure:
      baseline_dir/{org}/{model_name}/{lang}/...
    Returns the full path to {org}/{model_name} if found.
    """
    for entry in os.listdir(baseline_dir):
        top = os.path.join(baseline_dir, entry)
        if not os.path.isdir(top):
            continue
        # Direct match (e.g. baseline_dir/Qwen2.5-7B-Instruct)
        if entry == model_alias:
            return top
        # Two-level match (e.g. baseline_dir/Qwen/Qwen2.5-7B-Instruct)
        for sub in os.listdir(top):
            if sub == model_alias:
                return os.path.join(top, sub)
    return None


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    model_alias = args.model_alias or os.path.basename(args.model_path)

    print(f"\n{'='*60}")
    print(f"Model: {model_alias}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    results = {
        'model': model_alias,
        'jailbreak_vectors': {},      # lang -> layer-wise vector stats
        'refusal_directions': {},     # lang -> extracted
        'harmfulness_directions': {}, # lang -> extracted
        'cosine_similarities': {},    # (jb_lang, refusal_lang) -> [cos_sim per layer]
        'three_way_similarities': {}, # lang -> {jb_vs_refusal, jb_vs_harmfulness, refusal_vs_harmfulness}
    }

    # ── Step 1: Collect bypassed/refused instructions for each language ──────
    print("Step 1: Loading WildGuard-labeled baseline completions...")
    lang_bypassed = {}
    lang_refused  = {}

    # baseline_dir has two-level org/model structure: find the model subdir
    model_dir = _find_model_dir(args.baseline_dir, model_alias)
    if model_dir is None:
        raise FileNotFoundError(
            f"Cannot find '{model_alias}' under '{args.baseline_dir}'. "
            f"Available: {os.listdir(args.baseline_dir)}"
        )
    print(f"  Found model dir: {model_dir}\n")

    for lang in LANGS:
        eval_path = os.path.join(
            model_dir, lang, args.exp_id,
            'completions', 'harmful_baseline_evaluations.json'
        )
        if not os.path.exists(eval_path):
            print(f"  [{lang}] No baseline evaluation found, skipping.")
            continue

        bypassed, refused = load_bypassed_refused(eval_path)
        lang_bypassed[lang] = bypassed
        lang_refused[lang]  = refused
        print(f"  [{lang}] bypassed={len(bypassed)}, refused={len(refused)}", end="")
        if len(bypassed) < MIN_SAMPLES:
            print(f"  ← bypassed too few (< {MIN_SAMPLES}), jailbreak vector skipped")
        elif len(refused) < MIN_SAMPLES:
            print(f"  ← refused too few (< {MIN_SAMPLES}), jailbreak vector skipped")
        else:
            print()

    viable_langs = [l for l in lang_bypassed
                    if len(lang_bypassed[l]) >= MIN_SAMPLES
                    and len(lang_refused[l])  >= MIN_SAMPLES]
    print(f"\nLanguages with sufficient data: {viable_langs}\n")

    # ── Step 2: Load model (once) ────────────────────────────────────────────
    print("Step 2: Loading model...")
    model_base = construct_model_base(args.model_path, lang='en')
    n_layers = model_base.model.config.num_hidden_layers
    print(f"  Loaded. Layers: {n_layers}\n")

    # ── Step 3: Extract jailbreak vectors ────────────────────────────────────
    print("Step 3: Extracting jailbreak vectors...")
    jb_vectors = {}

    for lang in viable_langs:
        print(f"  [{lang}] Extracting jailbreak vector...")
        jb_vec, n_bp, n_ref = extract_jailbreak_vector(
            model_base, lang_bypassed[lang], lang_refused[lang], args.batch_size
        )
        if jb_vec is None:
            print(f"  [{lang}] Skipped.")
            continue

        save_path = os.path.join(args.output_dir, f'jb_vec_{lang}.pt')
        torch.save(jb_vec.cpu(), save_path)
        jb_vectors[lang] = jb_vec

        # Per-layer L2 norm: measures class separability at each layer
        l2_norms = jb_vec.float().norm(dim=-1).tolist()  # [n_layers]
        best_detect_layer = int(torch.tensor(l2_norms).argmax().item())

        results['jailbreak_vectors'][lang] = {
            'n_bypassed':        n_bp,
            'n_refused':         n_ref,
            'saved_to':          save_path,
            'l2_norms':          l2_norms,
            'best_detect_layer': best_detect_layer,
        }
        print(f"  [{lang}] Done. best_detect_layer={best_detect_layer}"
              f" (L2={l2_norms[best_detect_layer]:.3f})  saved to {save_path}")
        torch.cuda.empty_cache()

    # ── Step 4: Extract refusal directions ───────────────────────────────────
    print("\nStep 4: Extracting refusal directions from PolyRefuse train split...")
    refusal_dirs = {}

    for lang in viable_langs:
        print(f"  [{lang}] Extracting refusal direction...")
        try:
            refusal_dir = extract_refusal_direction(
                model_base, lang, args.n_train, args.random_seed, args.batch_size
            )
            save_path = os.path.join(args.output_dir, f'refusal_dir_{lang}.pt')
            torch.save(refusal_dir.cpu(), save_path)
            refusal_dirs[lang] = refusal_dir
            results['refusal_directions'][lang] = {'saved_to': save_path}
            print(f"  [{lang}] Done.")
        except Exception as e:
            print(f"  [{lang}] Failed: {e}")
        torch.cuda.empty_cache()

    # ── Step 4.5: Extract harmfulness directions ─────────────────────────────
    print("\nStep 4.5: Extracting harmfulness directions (bypassed - harmless)...")
    harmfulness_dirs = {}

    for lang in viable_langs:
        if lang not in jb_vectors:
            continue
        print(f"  [{lang}] Extracting harmfulness direction...")
        try:
            harm_dir = extract_harmfulness_direction(
                model_base, lang_bypassed[lang], lang,
                args.n_train, args.random_seed, args.batch_size
            )
            save_path = os.path.join(args.output_dir, f'harmfulness_dir_{lang}.pt')
            torch.save(harm_dir.cpu(), save_path)
            harmfulness_dirs[lang] = harm_dir
            results['harmfulness_directions'][lang] = {'saved_to': save_path}
            print(f"  [{lang}] Done.")
        except Exception as e:
            print(f"  [{lang}] Failed: {e}")
        torch.cuda.empty_cache()

    # ── Step 5: Compute cosine similarities ──────────────────────────────────
    print("\nStep 5: Computing cosine similarities...")
    cos_sims = {}

    for jb_lang, jb_vec in jb_vectors.items():
        for ref_lang, refusal_dir in refusal_dirs.items():
            key = f'{jb_lang}_vs_{ref_lang}'
            sim = compute_cosine_similarity(jb_vec.cpu(), refusal_dir.cpu())
            cos_sims[key] = sim
            avg = sum(sim) / len(sim)
            print(f"  jb={jb_lang} vs refusal={ref_lang}: avg_cos_sim={avg:.3f}")

    results['cosine_similarities'] = cos_sims

    # ── Step 5.5: Three-way geometric analysis ────────────────────────────────
    print("\nStep 5.5: Three-way geometric analysis (jb / refusal / harmfulness)...")
    three_way = {}

    for lang in viable_langs:
        if lang not in jb_vectors or lang not in refusal_dirs or lang not in harmfulness_dirs:
            continue
        jb = jb_vectors[lang].cpu()
        rd = refusal_dirs[lang].cpu()
        hd = harmfulness_dirs[lang].cpu()

        jb_vs_rd = compute_cosine_similarity(jb, rd)
        jb_vs_hd = compute_cosine_similarity(jb, hd)
        rd_vs_hd = compute_cosine_similarity(rd, hd)

        three_way[lang] = {
            'jb_vs_refusal':          jb_vs_rd,
            'jb_vs_harmfulness':      jb_vs_hd,
            'refusal_vs_harmfulness': rd_vs_hd,
        }
        avg_jb_rd = sum(jb_vs_rd) / len(jb_vs_rd)
        avg_jb_hd = sum(jb_vs_hd) / len(jb_vs_hd)
        avg_rd_hd = sum(rd_vs_hd) / len(rd_vs_hd)
        print(f"  [{lang}] jb_vs_refusal={avg_jb_rd:.3f}"
              f"  jb_vs_harmfulness={avg_jb_hd:.3f}"
              f"  refusal_vs_harmfulness={avg_rd_hd:.3f}")

    results['three_way_similarities'] = three_way

    # ── Step 6: Cross-lingual jailbreak vector similarity ────────────────────
    print("\nStep 6: Cross-lingual jailbreak vector similarity matrix...")
    jb_cross = {}
    jb_langs = list(jb_vectors.keys())

    for i, l1 in enumerate(jb_langs):
        for j, l2 in enumerate(jb_langs):
            if i >= j:
                continue
            # Use the middle layer for cross-lingual comparison
            mid = n_layers // 2
            sim_mid = F.cosine_similarity(
                jb_vectors[l1][mid].unsqueeze(0).float(),
                jb_vectors[l2][mid].unsqueeze(0).float()
            ).item()
            # Also compute layer-wise
            sim_all = compute_cosine_similarity(jb_vectors[l1].cpu(), jb_vectors[l2].cpu())
            jb_cross[f'{l1}_vs_{l2}'] = {'mid_layer': sim_mid, 'all_layers': sim_all}
            print(f"  jb[{l1}] vs jb[{l2}]: cos_sim(layer {mid}) = {sim_mid:.3f}")

    results['jailbreak_cross_lingual'] = jb_cross

    # ── Save all results ─────────────────────────────────────────────────────
    out_json = os.path.join(args.output_dir, 'geometry_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',   type=str, required=True,
                        help='HuggingFace model path or local path')
    parser.add_argument('--model_alias',  type=str, default=None,
                        help='Short name used in directory structure (default: basename of model_path)')
    parser.add_argument('--baseline_dir', type=str,
                        default='output/ja_vector_sweep',
                        help='Root dir containing baseline evaluations')
    parser.add_argument('--exp_id',       type=str,
                        default='20250519-232436/1',
                        help='Experiment timestamp/id subdirectory')
    parser.add_argument('--output_dir',   type=str,
                        default='output/jailbreak_analysis/Qwen2.5-7B-Instruct',
                        help='Where to save vectors and results')
    parser.add_argument('--n_train',      type=int, default=128,
                        help='Number of train samples for refusal direction')
    parser.add_argument('--random_seed',  type=int, default=1)
    parser.add_argument('--batch_size',   type=int, default=8)
    args = parser.parse_args()

    if args.model_alias is None:
        args.model_alias = os.path.basename(args.model_path)

    main(args)
