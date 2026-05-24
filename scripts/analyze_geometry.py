"""
Generate all thesis figures from geometry_results.json files.

Figures produced:
  Figure 1 (RQ1): Heatmap of avg cosine similarity — jailbreak lang × refusal lang, per model
  Figure 2 (RQ1): Layer-wise cosine similarity profiles for same-language pairs
  Figure 3 (RQ2): Cross-lingual jailbreak vector similarity heatmap, per model
  Figure 4:       Three-model comparison — diagonal cosine similarity by language

Usage:
  python scripts/analyze_geometry.py \
      --results_dir output/jailbreak_analysis \
      --output_dir  output/figures
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Language ordering ─────────────────────────────────────────────────────────
ALL_LANGS = ['en', 'de', 'ja', 'ko', 'zh', 'ru', 'th', 'yo', 'ar', 'es', 'fr', 'it', 'nl', 'pl']

MODELS = [
    'Qwen2.5-7B-Instruct',
    'Meta-Llama-3.1-8B-Instruct',
    'gemma-2-9b-it',
]

MODEL_LABELS = {
    'Qwen2.5-7B-Instruct':          'Qwen2.5-7B',
    'Meta-Llama-3.1-8B-Instruct':   'LLaMA-3.1-8B',
    'gemma-2-9b-it':                'Gemma-2-9B',
}

COLORS = {
    'Qwen2.5-7B-Instruct':          '#2196F3',
    'Meta-Llama-3.1-8B-Instruct':   '#FF5722',
    'gemma-2-9b-it':                '#4CAF50',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_results(results_dir):
    data = {}
    for model in MODELS:
        path = os.path.join(results_dir, model, 'geometry_results.json')
        if os.path.exists(path):
            with open(path) as f:
                data[model] = json.load(f)
            print(f'  Loaded: {model}')
        else:
            print(f'  Missing: {path}')
    return data


def get_langs(result):
    """Return sorted list of languages present in cosine_similarities."""
    keys = result['cosine_similarities'].keys()
    jb_langs = sorted(set(k.split('_vs_')[0] for k in keys),
                      key=lambda l: ALL_LANGS.index(l) if l in ALL_LANGS else 99)
    return jb_langs


def build_avg_matrix(result, jb_langs, ref_langs):
    """Build matrix[i,j] = avg cosine sim for jb_langs[i] vs ref_langs[j]."""
    cos_sims = result['cosine_similarities']
    mat = np.full((len(jb_langs), len(ref_langs)), np.nan)
    for i, jl in enumerate(jb_langs):
        for j, rl in enumerate(ref_langs):
            key = f'{jl}_vs_{rl}'
            if key in cos_sims:
                mat[i, j] = float(np.mean(cos_sims[key]))
    return mat


def build_cross_matrix(result, langs):
    """Build symmetric matrix of cross-lingual jailbreak vector similarities."""
    cross = result.get('jailbreak_cross_lingual', {})
    n = len(langs)
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 1.0)
    for i, l1 in enumerate(langs):
        for j, l2 in enumerate(langs):
            if i == j:
                continue
            key1 = f'{l1}_vs_{l2}'
            key2 = f'{l2}_vs_{l1}'
            val = None
            if key1 in cross:
                val = cross[key1]['mid_layer']
            elif key2 in cross:
                val = cross[key2]['mid_layer']
            if val is not None:
                mat[i, j] = val
                mat[j, i] = val
    return mat


# ── Figure 1: RQ1 heatmaps ────────────────────────────────────────────────────

def plot_rq1_heatmaps(data, output_dir):
    """One heatmap per model: jailbreak lang × refusal lang, avg cosine sim."""
    available = [m for m in MODELS if m in data]
    fig, axes = plt.subplots(1, len(available),
                             figsize=(6.5 * len(available), 5.5))
    if len(available) == 1:
        axes = [axes]

    vmin, vmax = -0.75, 0.20

    for ax, model in zip(axes, available):
        result  = data[model]
        langs   = get_langs(result)
        mat     = build_avg_matrix(result, langs, langs)

        im = ax.imshow(mat, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(len(langs)))
        ax.set_yticks(range(len(langs)))
        ax.set_xticklabels(langs, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(langs, fontsize=9)
        ax.set_xlabel('Refusal direction language', fontsize=10)
        ax.set_ylabel('Jailbreak vector language', fontsize=10)
        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight='bold')

        # Annotate cells
        for i in range(len(langs)):
            for j in range(len(langs)):
                val = mat[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 0.4 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=6.5, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='Avg cosine similarity')

    fig.suptitle('RQ1: Cosine similarity between jailbreak vectors and refusal directions',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_rq1_heatmaps.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 2: Layer-wise profiles ─────────────────────────────────────────────

def plot_layer_profiles(data, output_dir):
    """Layer-wise cosine similarity for same-language pairs across models."""
    # Pick languages that appear in all models
    lang_sets = [set(get_langs(data[m])) for m in MODELS if m in data]
    common = sorted(lang_sets[0].intersection(*lang_sets[1:]),
                    key=lambda l: ALL_LANGS.index(l) if l in ALL_LANGS else 99)
    common = [l for l in common if l != 'yo'][:6]  # top 6 non-yo languages

    n_cols = min(3, len(common))
    n_rows = (len(common) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, lang in zip(axes, common):
        for model in MODELS:
            if model not in data:
                continue
            key = f'{lang}_vs_{lang}'
            sims = data[model]['cosine_similarities'].get(key)
            if sims is None:
                continue
            layers = list(range(len(sims)))
            ax.plot(layers, sims,
                    label=MODEL_LABELS[model],
                    color=COLORS[model], linewidth=1.8)

        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
        ax.set_title(f'jb={lang} vs refusal={lang}', fontsize=10)
        ax.set_xlabel('Layer', fontsize=9)
        ax.set_ylabel('Cosine similarity', fontsize=9)
        ax.set_ylim(-1.0, 0.5)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes[len(common):]:
        ax.set_visible(False)

    fig.suptitle('RQ1: Layer-wise cosine similarity (same-language pairs)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_rq1_layer_profiles.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 3: RQ2 cross-lingual heatmaps ──────────────────────────────────────

def plot_rq2_heatmaps(data, output_dir):
    """Cross-lingual jailbreak vector similarity at middle layer, per model."""
    available = [m for m in MODELS if m in data]
    fig, axes = plt.subplots(1, len(available),
                             figsize=(5.5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, model in zip(axes, available):
        result = data[model]
        langs  = get_langs(result)
        mat    = build_cross_matrix(result, langs)

        im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=1.0, aspect='auto')
        ax.set_xticks(range(len(langs)))
        ax.set_yticks(range(len(langs)))
        ax.set_xticklabels(langs, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(langs, fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontsize=11, fontweight='bold')

        for i in range(len(langs)):
            for j in range(len(langs)):
                val = mat[i, j]
                if not np.isnan(val):
                    color = 'white' if val > 0.7 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=6.5, color=color)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='Cosine similarity (mid layer)')

    fig.suptitle('RQ2: Cross-lingual jailbreak vector similarity',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_rq2_cross_lingual.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 4: Three-model diagonal comparison ─────────────────────────────────

def plot_diagonal_comparison(data, output_dir):
    """Bar chart: same-language cosine similarity by language, grouped by model."""
    # Collect diagonal values
    records = {}
    for model in MODELS:
        if model not in data:
            continue
        result = data[model]
        langs  = get_langs(result)
        for lang in langs:
            key = f'{lang}_vs_{lang}'
            sims = result['cosine_similarities'].get(key)
            if sims is not None:
                records.setdefault(lang, {})[model] = float(np.mean(sims))

    # Sort languages by ALL_LANGS order
    langs = sorted(records.keys(),
                   key=lambda l: ALL_LANGS.index(l) if l in ALL_LANGS else 99)

    x      = np.arange(len(langs))
    n_mod  = sum(1 for m in MODELS if m in data)
    width  = 0.25
    offset = np.linspace(-(n_mod-1)/2, (n_mod-1)/2, n_mod) * width

    fig, ax = plt.subplots(figsize=(13, 4.5))

    for k, model in enumerate([m for m in MODELS if m in data]):
        vals = [records[l].get(model, np.nan) for l in langs]
        ax.bar(x + offset[k], vals, width,
               label=MODEL_LABELS[model],
               color=COLORS[model], alpha=0.85, edgecolor='white')

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=10)
    ax.set_ylabel('Avg cosine similarity', fontsize=11)
    ax.set_title('Same-language cosine similarity (jailbreak vector vs refusal direction) — three models',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(-0.85, 0.25)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig4_diagonal_comparison.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 5: Yoruba anomaly highlight ───────────────────────────────────────

def plot_yoruba_anomaly(data, output_dir):
    """Layer-wise cosine sim for yo vs yo across models, vs a 'normal' language."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, lang in zip(axes, ['ko', 'yo']):
        for model in MODELS:
            if model not in data:
                continue
            key  = f'{lang}_vs_{lang}'
            sims = data[model]['cosine_similarities'].get(key)
            if sims is None:
                continue
            layers = list(range(len(sims)))
            ax.plot(layers, sims,
                    label=MODEL_LABELS[model],
                    color=COLORS[model], linewidth=2)

        ax.axhline(0, color='gray', linewidth=1, linestyle='--')
        title = f'Korean (ko) — well-aligned' if lang == 'ko' else f'Yoruba (yo) — anomaly'
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('Cosine similarity', fontsize=10)
        ax.set_ylim(-1.0, 0.5)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('RQ1: Yoruba anomaly — layer-wise profiles vs Korean',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig5_yoruba_anomaly.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading results...')
    data = load_results(args.results_dir)

    if not data:
        print('No results found. Check --results_dir.')
        return

    print('\nGenerating figures...')
    plot_rq1_heatmaps(data, args.output_dir)
    plot_layer_profiles(data, args.output_dir)
    plot_rq2_heatmaps(data, args.output_dir)
    plot_diagonal_comparison(data, args.output_dir)
    plot_yoruba_anomaly(data, args.output_dir)

    print(f'\nDone. All figures saved to {args.output_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str,
                        default='output/jailbreak_analysis',
                        help='Directory containing per-model geometry_results.json files')
    parser.add_argument('--output_dir', type=str,
                        default='output/figures',
                        help='Where to save figures')
    args = parser.parse_args()
    main(args)
