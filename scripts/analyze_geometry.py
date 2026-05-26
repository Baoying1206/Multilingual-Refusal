"""
Generate all thesis figures from geometry_results.json files.

Figures produced:
  Figure 1 (RQ1): Heatmap of avg cosine similarity — jailbreak lang × refusal lang, per model
  Figure 2 (RQ1): Layer-wise cosine similarity profiles for same-language pairs
  Figure 3 (RQ2-geo): Cross-lingual jailbreak vector similarity heatmap, per model
  Figure 4:       Three-model comparison — diagonal cosine similarity by language
  Figure 5:       Yoruba anomaly highlight
  Figure 6:       L2 norm profiles per layer
  Figure 7 (RQ1): Three-way geometric relationship
  Figure 8 (RQ2-beh): Cross-lingual transfer bypass rate matrix

Usage:
  python scripts/analyze_geometry.py \
      --results_dir  output/jailbreak_analysis \
      --transfer_dir output/transfer \
      --output_dir   output/figures
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

    fig.suptitle('RQ2 (Geometric): Cross-lingual jailbreak vector similarity',
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


# ── Figure 6: L2 norm of jailbreak vector per layer ──────────────────────────

def plot_l2_norm_profiles(data, output_dir):
    """Layer-wise L2 norm of jailbreak vector for common languages across models."""
    # Collect languages that have l2_norms in at least one model
    lang_sets = []
    for model in MODELS:
        if model not in data:
            continue
        jb_vecs = data[model].get('jailbreak_vectors', {})
        langs_with_norms = {l for l, v in jb_vecs.items() if 'l2_norms' in v}
        if langs_with_norms:
            lang_sets.append(langs_with_norms)

    if not lang_sets:
        print('  Skipping fig6: no l2_norms found (re-run extract_jailbreak_vectors.py)')
        return

    common = sorted(lang_sets[0].intersection(*lang_sets[1:]) if len(lang_sets) > 1 else lang_sets[0],
                    key=lambda l: ALL_LANGS.index(l) if l in ALL_LANGS else 99)
    common = [l for l in common if l != 'yo'][:6]

    if not common:
        print('  Skipping fig6: no common languages with l2_norms.')
        return

    n_cols = min(3, len(common))
    n_rows = (len(common) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, lang in zip(axes, common):
        for model in MODELS:
            if model not in data:
                continue
            jb_info = data[model].get('jailbreak_vectors', {}).get(lang, {})
            norms = jb_info.get('l2_norms')
            if norms is None:
                continue
            layers = list(range(len(norms)))
            best_l = jb_info.get('best_detect_layer')
            ax.plot(layers, norms,
                    label=MODEL_LABELS[model],
                    color=COLORS[model], linewidth=1.8)
            if best_l is not None:
                ax.axvline(best_l, color=COLORS[model],
                           linewidth=1, linestyle=':', alpha=0.7)

        ax.set_title(f'lang={lang}', fontsize=10)
        ax.set_xlabel('Layer', fontsize=9)
        ax.set_ylabel('L2 norm', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(common):]:
        ax.set_visible(False)

    fig.suptitle('Jailbreak direction strength: layer-wise L2 norm\n'
                 '(dotted line = best detection layer per model)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig6_l2_norm_profiles.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 7: Three-way geometric relationship ───────────────────────────────

def plot_three_way(data, output_dir):
    """
    Layer-wise cosine similarity for all three direction pairs per language:
      jb_vec vs refusal_dir | jb_vec vs harmfulness_dir | refusal_dir vs harmfulness_dir
    One subplot per language, lines coloured by pair type, one panel per model.
    """
    PAIR_COLORS = {
        'jb_vs_refusal':          '#E53935',
        'jb_vs_harmfulness':      '#8E24AA',
        'refusal_vs_harmfulness': '#1E88E5',
    }
    PAIR_LABELS = {
        'jb_vs_refusal':          'JB vs Refusal',
        'jb_vs_harmfulness':      'JB vs Harmfulness',
        'refusal_vs_harmfulness': 'Refusal vs Harmfulness',
    }

    # Collect languages that have three_way data in any model
    all_langs = set()
    for model in MODELS:
        if model not in data:
            continue
        for lang in data[model].get('three_way_similarities', {}):
            all_langs.add(lang)

    if not all_langs:
        print('  Skipping fig7: no three_way_similarities found.')
        return

    langs = sorted(all_langs, key=lambda l: ALL_LANGS.index(l) if l in ALL_LANGS else 99)
    langs = [l for l in langs if l != 'yo'][:6]

    available = [m for m in MODELS if m in data]
    n_cols = len(langs)
    n_rows = len(available)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.2 * n_rows),
                             sharey=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for r, model in enumerate(available):
        three_way = data[model].get('three_way_similarities', {})
        for c, lang in enumerate(langs):
            ax = axes[r][c]
            if lang not in three_way:
                ax.set_visible(False)
                continue
            tw = three_way[lang]
            for pair, color in PAIR_COLORS.items():
                sims = tw.get(pair)
                if sims is None:
                    continue
                ax.plot(range(len(sims)), sims,
                        color=color, linewidth=1.6,
                        label=PAIR_LABELS[pair])
            ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax.set_ylim(-1.0, 1.0)
            ax.grid(True, alpha=0.3)
            if r == 0:
                ax.set_title(lang, fontsize=10, fontweight='bold')
            if c == 0:
                ax.set_ylabel(MODEL_LABELS[model], fontsize=9)
            if r == n_rows - 1:
                ax.set_xlabel('Layer', fontsize=8)
            if r == 0 and c == n_cols - 1:
                ax.legend(fontsize=7, loc='lower right')

    fig.suptitle('Three-way geometric relationship: JB vector / Refusal direction / Harmfulness direction',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig7_three_way_geometry.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.savefig(path.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print(f'  Saved: {path}')


# ── Figure 8: RQ2 behavioural transfer results ────────────────────────────────

def plot_transfer_results(transfer_dir, output_dir):
    """
    Grouped bar chart: bypass rate by target language and source jailbreak vector.
    One subplot per model. Baseline shown as a dashed horizontal line per language.
    Only models for which transfer_results.json exists are plotted.
    """
    TARGET_ORDER = ['en', 'de', 'es', 'fr', 'it']
    SOURCE_COLORS = {'yo': '#E53935', 'ko': '#FB8C00', 'ja': '#8E24AA'}
    SOURCE_LABELS = {'yo': 'Yoruba (yo)', 'ko': 'Korean (ko)', 'ja': 'Japanese (ja)'}

    # Load transfer data
    transfer_data = {}
    for model in MODELS:
        path = os.path.join(transfer_dir, model, 'transfer_results.json')
        if os.path.exists(path):
            with open(path) as f:
                transfer_data[model] = json.load(f)
            print(f'  Loaded transfer: {model}')
        else:
            print(f'  Missing transfer: {path}')

    if not transfer_data:
        print('  Skipping fig8: no transfer_results.json found.')
        return

    available = [m for m in MODELS if m in transfer_data]
    fig, axes = plt.subplots(1, len(available),
                             figsize=(6 * len(available), 4.5),
                             sharey=False)
    if len(available) == 1:
        axes = [axes]

    for ax, model in zip(axes, available):
        res = transfer_data[model]['results']
        src_langs = transfer_data[model]['source_langs']
        alpha     = transfer_data[model]['alpha']
        k_star    = transfer_data[model]['k_star']

        # Collect data for each target language (only those present in results)
        tgt_langs = [l for l in TARGET_ORDER if l in res]
        x = np.arange(len(tgt_langs))
        n_src = len(src_langs)
        width = 0.18
        offsets = np.linspace(-(n_src - 1) / 2, (n_src - 1) / 2, n_src) * width

        for k, src in enumerate(src_langs):
            bypass_vals = [res[tgt]['transfer'].get(src, np.nan) for tgt in tgt_langs]
            bars = ax.bar(x + offsets[k], bypass_vals, width,
                          label=SOURCE_LABELS.get(src, src),
                          color=SOURCE_COLORS.get(src, '#607D8B'),
                          alpha=0.85, edgecolor='white')

        # Baseline as scatter points
        baselines = [res[tgt]['baseline_bypass'] for tgt in tgt_langs]
        ax.scatter(x, baselines, marker='D', s=40, color='black',
                   zorder=5, label='Baseline')

        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(tgt_langs, fontsize=10)
        ax.set_xlabel('Target language', fontsize=10)
        ax.set_ylabel('Bypass rate', fontsize=10)
        ax.set_title(f'{MODEL_LABELS[model]}\n'
                     f'$\\alpha$={alpha}, $k^*$={k_star}',
                     fontsize=10, fontweight='bold')
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle('RQ2 (Behavioural): Cross-lingual jailbreak transfer — bypass rates',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'fig8_rq2_transfer_bypass.pdf')
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
    plot_l2_norm_profiles(data, args.output_dir)
    plot_three_way(data, args.output_dir)
    plot_transfer_results(args.transfer_dir, args.output_dir)

    print(f'\nDone. All figures saved to {args.output_dir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str,
                        default='output/jailbreak_analysis',
                        help='Directory containing per-model geometry_results.json files')
    parser.add_argument('--transfer_dir', type=str,
                        default='output/transfer',
                        help='Directory containing per-model transfer_results.json files')
    parser.add_argument('--output_dir', type=str,
                        default='output/figures',
                        help='Where to save figures')
    args = parser.parse_args()
    main(args)
