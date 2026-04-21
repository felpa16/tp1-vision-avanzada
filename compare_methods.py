"""compare_methods.py

Run all four Continual Learning methods on Seq-CIFAR-10 sequentially and
produce the comparison artefacts required by section 4.4 of the assignment:

  1. graphs/results.json          — per-method Class-IL and Task-IL accuracy after each task
  2. graphs/accuracy_curves.png   — Class-IL and Task-IL accuracy curves (all methods)
  3. graphs/forgetting_curves.png — Class-IL degradation as a forgetting proxy

Run from the project root:
    python3 compare_methods.py [--skip-training]

Flags
-----
  --skip-training   Load existing graphs/results.json instead of re-running methods.
                    Useful for regenerating plots after a completed run.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')          # headless-safe; no display required
import matplotlib.pyplot as plt

from models.finetune import train_finetune
from models.ewc      import train_ewc
from models.lwf      import train_lwf
from models.co2l     import train_co2l

# ── Constants ─────────────────────────────────────────────────────────────────
BACKBONE_WEIGHTS = 'backbone.pth'
GRAPHS_DIR       = 'graphs'
RESULTS_PATH     = os.path.join(GRAPHS_DIR, 'results.json')

METHODS = ['Fine-Tune', 'EWC', 'LwF', 'Co2L']
COLORS  = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green']
TASK_LABELS = [f'T{i}' for i in range(1, 6)]


# ── Training ──────────────────────────────────────────────────────────────────
def run_all_methods() -> dict:
    """Run all four CL methods and return a dict of results."""
    if not os.path.exists(BACKBONE_WEIGHTS):
        sys.exit(
            f"ERROR: '{BACKBONE_WEIGHTS}' not found.\n"
            "Run `python3 -m models.train_backbone` first."
        )

    results = {}

    CL_EPOCHS  = 10
    CL_LR      = 1e-3
    HEAD_EPOCHS = 10
    HEAD_LR    = 8e-4
    BATCH_SIZE = 64

    runs = [
        ('Fine-Tune', train_finetune, dict(
            num_epochs=CL_EPOCHS, lr=CL_LR, batch_size=BATCH_SIZE,
            head_epochs=HEAD_EPOCHS, head_lr=HEAD_LR,
        )),
        ('EWC', train_ewc, dict(
            lambda_ewc=5000.0, fisher_samples=500,
            num_epochs=CL_EPOCHS, lr=CL_LR, batch_size=BATCH_SIZE,
            head_epochs=HEAD_EPOCHS, head_lr=HEAD_LR,
        )),
        ('LwF', train_lwf, dict(
            lambda_lwf=1.0, temperature=2.0,
            num_epochs=CL_EPOCHS, lr=CL_LR, batch_size=BATCH_SIZE,
            head_epochs=HEAD_EPOCHS, head_lr=HEAD_LR,
        )),
        ('Co2L', train_co2l, dict(
            lambda_co2l=1.0, buffer_size=200,
            num_epochs=CL_EPOCHS, lr=CL_LR, batch_size=BATCH_SIZE,
            head_epochs=HEAD_EPOCHS, head_lr=HEAD_LR,
        )),
    ]

    for label, fn, kwargs in runs:
        print(f"\n{'='*70}")
        print(f"Running: {label}")
        print('='*70)
        results[label] = fn(backbone_weights=BACKBONE_WEIGHTS, **kwargs)

    return results


# ── Persistence ───────────────────────────────────────────────────────────────
def save_results(results: dict) -> None:
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


def load_results() -> dict:
    if not os.path.exists(RESULTS_PATH):
        sys.exit(f"ERROR: '{RESULTS_PATH}' not found. Run without --skip-training first.")
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ── Table ─────────────────────────────────────────────────────────────────────
def print_table(results: dict) -> None:
    col_w = 12
    task_nums = list(range(1, 6))

    header = f"{'Method':<{col_w}}" + "".join(
        f"  {'T'+str(t)+' CIL':>8}  {'T'+str(t)+' TIL':>8}" for t in task_nums
    )
    sep = '=' * len(header)

    print(f"\n{sep}")
    print("RESULTS — Class-IL (CIL) and Task-IL (TIL) accuracy (%) after each task")
    print(sep)
    print(header)
    print('-' * len(header))

    for method in METHODS:
        if method not in results:
            continue
        r = results[method]
        row = f"{method:<{col_w}}"
        for i in range(5):
            cil = r['class_il'][i]
            til = r['task_il'][i]
            row += f"  {cil:>8.2f}  {til:>8.2f}"
        print(row)

    print(sep)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_accuracy_curves(results: dict) -> None:
    """Two-panel figure: Class-IL (left) and Task-IL (right) accuracy vs tasks."""
    tasks = list(range(1, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Continual Learning on Seq-CIFAR-10', fontsize=14, fontweight='bold')

    for method, color in zip(METHODS, COLORS):
        if method not in results:
            continue
        r = results[method]
        ax1.plot(tasks, r['class_il'], marker='o', label=method,
                 color=color, linewidth=2, markersize=6)
        ax2.plot(tasks, r['task_il'],  marker='o', label=method,
                 color=color, linewidth=2, markersize=6)

    for ax, title in [
        (ax1, 'Class-Incremental Learning (Class-IL)'),
        (ax2, 'Task-Incremental Learning (Task-IL)'),
    ]:
        ax.set_xlabel('Tasks Learned', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(tasks)
        ax.set_xticklabels(TASK_LABELS)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, 'accuracy_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_forgetting_curves(results: dict) -> None:
    """Class-IL accuracy over tasks — a proxy for catastrophic forgetting.

    For each method, the Class-IL accuracy after each task shows how well the
    model retains knowledge of all classes seen so far. A steeper drop indicates
    more catastrophic forgetting.
    """
    tasks = list(range(1, 6))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        'Forgetting — Class-IL Accuracy Over Tasks\n',
        fontsize=12,
    )

    for method, color in zip(METHODS, COLORS):
        if method not in results:
            continue
        r = results[method]
        ax.plot(tasks, r['class_il'], marker='o', label=method,
                color=color, linewidth=2, markersize=6)

    ax.set_xlabel('Tasks Learned', fontsize=11)
    ax.set_ylabel('Class-IL Accuracy (%)', fontsize=11)
    ax.set_xticks(tasks)
    ax.set_xticklabels(TASK_LABELS)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(GRAPHS_DIR, 'forgetting_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Compare CL methods on Seq-CIFAR-10')
    parser.add_argument('--skip-training', action='store_true',
                        help='Load existing results.json instead of re-running methods')
    args = parser.parse_args()

    os.makedirs(GRAPHS_DIR, exist_ok=True)

    if args.skip_training:
        print(f"Loading results from {RESULTS_PATH} …")
        results = load_results()
    else:
        results = run_all_methods()
        save_results(results)

    print_table(results)
    plot_accuracy_curves(results)
    plot_forgetting_curves(results)
    print("\nDone. All outputs in graphs/")


if __name__ == '__main__':
    main()
