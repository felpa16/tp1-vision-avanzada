# Continual Learning — Results Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the backbone, run all four CL methods (Fine-Tune, EWC, LwF, Co2L), and produce the comparison plots and table required by section 4.4 of the assignment.

**Architecture:** A single `compare_methods.py` script at the project root orchestrates the full pipeline: it trains all four methods using their existing `train_*` functions, serialises the results to `graphs/results.json` so they can be reloaded without re-running, then generates two comparison figures and prints a unified table.

**Tech Stack:** Python 3, PyTorch, Matplotlib, existing `models/{finetune,ewc,lwf,co2l}.py`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Run (existing) | `models/train_backbone.py` | Train ResNet-18 backbone with SupCon loss, save `backbone.pth` |
| Create | `compare_methods.py` | Orchestrate all methods, save `graphs/results.json`, generate plots |
| Output | `graphs/results.json` | Serialised per-method `{class_il, task_il}` lists (5 floats each) |
| Output | `graphs/accuracy_curves.png` | Class-IL and Task-IL curves for all methods |
| Output | `graphs/forgetting_curves.png` | Class-IL degradation per method (forgetting proxy) |

All scripts must be run from the **project root** (`tp1-vision-avanzada/`).

---

## Task 1: Train the Backbone and Save `backbone.pth`

**Files:**
- Run: `models/train_backbone.py` (no changes needed)
- Output: `backbone.pth` in project root

- [ ] **Step 1: Verify CIFAR-10 data is present**

```bash
ls data/cifar-10-batches-py/
```

Expected output: `batches.meta  data_batch_1  data_batch_2  data_batch_3  data_batch_4  data_batch_5  readme.html  test_batch`

- [ ] **Step 2: Run backbone training**

```bash
cd /home/tron-mrs/Documents/2026/Vision\ Artificial\ Avanzada/tp1-vision-avanzada
python -m models.train_backbone
```

Expected: 25 epochs of SupCon loss printed, three t-SNE snapshots saved, then:
```
Backbone saved to backbone.pth
Figure saved to latent_space_snapshots.png
```

- [ ] **Step 3: Verify checkpoint exists**

```bash
ls -lh backbone.pth
```

Expected: file present, size ~40–50 MB.

- [ ] **Step 4: Commit the checkpoint** *(optional — .pth files are large; skip if gitignored)*

```bash
git add backbone.pth
git commit -m "chore: save pre-trained backbone checkpoint"
```

---

## Task 2: Create `compare_methods.py`

**Files:**
- Create: `compare_methods.py` (project root)

- [ ] **Step 1: Create `compare_methods.py` with the full implementation**

```python
"""compare_methods.py

Run all four Continual Learning methods on Seq-CIFAR-10 sequentially and
produce the comparison artefacts required by section 4.4 of the assignment:

  1. graphs/results.json      — per-method Class-IL and Task-IL accuracy after each task
  2. graphs/accuracy_curves.png — Class-IL and Task-IL accuracy curves (all methods)
  3. graphs/forgetting_curves.png — Class-IL degradation as a forgetting proxy

Run from the project root:
    python compare_methods.py [--skip-training]

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
matplotlib.use('Agg')          # headless-safe; remove if running in a notebook
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
            "Run `python -m models.train_backbone` first."
        )

    results = {}

    for label, fn in [
        ('Fine-Tune', train_finetune),
        ('EWC',       train_ewc),
        ('LwF',       train_lwf),
        ('Co2L',      train_co2l),
    ]:
        print(f"\n{'='*70}")
        print(f"Running: {label}")
        print('='*70)
        results[label] = fn(backbone_weights=BACKBONE_WEIGHTS)

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
    """Single-panel figure showing Class-IL accuracy degradation as a forgetting proxy.

    For each method, Class-IL accuracy after each task reflects how well the model
    retains knowledge of all classes seen so far. Methods that forget more will show
    a steeper drop from task 1 onwards.
    """
    tasks = list(range(1, 6))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        'Forgetting — Class-IL Accuracy Over Tasks\n'
        '(drop from T1 peak reflects catastrophic forgetting)',
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
```

- [ ] **Step 2: Verify the file was created**

```bash
python -c "import ast; ast.parse(open('compare_methods.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
git add compare_methods.py docs/superpowers/plans/2026-04-18-cl-comparison.md
git commit -m "feat: add compare_methods.py for section 4.4 comparison"
```

---

## Task 3: Run the Full Comparison and Verify Outputs

**Files:**
- Run: `compare_methods.py`
- Outputs: `graphs/results.json`, `graphs/accuracy_curves.png`, `graphs/forgetting_curves.png`

- [ ] **Step 1: Run the full pipeline**

```bash
python compare_methods.py
```

This takes ~15–40 minutes depending on whether a GPU is available (5 epochs × 4 methods × 5 tasks).
Expected final output:
```
Done. All outputs in graphs/
```

- [ ] **Step 2: Verify all outputs were produced**

```bash
ls -lh graphs/results.json graphs/accuracy_curves.png graphs/forgetting_curves.png
```

Expected: all three files present with non-zero sizes.

- [ ] **Step 3: Verify results.json is well-formed and has all methods**

```bash
python -c "
import json
r = json.load(open('graphs/results.json'))
for m in ['Fine-Tune','EWC','LwF','Co2L']:
    assert m in r, f'Missing method: {m}'
    assert len(r[m]['class_il']) == 5, f'Wrong length for {m} class_il'
    assert len(r[m]['task_il'])  == 5, f'Wrong length for {m} task_il'
print('results.json OK — all 4 methods with 5 task entries each')
"
```

Expected: `results.json OK — all 4 methods with 5 task entries each`

- [ ] **Step 4: Commit outputs**

```bash
git add graphs/results.json graphs/accuracy_curves.png graphs/forgetting_curves.png
git commit -m "results: add CL comparison plots and results table"
```

---

## Spec Coverage Check

| Assignment Requirement (§4.4) | Covered By |
|-------------------------------|------------|
| Unified Class-IL / Task-IL table for all methods | `print_table()` in Task 3 |
| Accuracy curves (Class-IL and Task-IL) per task, all methods same plot | `plot_accuracy_curves()` — `graphs/accuracy_curves.png` |
| Forgetting curve (optional) | `plot_forgetting_curves()` — `graphs/forgetting_curves.png` |
| All 4 CL methods run end-to-end | `run_all_methods()` calls all four `train_*` functions |
| Backbone checkpoint available | Task 1 re-trains and saves `backbone.pth` |
