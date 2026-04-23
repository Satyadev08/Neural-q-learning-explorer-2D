# Neural Q-Learning Robot Explorer


> Capstone project — Master of Science in Robotics, University of California Riverside, April 2026.

---

## Results Summary

| Variant | Environment | Coverage | Success Rate | Avg Steps | A\*/Episode | Rating |
|---|---|---|---|---|---|---|
| **Nearest Cell** | 20×20 grid | **92.40% ±0.36%** | **100%** | 132.8 | 1.52 | ✅ EXCELLENT |
| **Cluster A\*** | 20×20 grid | 92.36% ±0.34% | **100%** | 150.1 | 1.27 | ✅ EXCELLENT |
| **Periodic A\*** | 20×20 grid | 92.39% ±0.48% | 99.5% | 361.0 | 11.23 | ✅ EXCELLENT |

---

## Overview

This repo contains three **2D exploration variants** that share the same CNN Q-Network architecture, environment, and training loop — differing only in how and when A\* pathfinding guidance is invoked.

### Variant Comparison

```
nearest_cell  →  cluster_astar  →  periodic_astar
    │                │                  │
    │                │                  └─ A* fires every 25 steps proactively
    │                │                     (74% proactive / 26% reactive split)
    │                └─ Flood-fill to target largest unexplored cluster
    │                   avg 76.6-cell clusters, max 278-cell cluster found
    └─ BFS to nearest unvisited cell on emergency only
       43% of episodes completed with zero A* intervention
```

---

## Detailed Results

### Nearest Cell (Base Variant) — 200 Episodes

The reactive baseline with minimal intervention. A\* fires only when stuck detection triggers, navigating to the nearest unexplored cell within a 20-cell search radius.

| Scenario | Coverage | Success | Avg Steps |
|---|---|---|---|
| Standard obstacles | 92.35% ±0.31% | 100% | 127.0 |
| Dense obstacles | 92.48% ±0.35% | 100% | 169.8 |
| Sparse obstacles | 92.48% ±0.36% | 100% | 98.7 |
| Large block obstacles | 92.28% ±0.39% | 100% | 135.5 |
| **Overall** | **92.40% ±0.36%** | **100%** | **132.8** |

**A\* usage:** 305 total escapes across 200 episodes (1.52/ep) — 43% of episodes required zero intervention.

---

### Cluster A\* Variant — 200 Episodes

Replaces nearest-cell targeting with intelligent flood-fill analysis. Scores candidate regions by `cluster_size × 100 − distance`, heavily prioritizing large open areas.

| Metric | Value |
|---|---|
| Average Coverage | 92.36% ±0.34% |
| Success Rate | 100% |
| Average Steps | 150.1 ±99.9 |
| Average Reward | 3,140.7 |
| Total A\* escapes | 254 (100% cluster-based, no nearest-cell fallback) |
| Episodes using A\* | 125/200 (62.5%) |
| Average cluster size targeted | 76.6 cells (19% of grid) |
| Largest cluster found | **278 cells (69.5% of grid)** |

Only 13% more steps than the base variant for the strategic targeting benefits.

---

### Periodic A\* Variant — 200 Episodes

Implements a dual A\* strategy: scheduled guidance every 25 steps (proactive) combined with stuck-based emergency escapes (reactive). Uses a 6-channel state where Channel 5 encodes the A\* guidance path directly into the state tensor.

| Metric | Value |
|---|---|
| Average Coverage | 92.39% ±0.48% |
| Success Rate | 99.5% (1 failure in dense scenario) |
| Average Steps | 361.0 ±164.4 |
| Average Reward | 360.3 |

**Dual A\* Breakdown:**

| Type | Total | Episodes | Avg/Ep | Share |
|---|---|---|---|---|
| Periodic (scheduled) | 1,662 | 200/200 (100%) | 8.31 | 74% |
| Emergency (stuck) | 584 | 174/200 (87%) | 2.92 | 26% |
| **Combined** | **2,246** | **200/200** | **11.23** | 100% |

---

## Architecture

### 2D CNN Q-Network (~3.3M parameters)

```
Input: 5-channel 20×20 grid (6-channel for periodic_astar)
  Ch 0: Obstacles
  Ch 1: FOV-visited cells
  Ch 2: Agent position
  Ch 3: Convoy robot positions
  Ch 4: Current Field of View
  Ch 5: A* guidance signal  ← periodic_astar only

CNN:
  Conv1: 5→16 filters (3×3, pad=1) → ReLU   [detects local wall/obstacle patterns]
  Conv2: 16→32 filters (3×3, pad=1) → ReLU  [corridors, open regions, frontiers]
  Flatten → 12,800

FC:
  FC1: 12,800 → 256 → ReLU
  FC2: 256 → 128 → ReLU
  FC3: 128 → 9   (Q-values for each action)
```

### Training Configuration

| Parameter | Value |
|---|---|
| Episodes | 2,500 |
| Max steps/episode | 1,000 |
| Discount factor γ | 0.98 |
| Learning rate | 0.0005 |
| Optimizer | Adam |
| ε start / end / decay | 1.0 / 0.01 / 0.995 |
| Gradient clipping | 1.0 (L2 norm) |
| Early stop | 92% coverage |

---

## Environment

| Component | Specification |
|---|---|
| Grid | 20×20 (400 cells) |
| Agent start | Position (0,0) — top-left |
| Action space | 9 discrete (N, NE, E, SE, S, SW, W, NW, WAIT) |
| Field of view | 2-cell radius, Bresenham line-of-sight |
| Obstacles | 3 random line segments (3–10 cells) per episode |
| Convoy robots | 3 dynamic robots on A\*-planned border paths |
| Evaluation scenarios | Standard, Dense, Sparse, Large Blocks |

---

## Reward Function

| Event | Reward |
|---|---|
| New cell explored via FoV | +10.0 × count |
| Movement step | −1.0 |
| Revisiting explored cell | −2.0 |
| WAIT action | −1.5 |
| Collision | −100.0 |
| Loop detected | −3.0 |
| Stuck detected | −5.0 |

---

## Dual Detection System

### Loop Detection — every 50 steps
1. **Entropy check** — unique positions in last 20 steps ÷ 20 < 0.30 → suspect loop
2. **Pattern check** — confirms repeating sequence with ≥3 repetitions
3. If confirmed with no coverage progress → trigger A\* escape + −3.0 penalty

### Stuck Detection — every 20 steps
1. **Spatial clustering** — avg distance of last 30 positions from centroid < 3.0 cells
2. **Coverage stagnation** — no 0.5% gain for 40+ consecutive steps
3. Both conditions must be true → trigger A\* escape + −5.0 penalty

---

## Action Space

| Index | Action | Δrow | Δcol |
|---|---|---|---|
| 0 | N | −1 | 0 |
| 1 | NE | −1 | +1 |
| 2 | E | 0 | +1 |
| 3 | SE | +1 | +1 |
| 4 | S | +1 | 0 |
| 5 | SW | +1 | −1 |
| 6 | W | 0 | −1 |
| 7 | NW | −1 | −1 |
| 8 | WAIT | 0 | 0 |

---

## Project Structure

```
neural-q-learning-explorer/
├── variants/
│   ├── nearest_cell/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── cluster_astar/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── periodic_astar/
│       ├── train.py       # 6-channel, periodic A* every 25 steps
│       ├── evaluate.py    # 6-channel evaluator
│       └── NOTES.md       # Architecture details and compatibility notes
├── checkpoints/
│   ├── nearest_cell/      # checkpoint_ep500/1000/1500/2000.pt + final_model.pt
│   ├── cluster_astar/     # checkpoint_ep500/1000/1500/2000.pt + final_model.pt
│   └── periodic_astar/    # checkpoint_ep500/1000/1500/2000.pt + final_model.pt
├── results/
│   ├── nearest_cell_training_curves.png
│   ├── cluster_astar_training_curves.png
│   └── periodic_astar_training_curves.png
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Pretrained Models

All three variants ship with pretrained checkpoints (2,500 episodes each, ~40 MB per file):

| File | Episode | Location |
|---|---|---|
| `final_model.pt` | 2,500 | `checkpoints/<variant>/` |
| `checkpoint_ep2000.pt` | 2,000 | `checkpoints/<variant>/` |
| `checkpoint_ep1500.pt` | 1,500 | `checkpoints/<variant>/` |
| `checkpoint_ep1000.pt` | 1,000 | `checkpoints/<variant>/` |
| `checkpoint_ep500.pt` | 500 | `checkpoints/<variant>/` |

> **Git LFS:** Each `.pt` file is ~40 MB (~600 MB total). If you hit GitHub's 100 MB file limit, use Git LFS:
> ```bash
> git lfs install
> git lfs track "*.pt"
> git add .gitattributes
> ```

Training curves for all variants are in `results/`.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/neural-q-learning-explorer.git
cd neural-q-learning-explorer
pip install -r requirements.txt
```

---

## Usage

Run all commands from the **repo root**.

### Training

```bash
python variants/nearest_cell/train.py
python variants/cluster_astar/train.py
python variants/periodic_astar/train.py
```

Key hyperparameters (top of each `train.py`):

```python
N_EPISODES                  = 2500
MAX_STEPS_PER_EPISODE       = 1000
GRID_SIZE                   = 20
FOV_RANGE                   = 2
EARLY_STOP_COVERAGE         = 0.92
LEARNING_RATE               = 0.0005
GAMMA                       = 0.98
# periodic_astar only:
ASTAR_PERIODIC_INTERVAL     = 25
PERIODIC_COVERAGE_THRESHOLD = 0.85
```

### Evaluation

```bash
# nearest_cell
python variants/nearest_cell/evaluate.py \
  --model_path checkpoints/nearest_cell/final_model.pt

# cluster_astar
python variants/cluster_astar/evaluate.py \
  --model_path checkpoints/cluster_astar/final_model.pt

# periodic_astar (6-channel — must use its own evaluator)
python variants/periodic_astar/evaluate.py \
  --model_path checkpoints/periodic_astar/final_model.pt
```

Each evaluator tests four obstacle scenarios — Standard, Dense, Sparse, Large Blocks — and prints a final performance rating.

---

## Which Variant to Use

| Goal | Recommended Variant |
|---|---|
| Maximum step efficiency | **Nearest Cell** — 132.8 avg steps, 100% success |
| Strategic targeting of large areas | **Cluster A\*** — 76.6-cell avg cluster, 100% success |
| Curriculum / guided training | **Periodic A\*** — 74% proactive guidance support |

All three converge to essentially the same coverage (92.36–92.40%), so the choice depends on your efficiency and intervention philosophy requirements.

---

## Requirements

- Python 3.8+
- PyTorch ≥ 2.0
- NumPy ≥ 1.24
- Matplotlib ≥ 3.7

---

## License

MIT
