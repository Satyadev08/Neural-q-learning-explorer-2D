# Neural Q-Learning Robot Explorer

A reinforcement learning system that trains a robot to autonomously explore a 20×20 grid using a **CNN-based Q-Network**, with intelligent stuck/loop detection and **A\* pathfinding escape** strategies.

This repo contains three experimental **escape strategy variants** that share the same base architecture, environment, and training loop — differing in how and when the robot decides to invoke A\* guidance.

---

## Variants

| Variant | Folder | Escape Strategy | Channels | A\* Trigger |
|---|---|---|---|---|
| **Nearest Cell** | `variants/nearest_cell/` | BFS to the single nearest unexplored cell | 5 | Emergency only (stuck / loop) |
| **Cluster A\*** | `variants/cluster_astar/` | Flood-fill → target the largest unexplored cluster | 5 | Emergency only (stuck / loop) |
| **Periodic A\*** | `variants/periodic_astar/` | Nearest cell, also called proactively every 25 steps | 6 | Every 25 steps + emergency |


### Evolution of escape strategies

```
nearest_cell  →  cluster_astar  →  periodic_astar
    │                │                  │
    │                │                  └─ A* fires every 25 steps proactively
    │                │                     (not only when stuck/looping)
    │                └─ Targets largest cluster of unexplored cells
    │                   instead of nearest single cell
    └─ BFS to nearest unvisited cell, A* path to it
```

---

## Shared Architecture

All variants use the identical Q-Network (5-channel version shown):

```
Input: 5-channel 20×20 grid
  Ch 0: Obstacles
  Ch 1: FOV-visited cells
  Ch 2: Robot position
  Ch 3: Convoy robot positions
  Ch 4: Current Field of View
 (Ch 5: A* guidance signal — 6-channel eval variant only)

CNN:
  Conv1: 5 → 16 filters (3×3, padding=1) → ReLU
  Conv2: 16 → 32 filters (3×3, padding=1) → ReLU → Flatten

FC:
  FC1: 12,800 → 256 → ReLU
  FC2: 256 → 128 → ReLU
  FC3: 128 → 9  (Q-values per action)
```

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

## Reward Structure

| Event | Reward |
|---|---|
| New cell explored via FoV | +10 × n_cells |
| Revisiting a cell | −2.0 |
| Each step taken | −1.0 |
| WAIT action | −1.5 |
| Collision | −100.0 |
| Loop detected (no progress) | −3.0 |
| Stuck detected | −5.0 |

---

## Detection Systems (all variants)

### Loop Detection — every 50 steps
1. **Entropy check** — unique positions in last 20 steps ÷ 20 < 0.30 → suspect loop
2. **Pattern check** — confirms with repeating position sequence (≥3 repetitions)
3. If confirmed with no coverage progress → trigger A\* escape

### Stuck Detection — every 20 steps
1. **Spatial clustering** — avg Manhattan distance of last 30 positions from centroid < 3.0
2. **Coverage stagnation** — no coverage gain for 40+ consecutive steps
3. If both conditions true → trigger A\* escape

### Periodic A\* (periodic_astar only)
- Fires every **25 steps** regardless of stuck/loop status
- Only active when coverage < 85%
- Usage tracked separately from emergency A\* in training stats

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
│       ├── train.py       # 5-channel, periodic A* every 25 steps
│       ├── evaluate.py    
│       └── NOTES.md       # What is new in this variant + compatibility notes
├── checkpoints/
│   ├── nearest_cell/          # checkpoint_ep500/1000/1500/2000.pt + final_model.pt
│   ├── cluster_astar/         # checkpoint_ep500/1000/1500/2000.pt + final_model.pt
│   └── periodic_astar/        # checkpoint_ep500/1000/1500/2000.pt + final_model.pt
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

All three variants ship with pretrained checkpoints (2500 episodes each, ~40 MB per file):

| File | Episode | Location |
|---|---|---|
| `final_model.pt` | 2500 | `checkpoints/<variant>/` |
| `checkpoint_ep2000.pt` | 2000 | `checkpoints/<variant>/` |
| `checkpoint_ep1500.pt` | 1500 | `checkpoints/<variant>/` |
| `checkpoint_ep1000.pt` | 1000 | `checkpoints/<variant>/` |
| `checkpoint_ep500.pt` | 500 | `checkpoints/<variant>/` |

> **Git LFS:** The `.pt` files are ~40 MB each (~600 MB total). If you hit GitHub push limits, track them with Git LFS:
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

Key settings (top of each `train.py`):

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
python variants/periodic_astar/evaluate.py \
  --model_path checkpoints/nearest_cell/final_model.pt

# cluster_astar
python variants/cluster_astar/evaluate.py \
  --model_path checkpoints/cluster_astar/final_model.pt

# periodic_astar
python variants/periodic_astar/evaluate.py \
  --model_path checkpoints/periodic_astar/final_model.pt
```

Evaluation tests four obstacle scenarios — Standard, Dense, Sparse, Large Blocks — and prints a performance rating.

---

## Environment Details

- **Grid**: 20×20, random obstacles regenerated each episode
- **Field of View**: Bresenham line-of-sight, radius 2
- **Convoy Robots**: 3 dynamic robots on A\*-planned border-to-border routes
- **Early stop**: Episode ends at 92% coverage

---

## Requirements

- Python 3.8+
- PyTorch >= 2.0
- NumPy >= 1.24
- Matplotlib >= 3.7

---

## License

MIT
