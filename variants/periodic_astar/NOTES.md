# Periodic A* Variant — Notes

## What's New vs nearest_cell / cluster_astar

### 1. Periodic A* Guidance
A* is called **proactively every 25 steps** during both training and evaluation — not only as an emergency escape when stuck or looping. A coverage threshold gate prevents it from firing once the robot is already doing well:

```python
ASTAR_PERIODIC_INTERVAL     = 25    # Fire every N steps
PERIODIC_COVERAGE_THRESHOLD = 0.85  # Only if coverage below this
```

Periodic and emergency A* uses are tracked separately in training stats.

### 2. 6-Channel State Representation

The A* guidance signal is encoded directly into the state tensor as a 6th channel, so the network can **see** when it's being guided and learn to associate that signal with good behaviour.

| Channel | Content |
|---|---|
| 0 | Obstacles |
| 1 | FOV-visited cells |
| 2 | Robot position |
| 3 | Convoy robot positions |
| 4 | Current Field of View |
| **5** | **A\* guidance signal (cells along the planned path)** |

`train.py` and `evaluate.py` in this folder are fully compatible — both use `n_channels=6`.

## Compatibility with Other Variants

Models trained here **cannot** be loaded by `nearest_cell/evaluate.py` or `cluster_astar/evaluate.py` (those expect 5-channel input). Always use `periodic_astar/evaluate.py` for models trained in this variant.
