"""
TEST FILE FOR 3-LAYER NEURAL Q-LEARNING MODEL
==============================================
This test file is specifically for models trained with Q_learning_with_neural.py

Model Architecture:
- Conv1: 5 → 16 filters (3×3)
- Conv2: 16 → 32 filters (3×3)
- FC1: 12800 → 256
- FC2: 256 → 128
- FC3: 128 → 9 (output)

Checkpoint Location: ../../checkpoints/nearest_cell/
Checkpoint Keys: 'q_network_state_dict', 'optimizer_state_dict'
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Patch
from dataclasses import dataclass
import heapq
import time
import os
import argparse

# --------------------------
# SETTINGS
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EVAL_EPISODES = 50

# ANIMATION SETTINGS
ANIMATE_EVALUATION = True          # Set to True to see animation!
ANIMATE_EPISODES = 3               # Number of episodes to animate (first N episodes)
RENDER_EVERY_N_STEPS = 5           # Render every N steps (lower = smoother, slower)
EVAL_PAUSE = 0.01                  # Pause between frames (seconds)

# Which scenarios to animate - use list for easier configuration
SCENARIOS_TO_ANIMATE = ['Standard']  # Options: 'Standard', 'Dense', 'Sparse', 'Large'

TEST_SEEN_OBSTACLES = True
TEST_DENSE_OBSTACLES = True
TEST_SPARSE_OBSTACLES = True
TEST_LARGE_OBSTACLES = True

GRID_SIZE = 20
FOV_RANGE = 2
FOV_ENABLED = True
MAX_STEPS_PER_EPISODE = 1000

# ===== A* ESCAPE FOR EVALUATION =====
USE_ASTAR_ESCAPE = True  # Enable A* escape during testing
STUCK_CHECK_INTERVAL = 20  # Check if stuck every N steps
STUCK_RADIUS = 3.0  # Stuck if positions within this radius
STUCK_WINDOW = 25  # Track last N positions
ASTAR_FOLLOW_STEPS = 10  # Follow A* path for N steps
ASTAR_SEARCH_RADIUS = 20  # Search radius for unexplored cells
# ====================================


# --------------------------
# A* ESCAPE HELPER FOR EVALUATION
# --------------------------
class AStarEscapeHelper:
    """Uses A* to navigate to nearest unexplored area during evaluation"""
    
    def __init__(self, grid_size=20, search_radius=20):
        self.grid_size = grid_size
        self.search_radius = search_radius
        
        self.action_to_direction = {
            0: (-1, 0), 1: (-1, 1), 2: (0, 1), 3: (1, 1),
            4: (1, 0), 5: (1, -1), 6: (0, -1), 7: (-1, -1), 8: (0, 0)
        }
        self.direction_to_action = {v: k for k, v in self.action_to_direction.items()}
        
        self.paths_used = 0
    
    def find_nearest_unexplored(self, env, current_pos):
        """BFS to find nearest unexplored cell"""
        from collections import deque
        
        queue = deque([current_pos])
        explored = {current_pos}
        
        while queue:
            pos = queue.popleft()
            
            if pos not in env.visited_cells and env.is_free(pos[0], pos[1]):
                return pos
            
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                next_pos = (pos[0] + dr, pos[1] + dc)
                
                if not (0 <= next_pos[0] < self.grid_size and 0 <= next_pos[1] < self.grid_size):
                    continue
                if next_pos in explored:
                    continue
                if not env.is_free(next_pos[0], next_pos[1]):
                    continue
                if abs(next_pos[0] - current_pos[0]) + abs(next_pos[1] - current_pos[1]) > self.search_radius:
                    continue
                
                explored.add(next_pos)
                queue.append(next_pos)
        
        return None
    
    def astar_path(self, env, start, goal):
        """A* pathfinding"""
        if not env.is_free(goal[0], goal[1]):
            return None
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                    continue
                if not env.can_move_to(neighbor[0], neighbor[1]):
                    continue
                
                move_cost = 1.4 if abs(dr) + abs(dc) == 2 else 1.0
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, neighbor))
        
        return None
    
    def get_escape_actions(self, env, current_pos):
        """Get action sequence to nearest unexplored area"""
        target = self.find_nearest_unexplored(env, current_pos)
        if target is None:
            return []
        
        path = self.astar_path(env, current_pos, target)
        if path is None or len(path) < 2:
            return []
        
        actions = []
        for i in range(len(path) - 1):
            dr = path[i+1][0] - path[i][0]
            dc = path[i+1][1] - path[i][1]
            dr = max(-1, min(1, dr))
            dc = max(-1, min(1, dc))
            direction = (dr, dc)
            action = self.direction_to_action.get(direction, 8)
            actions.append(action)
        
        self.paths_used += 1
        return actions


class SimpleStuckDetector:
    """Detects if robot stuck in small area during evaluation"""
    
    def __init__(self, window_size=25, radius=3.0):
        from collections import deque
        self.positions = deque(maxlen=window_size)
        self.window_size = window_size
        self.radius = radius
    
    def add_position(self, pos):
        self.positions.append(tuple(pos))
    
    def is_stuck(self):
        if len(self.positions) < self.window_size:
            return False
        
        positions = list(self.positions)
        center_r = np.mean([p[0] for p in positions])
        center_c = np.mean([p[1] for p in positions])
        
        avg_dist = np.mean([abs(p[0] - center_r) + abs(p[1] - center_c) for p in positions])
        
        return avg_dist < self.radius
    
    def reset(self):
        self.positions.clear()


# --------------------------
# 3-LAYER NETWORK (from Q_learning_with_neural.py)
# --------------------------
class QNetwork(nn.Module):
    """
    CNN Q-Network for Q-Learning
    Architecture:
    - Conv1: 5 → 16 filters (3×3)
    - Conv2: 16 → 32 filters (3×3)
    - FC1: 12800 → 256
    - FC2: 256 → 128
    - FC3 (Output): 128 → 9 actions
    """
    
    def __init__(self, grid_size=20, n_actions=9, n_channels=5):
        super(QNetwork, self).__init__()
        
        self.grid_size = grid_size
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        conv_output_size = grid_size * grid_size * 32
        
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_actions)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


# --------------------------
# EVALUATION AGENT
# --------------------------
class EvaluationAgent:
    """Evaluation agent for 3-layer model"""
    
    def __init__(self, model_path, grid_size=20, n_actions=9, n_channels=5):
        self.grid_size = grid_size
        self.n_actions = n_actions
        self.n_channels = n_channels
        
        print(f"\n{'='*70}")
        print("Loading 3-Layer Model...")
        print(f"{'='*70}")
        
        self.q_network = QNetwork(grid_size, n_actions, n_channels).to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Load with correct key name
        if 'q_network_state_dict' in checkpoint:
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            print("✓ Loaded using 'q_network_state_dict' key")
        elif 'q_network_state' in checkpoint:
            self.q_network.load_state_dict(checkpoint['q_network_state'])
            print("✓ Loaded using 'q_network_state' key")
        else:
            raise ValueError(f"Cannot find model weights. Keys: {list(checkpoint.keys())}")
        
        self.q_network.eval()
        
        if isinstance(checkpoint, dict):
            self.training_epsilon = checkpoint.get('epsilon', 0.0)
            self.training_stats = checkpoint.get('training_stats', None)
        else:
            self.training_epsilon = 0.0
            self.training_stats = None
        
        print(f"✓ 3-Layer Model loaded from {model_path}")
        print(f"  Architecture: Conv(5→16→32) → FC(12800→256→128→9)")
        if self.training_epsilon > 0:
            print(f"  Training epsilon: {self.training_epsilon:.4f}")
        if self.training_stats and 'episodes' in self.training_stats:
            print(f"  Trained for {len(self.training_stats['episodes'])} episodes")
        print(f"{'='*70}\n")
    
    def get_state_representation(self, env):
        state = np.zeros((self.n_channels, self.grid_size, self.grid_size), dtype=np.float32)
        state[0] = env.grid.astype(np.float32)
        for r, c in env.visited_cells:
            state[1, r, c] = 1.0
        if env.robot0_pos:
            state[2, env.robot0_pos[0], env.robot0_pos[1]] = 1.0
        for robot in env.convoy_robots:
            if robot.active and robot.current_pos:
                r, c = robot.current_pos
                state[3, r, c] = 1.0
        if env.robot0_pos:
            fov_cells = env.get_fov_cells(env.robot0_pos)
            for r, c in fov_cells:
                state[4, r, c] = 1.0
        return state
    
    def select_action(self, state, valid_actions):
        if not valid_actions:
            return 0
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.q_network(state_tensor).squeeze(0).cpu().detach().tolist()
            # Mask invalid actions by giving them -inf, pick best valid action
            best_action = max(valid_actions, key=lambda a: q_values[a])
            return int(best_action)


# --------------------------
# ENVIRONMENT (same for both)
# --------------------------
@dataclass
class Pose:
    row: int
    col: int


class ConvoyRobot:
    def __init__(self, robot_id, path, color, start_delay=0):
        self.id = robot_id
        self.path = path
        self.color = color
        self.path_index = 0
        self.start_delay = start_delay
        self.current_step = 0
        self.active = (start_delay == 0)
        self.current_pos = path[0] if path else None
    
    def step(self):
        self.current_step += 1
        if not self.active and self.current_step >= self.start_delay:
            self.active = True
        if not self.active:
            return self.current_pos
        if self.path_index < len(self.path) - 1:
            self.path_index += 1
        else:
            self.path_index = 0
        self.current_pos = self.path[self.path_index]
        return self.current_pos
    
    def reset(self):
        self.path_index = 0
        self.current_step = 0
        self.active = (self.start_delay == 0)
        self.current_pos = self.path[0] if self.path else None


class ExplorationGridWorld:
    def __init__(self, rows=20, cols=20, fov_range=2, fov_enabled=True):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=np.int8)
        self.fov_range = fov_range
        self.fov_enabled = fov_enabled
        self.robot0_pos = None
        self.robot0_start = None
        self.convoy_robots = []
        self.visited_cells = set()
        self.physically_visited = set()
        self.visit_count = np.zeros((rows, cols), dtype=np.int32)
        self.total_explorable = 0
        self.current_step = 0
        
        self.actions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1),
            (0, 0)
        ]
        self.action_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'WAIT']
    
    def clear_obstacles(self):
        self.grid.fill(0)
    
    def set_obstacle(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.grid[row, col] = 1
    
    def bresenham_line(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            points.append((y, x))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return points
    
    def has_line_of_sight(self, from_pos, to_pos):
        row0, col0 = from_pos
        row1, col1 = to_pos
        line_cells = self.bresenham_line(col0, row0, col1, row1)
        for i, (r, c) in enumerate(line_cells):
            if i == 0:
                continue
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                return False
            if self.grid[r, c] == 1:
                return False
        return True
    
    def calculate_explorable_cells(self):
        self.total_explorable = np.sum(self.grid == 0)
        return self.total_explorable
    
    def set_robot0_start(self, row, col):
        self.robot0_start = (row, col)
        self.robot0_pos = (row, col)
    
    def clear_convoy_robots(self):
        self.convoy_robots.clear()
    
    def add_convoy_robot(self, robot_id, path, color, start_delay=0):
        robot = ConvoyRobot(robot_id, path, color, start_delay)
        self.convoy_robots.append(robot)
    
    def get_fov_cells(self, pos):
        if not self.fov_enabled:
            return {pos}
        fov_cells = set()
        row, col = pos
        for dr in range(-self.fov_range, self.fov_range + 1):
            for dc in range(-self.fov_range, self.fov_range + 1):
                r, c = row + dr, col + dc
                if not (0 <= r < self.rows and 0 <= c < self.cols):
                    continue
                if self.grid[r, c] != 0:
                    continue
                if self.has_line_of_sight(pos, (r, c)):
                    fov_cells.add((r, c))
        return fov_cells
    
    def update_explored_cells(self, pos):
        newly_explored = 0
        fov_cells = self.get_fov_cells(pos)
        for cell in fov_cells:
            if cell not in self.visited_cells:
                self.visited_cells.add(cell)
                newly_explored += 1
        if pos not in self.physically_visited:
            self.physically_visited.add(pos)
        self.visit_count[pos[0], pos[1]] += 1
        return newly_explored
    
    def is_free(self, row, col):
        return (0 <= row < self.rows and 0 <= col < self.cols and 
                self.grid[row, col] == 0)
    
    def is_occupied_by_convoy(self, row, col):
        for robot in self.convoy_robots:
            if robot.active and robot.current_pos == (row, col):
                return True
        return False
    
    def can_move_to(self, row, col):
        return self.is_free(row, col) and not self.is_occupied_by_convoy(row, col)
    
    def get_valid_actions(self, pos):
        valid = []
        for action_idx, (dr, dc) in enumerate(self.actions):
            new_row, new_col = pos[0] + dr, pos[1] + dc
            if action_idx == 8:
                valid.append(action_idx)
            elif self.can_move_to(new_row, new_col):
                valid.append(action_idx)
        return valid
    
    def step_robot0(self, action_idx):
        dr, dc = self.actions[action_idx]
        new_row, new_col = self.robot0_pos[0] + dr, self.robot0_pos[1] + dc
        
        if action_idx == 8:
            reward = -1.5
            coverage = len(self.visited_cells) / self.total_explorable
            return self.robot0_pos, reward, False, {'coverage': coverage, 'collision': False}
        
        if not self.can_move_to(new_row, new_col):
            return self.robot0_pos, -100.0, False, {'coverage': 0, 'collision': True}
        
        old_pos = self.robot0_pos
        self.robot0_pos = (new_row, new_col)
        newly_explored = self.update_explored_cells(self.robot0_pos)
        coverage = len(self.visited_cells) / self.total_explorable
        reward = -1.0
        if newly_explored > 0:
            reward += newly_explored * 10.0
        if self.visit_count[new_row, new_col] > 1:
            reward += -2.0
        done = (coverage >= 0.92)
        return self.robot0_pos, reward, done, {'coverage': coverage, 'collision': False}
    
    def step_convoy_robots(self):
        for robot in self.convoy_robots:
            robot.step()
        collision = self.is_occupied_by_convoy(self.robot0_pos[0], self.robot0_pos[1])
        return collision
    
    def reset(self):
        self.robot0_pos = self.robot0_start
        for robot in self.convoy_robots:
            robot.reset()
        self.visited_cells.clear()
        self.physically_visited.clear()
        self.visit_count.fill(0)
        self.current_step = 0
        self.update_explored_cells(self.robot0_pos)
    
    def render(self, fig, ax, episode, step, coverage, scenario_name):
        """Render the environment for visualization"""
        ax.clear()
        
        # Draw grid lines
        for i in range(self.rows + 1):
            ax.plot([0, self.cols], [i, i], 'k-', linewidth=0.5, alpha=0.3)
        for j in range(self.cols + 1):
            ax.plot([j, j], [0, self.rows], 'k-', linewidth=0.5, alpha=0.3)
        
        # Draw obstacles (black)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == 1:
                    rect = Rectangle((c, self.rows - r - 1), 1, 1, 
                                   facecolor='black', edgecolor='none')
                    ax.add_patch(rect)
        
        # Draw FOV-explored cells (light blue)
        for (r, c) in self.visited_cells:
            rect = Rectangle((c, self.rows - r - 1), 1, 1,
                           facecolor='lightblue', edgecolor='none', alpha=0.5)
            ax.add_patch(rect)
        
        # Draw physically visited cells (dark blue)
        for (r, c) in self.physically_visited:
            rect = Rectangle((c, self.rows - r - 1), 1, 1,
                           facecolor='blue', edgecolor='none', alpha=0.3)
            ax.add_patch(rect)
        
        # Draw current FOV (yellow highlight)
        if self.robot0_pos:
            fov_cells = self.get_fov_cells(self.robot0_pos)
            for (r, c) in fov_cells:
                rect = Rectangle((c, self.rows - r - 1), 1, 1,
                               facecolor='yellow', edgecolor='none', alpha=0.2)
                ax.add_patch(rect)
        
        # Draw convoy robots (red circles)
        for robot in self.convoy_robots:
            if robot.active and robot.current_pos:
                r, c = robot.current_pos
                circle = Circle((c + 0.5, self.rows - r - 0.5), 0.3,
                              facecolor='red', edgecolor='darkred', linewidth=2)
                ax.add_patch(circle)
        
        # Draw Robot 0 (green circle)
        if self.robot0_pos:
            r0, c0 = self.robot0_pos
            circle = Circle((c0 + 0.5, self.rows - r0 - 0.5), 0.4,
                          facecolor='green', edgecolor='darkgreen', linewidth=2)
            ax.add_patch(circle)
        
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect('equal')
        ax.set_title(f'{scenario_name} | Episode {episode} | Step {step} | Coverage: {coverage:.1%}',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        # Legend
        legend_elements = [
            Patch(facecolor='black', label='Obstacles'),
            Patch(facecolor='lightblue', alpha=0.5, label='FOV Explored'),
            Patch(facecolor='blue', alpha=0.3, label='Physically Visited'),
            Patch(facecolor='yellow', alpha=0.2, label='Current FOV'),
            Patch(facecolor='green', label='Robot 0 (Learning Agent)'),
            Patch(facecolor='red', label='Convoy Robots')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.draw()
        plt.pause(EVAL_PAUSE)


# Environment generation functions
def generate_standard_obstacles(env, seed=None):
    if seed is not None:
        np.random.seed(seed)
    env.clear_obstacles()
    for _ in range(3):
        start_row = np.random.randint(1, env.rows - 1)
        start_col = np.random.randint(1, env.cols - 1)
        length = np.random.randint(3, 11)
        if np.random.random() < 0.5:
            for i in range(length):
                col = min(start_col + i, env.cols - 2)
                env.set_obstacle(start_row, col)
        else:
            for i in range(length):
                row = min(start_row + i, env.rows - 2)
                env.set_obstacle(row, start_col)
    env.calculate_explorable_cells()


def generate_dense_obstacles(env, seed=None):
    if seed is not None:
        np.random.seed(seed)
    env.clear_obstacles()
    for _ in range(5):
        start_row = np.random.randint(1, env.rows - 1)
        start_col = np.random.randint(1, env.cols - 1)
        length = np.random.randint(5, 12)
        if np.random.random() < 0.5:
            for i in range(length):
                col = min(start_col + i, env.cols - 2)
                env.set_obstacle(start_row, col)
        else:
            for i in range(length):
                row = min(start_row + i, env.rows - 2)
                env.set_obstacle(row, start_col)
    env.calculate_explorable_cells()


def generate_sparse_obstacles(env, seed=None):
    if seed is not None:
        np.random.seed(seed)
    env.clear_obstacles()
    for _ in range(2):
        start_row = np.random.randint(1, env.rows - 1)
        start_col = np.random.randint(1, env.cols - 1)
        length = np.random.randint(2, 6)
        if np.random.random() < 0.5:
            for i in range(length):
                col = min(start_col + i, env.cols - 2)
                env.set_obstacle(start_row, col)
        else:
            for i in range(length):
                row = min(start_row + i, env.rows - 2)
                env.set_obstacle(row, start_col)
    env.calculate_explorable_cells()


def generate_large_obstacles(env, seed=None):
    if seed is not None:
        np.random.seed(seed)
    env.clear_obstacles()
    for _ in range(np.random.randint(2, 4)):
        start_row = np.random.randint(1, env.rows - 6)
        start_col = np.random.randint(1, env.cols - 6)
        height = np.random.randint(3, 6)
        width = np.random.randint(3, 6)
        for r in range(height):
            for c in range(width):
                if start_row + r < env.rows - 1 and start_col + c < env.cols - 1:
                    env.set_obstacle(start_row + r, start_col + c)
    env.calculate_explorable_cells()


def astar_path(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue
            if grid[neighbor[0], neighbor[1]] == 1:
                continue
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                heapq.heappush(open_set, (f, neighbor))
    return None


def setup_random_convoy_robots(env, seed=None):
    if seed is not None:
        np.random.seed(seed + 1000)
    env.clear_convoy_robots()
    def random_border_position():
        side = np.random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            return (0, np.random.randint(0, env.cols))
        elif side == 'bottom':
            return (env.rows - 1, np.random.randint(0, env.cols))
        elif side == 'left':
            return (np.random.randint(0, env.rows), 0)
        else:
            return (np.random.randint(0, env.rows), env.cols - 1)
    for _ in range(3):
        robot_start = random_border_position()
        robot_goal = random_border_position()
        robot_path = astar_path(env.grid, robot_start, robot_goal)
        if robot_path and len(robot_path) > 1:
            env.add_convoy_robot(len(env.convoy_robots), robot_path, 'red', start_delay=5)


def evaluate_on_scenario(env, agent, scenario_name, obstacle_generator, n_episodes, animate=False):
    print(f"\n{'='*70}")
    print(f"📊 Evaluating on: {scenario_name}")
    if animate:
        print(f"   🎬 Animation ENABLED (first {ANIMATE_EPISODES} episodes)")
    if USE_ASTAR_ESCAPE:
        print(f"   🗺️  A* Escape: ENABLED (check every {STUCK_CHECK_INTERVAL} steps)")
    print(f"{'='*70}")
    
    results = {
        'coverage': [],
        'steps': [],
        'rewards': [],
        'explorable_cells': [],
        'collisions': [],
        'astar_uses': []
    }
    
    # Create A* helper if enabled
    astar_helper = None
    if USE_ASTAR_ESCAPE:
        astar_helper = AStarEscapeHelper(grid_size=GRID_SIZE, search_radius=ASTAR_SEARCH_RADIUS)
    
    # Setup animation if enabled
    if animate:
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle(f'3-Layer Model Evaluation: {scenario_name}', fontsize=14, fontweight='bold')
    
    for episode in range(n_episodes):
        obstacle_generator(env, seed=episode + 10000)
        setup_random_convoy_robots(env, seed=episode + 10000)
        env.reset()
        state = agent.get_state_representation(env)
        episode_reward = 0
        collision_count = 0
        astar_use_count = 0
        
        # Create stuck detector
        stuck_detector = None
        if USE_ASTAR_ESCAPE:
            stuck_detector = SimpleStuckDetector(window_size=STUCK_WINDOW, radius=STUCK_RADIUS)
        
        # A* path following state
        astar_path_actions = []
        following_astar_path = False
        astar_steps_followed = 0
        
        # Check if we should animate this episode
        should_animate = animate and episode < ANIMATE_EPISODES
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # Track position for stuck detection
            if USE_ASTAR_ESCAPE and stuck_detector is not None:
                stuck_detector.add_position(env.robot0_pos)
            
            valid_actions = env.get_valid_actions(env.robot0_pos)
            if not valid_actions:
                break
            
            # Determine action
            action = None
            
            # Priority 1: Follow A* path if active
            if following_astar_path and astar_path_actions:
                action = astar_path_actions.pop(0)
                astar_steps_followed += 1
                
                # Stop following if reached limit or made good progress
                if not astar_path_actions or astar_steps_followed >= ASTAR_FOLLOW_STEPS:
                    following_astar_path = False
                    astar_steps_followed = 0
                    if should_animate:
                        print(f"     ✓ Completed A* path segment")
            
            # Priority 2: Check if stuck and use A* escape
            elif USE_ASTAR_ESCAPE and step > 30 and step % STUCK_CHECK_INTERVAL == 0:
                if stuck_detector.is_stuck():
                    current_coverage = len(env.visited_cells) / env.total_explorable
                    
                    if current_coverage < 0.95:  # Only if not near complete
                        astar_actions = astar_helper.get_escape_actions(env, env.robot0_pos)
                        
                        if astar_actions:
                            if should_animate or episode < 5:
                                print(f"     🚨 Step {step}: Stuck at {env.robot0_pos} ({current_coverage:.1%} coverage)")
                                print(f"     🗺️  A* escape: {len(astar_actions)} steps to unexplored area")
                            
                            astar_path_actions = astar_actions[:ASTAR_FOLLOW_STEPS]
                            following_astar_path = True
                            astar_steps_followed = 0
                            astar_use_count += 1
                            action = astar_path_actions.pop(0)
                            astar_steps_followed += 1
            
            # Priority 3: Normal agent action
            if action is None:
                action = agent.select_action(state, valid_actions)
            
            _, reward, done, info = env.step_robot0(action)
            env.step_convoy_robots()
            if info['collision']:
                collision_count += 1
            state = agent.get_state_representation(env)
            episode_reward += reward
            
            # Render if animating
            if should_animate and step % RENDER_EVERY_N_STEPS == 0:
                env.render(fig, ax, episode, step, info['coverage'], scenario_name)
            
            if done or info['coverage'] >= 0.92:
                break
        
        final_coverage = len(env.visited_cells) / env.total_explorable
        results['coverage'].append(final_coverage)
        results['steps'].append(step + 1)
        results['rewards'].append(episode_reward)
        results['explorable_cells'].append(env.total_explorable)
        results['collisions'].append(collision_count)
        results['astar_uses'].append(astar_use_count)
        
        astar_info = f" | A*: {astar_use_count}" if astar_use_count > 0 else ""
        if episode < 3 or episode >= n_episodes - 3 or episode % 20 == 0 or should_animate:
            print(f"  Episode {episode:3d} | Coverage: {final_coverage:.1%} | "
                  f"Steps: {step+1:4d} | Reward: {episode_reward:7.1f}{astar_info}")
    
    if animate:
        plt.ioff()
        plt.close()
    
    print(f"\n{scenario_name} Results:")
    print(f"  Average Coverage: {np.mean(results['coverage']):.2%} ± {np.std(results['coverage']):.2%}")
    print(f"  Average Steps: {np.mean(results['steps']):.1f} ± {np.std(results['steps']):.1f}")
    print(f"  Average Reward: {np.mean(results['rewards']):.1f} ± {np.std(results['rewards']):.1f}")
    print(f"  Success Rate (>90%): {sum(c >= 0.90 for c in results['coverage'])/len(results['coverage']):.1%}")
    if USE_ASTAR_ESCAPE and results['astar_uses']:
        total_astar = sum(results['astar_uses'])
        avg_astar = np.mean(results['astar_uses'])
        episodes_with_astar = sum(1 for x in results['astar_uses'] if x > 0)
        print(f"  A* Escapes: {total_astar} total, {avg_astar:.1f} avg/episode, used in {episodes_with_astar}/{len(results['astar_uses'])} episodes")
    return results


def main():
    parser = argparse.ArgumentParser(description='Test 3-Layer Neural Q-Learning Model')
    parser.add_argument('--model_path', type=str, 
                       default='../../checkpoints/nearest_cell/final_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--n_episodes', type=int, default=50,
                       help='Episodes per scenario')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🧪 TESTING 3-LAYER NEURAL Q-LEARNING MODEL")
    print("="*70)
    print(f"Architecture: Conv(5→16→32) → FC(256→128→9)")
    print(f"Model path: {args.model_path}")
    print(f"Episodes per scenario: {args.n_episodes}")
    print("="*70)
    
    try:
        agent = EvaluationAgent(args.model_path, grid_size=GRID_SIZE, 
                               n_actions=9, n_channels=5)
    except FileNotFoundError:
        print(f"\n❌ Model not found at {args.model_path}")
        print("Make sure you trained with Q_learning_with_neural.py")
        return
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return
    
    env = ExplorationGridWorld(rows=GRID_SIZE, cols=GRID_SIZE,
                               fov_range=FOV_RANGE, fov_enabled=FOV_ENABLED)
    env.set_robot0_start(0, 0)
    
    all_results = {}
    
    if TEST_SEEN_OBSTACLES:
        all_results["Standard"] = evaluate_on_scenario(
            env, agent, "Standard Obstacles", generate_standard_obstacles, 
            args.n_episodes, animate=ANIMATE_EVALUATION and 'Standard' in SCENARIOS_TO_ANIMATE)
    
    if TEST_DENSE_OBSTACLES:
        all_results["Dense"] = evaluate_on_scenario(
            env, agent, "Dense Obstacles", generate_dense_obstacles, 
            args.n_episodes, animate=ANIMATE_EVALUATION and 'Dense' in SCENARIOS_TO_ANIMATE)
    
    if TEST_SPARSE_OBSTACLES:
        all_results["Sparse"] = evaluate_on_scenario(
            env, agent, "Sparse Obstacles", generate_sparse_obstacles, 
            args.n_episodes, animate=ANIMATE_EVALUATION and 'Sparse' in SCENARIOS_TO_ANIMATE)
    
    if TEST_LARGE_OBSTACLES:
        all_results["Large Blocks"] = evaluate_on_scenario(
            env, agent, "Large Block Obstacles", generate_large_obstacles, 
            args.n_episodes, animate=ANIMATE_EVALUATION and 'Large' in SCENARIOS_TO_ANIMATE)
    
    print("\n" + "="*70)
    print("✅ 3-LAYER MODEL EVALUATION COMPLETE")
    print("="*70)
    print("\nPer-Scenario Results:")
    for scenario, results in all_results.items():
        print(f"\n{scenario}:")
        print(f"  Coverage: {np.mean(results['coverage']):.2%} ± {np.std(results['coverage']):.2%}")
        print(f"  Success: {sum(c >= 0.90 for c in results['coverage'])/len(results['coverage']):.1%}")
        print(f"  Steps: {np.mean(results['steps']):.1f}")
    
    # Overall summary across all scenarios
    print("\n" + "="*70)
    print("📊 OVERALL SUMMARY (All Scenarios Combined)")
    print("="*70)
    
    all_coverage = []
    all_steps = []
    all_rewards = []
    all_astar_uses = []
    
    for results in all_results.values():
        all_coverage.extend(results['coverage'])
        all_steps.extend(results['steps'])
        all_rewards.extend(results['rewards'])
        if 'astar_uses' in results:
            all_astar_uses.extend(results['astar_uses'])
    
    print(f"\nAcross ALL {len(all_coverage)} episodes:")
    print(f"  Average Coverage: {np.mean(all_coverage):.2%} ± {np.std(all_coverage):.2%}")
    print(f"  Success Rate (≥90%): {sum(c >= 0.90 for c in all_coverage)/len(all_coverage):.1%}")
    print(f"  Average Steps: {np.mean(all_steps):.1f} ± {np.std(all_steps):.1f}")
    print(f"  Average Reward: {np.mean(all_rewards):.1f}")
    
    if USE_ASTAR_ESCAPE and all_astar_uses:
        total_astar = sum(all_astar_uses)
        episodes_with_astar = sum(1 for x in all_astar_uses if x > 0)
        print(f"\n🗺️  A* Escape Usage:")
        print(f"  Total escapes: {total_astar}")
        print(f"  Episodes using A*: {episodes_with_astar}/{len(all_astar_uses)} ({episodes_with_astar/len(all_astar_uses):.1%})")
        print(f"  Average per episode: {np.mean(all_astar_uses):.2f}")
    
    # Performance rating
    avg_coverage = np.mean(all_coverage)
    success_rate = sum(c >= 0.90 for c in all_coverage)/len(all_coverage)
    
    print(f"\n{'='*70}")
    print("🎯 PERFORMANCE RATING:")
    if avg_coverage >= 0.90 and success_rate >= 0.80:
        print("  ⭐⭐⭐ EXCELLENT - Model performs very well!")
    elif avg_coverage >= 0.85 and success_rate >= 0.60:
        print("  ⭐⭐ GOOD - Model performs well, minor improvements possible")
    elif avg_coverage >= 0.75:
        print("  ⭐ FAIR - Model works but needs improvement")
    else:
        print("  ⚠️  NEEDS TRAINING - Model performance is below expectations")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()