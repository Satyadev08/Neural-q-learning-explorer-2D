"""
TEST FILE FOR 6-CHANNEL NEURAL Q-LEARNING MODEL WITH PERIODIC A* USAGE
========================================================================
This test file is for models trained with Q_learning_with_neural_6CHANNEL.py

Model Architecture:
- Conv1: 6 → 16 filters (3×3)  ← UPDATED TO 6 CHANNELS
- Conv2: 16 → 32 filters (3×3)
- FC1: 12800 → 256
- FC2: 256 → 128
- FC3: 128 → 9 (output)

State Channels (6 total):
- Channel 0: Obstacles
- Channel 1: Visited cells
- Channel 2: Robot position
- Channel 3: Convoy robots
- Channel 4: FOV
- Channel 5: A* guidance signal (NEW!)

Checkpoint Location: ../../checkpoints/periodic_astar/
Checkpoint Keys: 'q_network_state_dict', 'optimizer_state_dict'

NEW FEATURES:
- 6-channel state representation with A* guidance signal
- A* used periodically (every N steps)
- A* also used when stuck (emergency escape)
- A* follows COMPLETE path to target cell before returning control to NN
- A* signal tracking helps NN understand when A* is guiding
- Separate tracking of periodic vs stuck A* usage
- Full convoy robots functionality included
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

# ===== A* CONFIGURATION =====
USE_ASTAR_ESCAPE = True  # Enable A* escape during testing

# PERIODIC A* (optional - can reduce performance if overused)
USE_PERIODIC_ASTAR = True  # Set to False to disable periodic A* and only use stuck detection
ASTAR_PERIODIC_INTERVAL = 25  # Use A* every N steps (only if USE_PERIODIC_ASTAR=True)
PERIODIC_COVERAGE_THRESHOLD = 0.85  # Only use periodic A* if coverage below this threshold

# STUCK DETECTION A* (recommended to keep enabled)
STUCK_CHECK_INTERVAL = 20  # Check if stuck every N steps
STUCK_RADIUS = 3.0  # Stuck if positions within this radius
STUCK_WINDOW = 25  # Track last N positions

# A* BEHAVIOR
ASTAR_SEARCH_RADIUS = 20  # Search radius for unexplored cells
# Note: A* will now follow the COMPLETE path to the target cell before returning control to NN
# This ensures the robot actually reaches unexplored areas instead of just moving partway there
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
        
        positions_array = np.array(list(self.positions))
        centroid = positions_array.mean(axis=0)
        distances = np.linalg.norm(positions_array - centroid, axis=1)
        max_dist = distances.max()
        
        return max_dist < self.radius
    
    def reset(self):
        self.positions.clear()


# --------------------------
# 6-CHANNEL NETWORK (for 6-channel trained models)
# --------------------------
class QNetwork(nn.Module):
    """
    CNN Q-Network for Q-Learning - 6 CHANNEL VERSION
    Architecture:
    - Conv1: 6 → 16 filters (3×3)
    - Conv2: 16 → 32 filters (3×3)
    - FC1: 12800 → 256
    - FC2: 256 → 128
    - FC3 (Output): 128 → 9 actions
    """
    
    def __init__(self, grid_size=20, n_actions=9, n_channels=6):  # Changed to 6
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
    """Evaluation agent for 6-channel model"""
    
    def __init__(self, model_path, grid_size=20, n_actions=9, n_channels=6):  # Changed to 6
        self.grid_size = grid_size
        self.n_actions = n_actions
        self.n_channels = n_channels
        
        print(f"\n{'='*70}")
        print("Loading 6-Channel Model...")
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
        
        print(f"✓ 6-Channel Model loaded from {model_path}")
        print(f"  Architecture: Conv(6→16→32) → FC(12800→256→128→9)")
        if self.training_epsilon > 0:
            print(f"  Training epsilon: {self.training_epsilon:.4f}")
        if self.training_stats and 'episodes' in self.training_stats:
            print(f"  Trained for {len(self.training_stats['episodes'])} episodes")
        print(f"{'='*70}\n")
    
    def get_state_representation(self, env):
        state = np.zeros((self.n_channels, self.grid_size, self.grid_size), dtype=np.float32)
        # Channel 0: Obstacles
        state[0] = env.grid.astype(np.float32)
        # Channel 1: Visited cells
        for r, c in env.visited_cells:
            state[1, r, c] = 1.0
        # Channel 2: Robot position
        if env.robot0_pos:
            state[2, env.robot0_pos[0], env.robot0_pos[1]] = 1.0
        # Channel 3: Convoy robots
        for robot in env.convoy_robots:
            if robot.active and robot.current_pos:
                r, c = robot.current_pos
                state[3, r, c] = 1.0
        # Channel 4: FOV
        if env.robot0_pos:
            fov_cells = env.get_fov_cells(env.robot0_pos)
            for r, c in fov_cells:
                state[4, r, c] = 1.0
        # Channel 5: A* guidance signal (NEW!)
        if hasattr(env, 'get_astar_signal_value'):
            astar_signal = env.get_astar_signal_value()
            if astar_signal > 0:
                state[5, :, :] = astar_signal
        return state
    
    def select_action(self, state, valid_actions):
        if not valid_actions:
            return 0
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.q_network(state_tensor).squeeze(0).cpu().detach().tolist()
            best_action = max(valid_actions, key=lambda a: q_values[a])
            return int(best_action)


# --------------------------
# ENVIRONMENT (with convoy robots)
# --------------------------
@dataclass
class Pose:
    row: int
    col: int


class ConvoyRobot:
    """Convoy robot that follows a predefined path"""
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
        
        # NEW: A* guidance signal tracking
        self.astar_active = False
        self.steps_since_astar_complete = 0
        self.astar_signal_decay_steps = 20
        
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
        for r in range(max(0, row - self.fov_range), min(self.rows, row + self.fov_range + 1)):
            for c in range(max(0, col - self.fov_range), min(self.cols, col + self.fov_range + 1)):
                if self.has_line_of_sight(pos, (r, c)) and self.grid[r, c] == 0:
                    fov_cells.add((r, c))
        return fov_cells
    
    def set_astar_active(self, active):
        """Set whether A* is currently active"""
        self.astar_active = active
        if not active:
            # A* just completed - start decay countdown
            self.steps_since_astar_complete = 0
    
    def step_astar_signal_decay(self):
        """Decay A* signal over time after completion"""
        if not self.astar_active and self.steps_since_astar_complete < self.astar_signal_decay_steps:
            self.steps_since_astar_complete += 1
    
    def get_astar_signal_value(self):
        """
        Get current A* signal value for state representation
        1.0 = A* actively guiding
        0.5 = Just completed (explore here!)
        0.3 = Fading
        0.0 = Normal operation
        """
        if self.astar_active:
            return 1.0
        elif self.steps_since_astar_complete < 10:
            return 0.5
        elif self.steps_since_astar_complete < 20:
            return 0.3
        else:
            return 0.0
    
    def is_free(self, row, col):
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        return self.grid[row, col] == 0
    
    def can_move_to(self, row, col):
        if not self.is_free(row, col):
            return False
        for robot in self.convoy_robots:
            if robot.active and robot.current_pos and robot.current_pos == (row, col):
                return False
        return True
    
    def step_robot0(self, action):
        dr, dc = self.actions[action]
        new_row = self.robot0_pos[0] + dr
        new_col = self.robot0_pos[1] + dc
        
        collision = False
        if self.can_move_to(new_row, new_col):
            self.robot0_pos = (new_row, new_col)
            self.physically_visited.add((new_row, new_col))
            fov_cells = self.get_fov_cells(self.robot0_pos)
            for cell in fov_cells:
                self.visited_cells.add(cell)
                self.visit_count[cell[0], cell[1]] += 1
        else:
            collision = True
        
        self.current_step += 1
        coverage = len(self.visited_cells) / self.total_explorable if self.total_explorable > 0 else 0
        reward = 1.0 if not collision else -0.5
        done = coverage >= 0.92
        
        return None, reward, done, {'coverage': coverage, 'collision': collision}
    
    def step_convoy_robots(self):
        """Update all convoy robots"""
        for robot in self.convoy_robots:
            robot.step()
    
    def reset(self):
        self.grid.fill(0)
        self.visited_cells.clear()
        self.physically_visited.clear()
        self.visit_count.fill(0)
        self.current_step = 0
        self.robot0_pos = self.robot0_start
        for robot in self.convoy_robots:
            robot.reset()
    
    def render(self, fig, ax, episode, step, coverage, title):
        ax.clear()
        
        # Draw grid cells
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r, c] == 1:
                    rect = Rectangle((c, r), 1, 1, facecolor='black')
                    ax.add_patch(rect)
                elif (r, c) in self.visited_cells:
                    rect = Rectangle((c, r), 1, 1, facecolor='lightblue', alpha=0.5)
                    ax.add_patch(rect)
        
        # Draw FOV
        if self.fov_enabled and self.robot0_pos:
            fov_cells = self.get_fov_cells(self.robot0_pos)
            for r, c in fov_cells:
                rect = Rectangle((c, r), 1, 1, 
                               facecolor='yellow', 
                               alpha=0.3, 
                               edgecolor='orange',
                               linewidth=0.5,
                               zorder=3)
                ax.add_patch(rect)
        
        # Draw convoy robots
        for robot in self.convoy_robots:
            if robot.active and robot.current_pos:
                r, c = robot.current_pos
                circle = Circle((c + 0.5, r + 0.5), 0.25, color=robot.color, zorder=4)
                ax.add_patch(circle)
        
        # Draw main robot
        if self.robot0_pos:
            circle = Circle((self.robot0_pos[1] + 0.5, self.robot0_pos[0] + 0.5), 
                          0.3, color='red', zorder=5)
            ax.add_patch(circle)
        
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'{title} | Ep {episode} | Step {step} | Coverage: {coverage:.1%}')
        ax.grid(True, alpha=0.3)
        
        # Legend
        legend_elements = [
            Patch(facecolor='red', label='Main Robot'),
            Patch(facecolor='lightblue', alpha=0.5, label='Visited'),
            Patch(facecolor='black', label='Obstacles'),
            Patch(facecolor='yellow', alpha=0.3, label=f'FOV (range={self.fov_range})')
        ]
        if self.convoy_robots:
            legend_elements.append(Patch(facecolor='blue', label='Convoy Robots'))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.pause(EVAL_PAUSE)


# --------------------------
# OBSTACLE GENERATORS
# --------------------------
def generate_standard_obstacles(env):
    env.reset()
    obstacles = [
        (5, 5), (5, 6), (5, 7),
        (10, 10), (10, 11), (11, 10), (11, 11),
        (15, 8), (15, 9), (15, 10)
    ]
    for obs in obstacles:
        env.set_obstacle(obs[0], obs[1])
    env.calculate_explorable_cells()

def generate_dense_obstacles(env):
    env.reset()
    np.random.seed(42)
    for _ in range(60):
        r, c = np.random.randint(0, env.rows), np.random.randint(0, env.cols)
        if (r, c) != (0, 0):
            env.set_obstacle(r, c)
    env.calculate_explorable_cells()

def generate_sparse_obstacles(env):
    env.reset()
    obstacles = [(5, 5), (10, 10), (15, 15)]
    for obs in obstacles:
        env.set_obstacle(obs[0], obs[1])
    env.calculate_explorable_cells()

def generate_large_obstacles(env):
    env.reset()
    for r in range(8, 12):
        for c in range(8, 12):
            env.set_obstacle(r, c)
    env.calculate_explorable_cells()


# --------------------------
# CONVOY ROBOT SETUP
# --------------------------
def astar_path_for_convoy(grid, start, goal):
    """Simple A* pathfinding for convoy robot paths"""
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
                h = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                f = tentative_g + h
                heapq.heappush(open_set, (f, neighbor))
    
    return None


def setup_random_convoy_robots(env, seed=None):
    """Create convoy robots with random paths along borders"""
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
    
    # Create 3 convoy robots with random paths
    for _ in range(3):
        robot_start = random_border_position()
        robot_goal = random_border_position()
        robot_path = astar_path_for_convoy(env.grid, robot_start, robot_goal)
        
        if robot_path and len(robot_path) > 1:
            env.add_convoy_robot(
                robot_id=len(env.convoy_robots), 
                path=robot_path, 
                color='blue',  # Blue for convoy robots
                start_delay=5
            )


# --------------------------
# EVALUATION FUNCTION
# --------------------------
def evaluate_on_scenario(env, agent, scenario_name, obstacle_generator, n_episodes=50, animate=False):
    print(f"\n{'='*70}")
    print(f"Evaluating: {scenario_name}")
    print(f"{'='*70}")
    
    results = {
        'coverage': [],
        'steps': [],
        'rewards': [],
        'explorable_cells': [],
        'collisions': [],
        'astar_periodic_uses': [],
        'astar_stuck_uses': []
    }
    
    if animate:
        plt.ion()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    astar_helper = AStarEscapeHelper(grid_size=GRID_SIZE, search_radius=ASTAR_SEARCH_RADIUS)
    
    for episode in range(n_episodes):
        obstacle_generator(env)
        setup_random_convoy_robots(env, seed=episode + 10000)
        env.set_robot0_start(0, 0)
        
        # Initialize coverage with starting position
        fov_cells = env.get_fov_cells(env.robot0_pos)
        for cell in fov_cells:
            env.visited_cells.add(cell)
        
        state = agent.get_state_representation(env)
        episode_reward = 0
        collision_count = 0
        astar_periodic_count = 0
        astar_stuck_count = 0
        
        stuck_detector = SimpleStuckDetector(window_size=STUCK_WINDOW, radius=STUCK_RADIUS)
        stuck_detector.add_position(env.robot0_pos)
        
        following_astar_path = False
        astar_path_actions = []
        
        should_animate = animate and episode < ANIMATE_EPISODES
        if should_animate:
            print(f"\n🎬 Animating Episode {episode}")
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # Decay A* signal if not active
            env.step_astar_signal_decay()
            
            stuck_detector.add_position(env.robot0_pos)
            
            valid_actions = []
            for action_idx in range(9):
                dr, dc = env.actions[action_idx]
                new_r = env.robot0_pos[0] + dr
                new_c = env.robot0_pos[1] + dc
                if env.can_move_to(new_r, new_c):
                    valid_actions.append(action_idx)
            
            action = None
            
            # Priority 1: Follow existing A* path to completion
            if following_astar_path and astar_path_actions:
                action = astar_path_actions.pop(0)
                
                # Give back control when path is complete
                if not astar_path_actions:
                    following_astar_path = False
                    env.set_astar_active(False)  # Signal A* complete, start decay
                    if should_animate:
                        print(f"     ✓ A* complete - reached target, signal decay starting")
            
            # Priority 2: Periodic A* (if enabled)
            elif USE_PERIODIC_ASTAR and step > 0 and step % ASTAR_PERIODIC_INTERVAL == 0:
                current_coverage = len(env.visited_cells) / env.total_explorable
                
                if current_coverage < PERIODIC_COVERAGE_THRESHOLD:
                    astar_actions = astar_helper.get_escape_actions(env, env.robot0_pos)
                    
                    if astar_actions:
                        if should_animate or episode < 5:
                            print(f"     ⏰ Step {step}: Periodic A* triggered ({current_coverage:.1%} coverage)")
                            print(f"     🗺️  Following A* path: {len(astar_actions)} steps to unexplored area")
                        
                        astar_path_actions = astar_actions  # Follow COMPLETE path
                        following_astar_path = True
                        env.set_astar_active(True)  # Signal A* is active
                        astar_periodic_count += 1
                        action = astar_path_actions.pop(0)
            
            # Priority 3: Stuck detection A*
            elif USE_ASTAR_ESCAPE and step > 30 and step % STUCK_CHECK_INTERVAL == 0:
                if stuck_detector.is_stuck():
                    current_coverage = len(env.visited_cells) / env.total_explorable
                    
                    if current_coverage < 0.95:
                        astar_actions = astar_helper.get_escape_actions(env, env.robot0_pos)
                        
                        if astar_actions:
                            if should_animate or episode < 5:
                                print(f"     🚨 Step {step}: STUCK detected at {env.robot0_pos} ({current_coverage:.1%} coverage)")
                                print(f"     🗺️  A* escape: {len(astar_actions)} steps to unexplored area")
                            
                            astar_path_actions = astar_actions  # Follow COMPLETE path
                            following_astar_path = True
                            env.set_astar_active(True)  # Signal A* is active
                            astar_stuck_count += 1
                            action = astar_path_actions.pop(0)
            
            # Priority 4: Normal agent action
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
        results['astar_periodic_uses'].append(astar_periodic_count)
        results['astar_stuck_uses'].append(astar_stuck_count)
        
        # Enhanced logging
        astar_info = ""
        if astar_periodic_count > 0 or astar_stuck_count > 0:
            astar_info = f" | A* [periodic: {astar_periodic_count}, stuck: {astar_stuck_count}]"
        
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
    
    # Enhanced A* statistics
    if USE_ASTAR_ESCAPE:
        total_periodic = sum(results['astar_periodic_uses'])
        total_stuck = sum(results['astar_stuck_uses'])
        total_astar = total_periodic + total_stuck
        
        if total_astar > 0:
            avg_periodic = np.mean(results['astar_periodic_uses'])
            avg_stuck = np.mean(results['astar_stuck_uses'])
            episodes_with_periodic = sum(1 for x in results['astar_periodic_uses'] if x > 0)
            episodes_with_stuck = sum(1 for x in results['astar_stuck_uses'] if x > 0)
            
            print(f"\n  🗺️  A* Usage Statistics:")
            if USE_PERIODIC_ASTAR and total_periodic > 0:
                print(f"    Periodic (every {ASTAR_PERIODIC_INTERVAL} steps): {total_periodic} total, {avg_periodic:.1f} avg/episode, used in {episodes_with_periodic}/{len(results['astar_periodic_uses'])} episodes")
            if total_stuck > 0:
                print(f"    Stuck detection: {total_stuck} total, {avg_stuck:.1f} avg/episode, used in {episodes_with_stuck}/{len(results['astar_stuck_uses'])} episodes")
            print(f"    TOTAL A* uses: {total_astar} ({total_periodic} periodic + {total_stuck} stuck)")
        elif not USE_PERIODIC_ASTAR:
            print(f"\n  🗺️  A* Usage: Periodic A* disabled, only stuck detection active")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test 6-Channel Neural Q-Learning Model with Periodic A*')
    parser.add_argument('--model_path', type=str, 
                       default='../../checkpoints/periodic_astar/final_model.pt',
                       help='Path to trained 6-channel model checkpoint')
    parser.add_argument('--n_episodes', type=int, default=50,
                       help='Episodes per scenario')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🧪 TESTING 6-CHANNEL NEURAL Q-LEARNING MODEL + PERIODIC A* + CONVOY")
    print("="*70)
    print(f"Architecture: Conv(6→16→32) → FC(256→128→9)")
    print(f"             + 6th channel for A* guidance signal")
    print(f"Model path: {args.model_path}")
    print(f"Episodes per scenario: {args.n_episodes}")
    print(f"\nA* Configuration:")
    if USE_PERIODIC_ASTAR:
        print(f"  ✓ Periodic A*: Every {ASTAR_PERIODIC_INTERVAL} steps (when coverage < {PERIODIC_COVERAGE_THRESHOLD:.0%})")
    else:
        print(f"  ✗ Periodic A*: DISABLED")
    print(f"  ✓ Stuck detection: Every {STUCK_CHECK_INTERVAL} steps (radius: {STUCK_RADIUS})")
    print(f"  ✓ A* follows COMPLETE path to target before returning control to NN")
    print(f"  ✓ A* signal tracked in Channel 5 (NN aware of A* state)")
    print("="*70)
    
    try:
        agent = EvaluationAgent(args.model_path, grid_size=GRID_SIZE, 
                               n_actions=9, n_channels=6)  # Changed to 6
    except FileNotFoundError:
        print(f"\n❌ Model not found at {args.model_path}")
        print("Make sure you trained with Q_learning_with_neural_6CHANNEL.py")
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
    
    # Overall summary
    print("\n" + "="*70)
    print("📊 OVERALL SUMMARY (All Scenarios Combined)")
    print("="*70)
    
    all_coverage = []
    all_steps = []
    all_rewards = []
    all_astar_periodic = []
    all_astar_stuck = []
    
    for results in all_results.values():
        all_coverage.extend(results['coverage'])
        all_steps.extend(results['steps'])
        all_rewards.extend(results['rewards'])
        if 'astar_periodic_uses' in results:
            all_astar_periodic.extend(results['astar_periodic_uses'])
        if 'astar_stuck_uses' in results:
            all_astar_stuck.extend(results['astar_stuck_uses'])
    
    print(f"\nAcross ALL {len(all_coverage)} episodes:")
    print(f"  Average Coverage: {np.mean(all_coverage):.2%} ± {np.std(all_coverage):.2%}")
    print(f"  Success Rate (≥90%): {sum(c >= 0.90 for c in all_coverage)/len(all_coverage):.1%}")
    print(f"  Average Steps: {np.mean(all_steps):.1f} ± {np.std(all_steps):.1f}")
    print(f"  Average Reward: {np.mean(all_rewards):.1f}")
    
    # Comprehensive A* statistics
    if USE_ASTAR_ESCAPE and (all_astar_periodic or all_astar_stuck):
        total_periodic = sum(all_astar_periodic)
        total_stuck = sum(all_astar_stuck)
        total_astar = total_periodic + total_stuck
        episodes_with_periodic = sum(1 for x in all_astar_periodic if x > 0)
        episodes_with_stuck = sum(1 for x in all_astar_stuck if x > 0)
        episodes_with_any_astar = sum(1 for p, s in zip(all_astar_periodic, all_astar_stuck) if p > 0 or s > 0)
        
        print(f"\n🗺️  A* USAGE SUMMARY:")
        print(f"  Total episodes: {len(all_astar_periodic)}")
        print(f"  Episodes using A* (any type): {episodes_with_any_astar} ({episodes_with_any_astar/len(all_astar_periodic):.1%})")
        
        if USE_PERIODIC_ASTAR and total_periodic > 0:
            print(f"\n  PERIODIC A* (every {ASTAR_PERIODIC_INTERVAL} steps, coverage < {PERIODIC_COVERAGE_THRESHOLD:.0%}):")
            print(f"    Total uses: {total_periodic}")
            print(f"    Episodes with periodic A*: {episodes_with_periodic} ({episodes_with_periodic/len(all_astar_periodic):.1%})")
            print(f"    Average per episode: {np.mean(all_astar_periodic):.2f}")
        elif not USE_PERIODIC_ASTAR:
            print(f"\n  PERIODIC A*: DISABLED")
        
        if total_stuck > 0:
            print(f"\n  STUCK-BASED A* (emergency escape):")
            print(f"    Total uses: {total_stuck}")
            print(f"    Episodes with stuck A*: {episodes_with_stuck} ({episodes_with_stuck/len(all_astar_stuck):.1%})")
            print(f"    Average per episode: {np.mean(all_astar_stuck):.2f}")
        
        if USE_PERIODIC_ASTAR and total_astar > 0:
            print(f"\n  COMBINED:")
            print(f"    Total A* uses: {total_astar} ({total_periodic} periodic + {total_stuck} stuck)")
            print(f"    Ratio: {total_periodic/total_astar:.1%} periodic, {total_stuck/total_astar:.1%} stuck")
    
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