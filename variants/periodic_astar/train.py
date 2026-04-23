"""
Neural Network Q-Learning with Periodic A* Guidance + 6-Channel State
====================================================================
Combines:
1. LOOP DETECTION - Robot repeating position patterns
2. STUCK DETECTION - Robot staying in same small area
3. PERIODIC A* - A* guidance every N steps
4. EMERGENCY A* - A* when stuck
5. 6-CHANNEL STATE - NEW! Includes A* guidance signal channel

NEW 6th Channel:
- Tells the NN when A* is active or recently completed
- Helps NN learn to handle A* "teleportation" properly
- Value = 1.0 when A* is guiding
- Value = 0.5 for 10 steps after A* completes (explore here!)
- Value decays to 0.0 gradually

This solves the "NN returns to old area after A* escape" problem!
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict, deque
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
import heapq
import time
import os

# --------------------------
# NEURAL Q-LEARNING SETTINGS
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Training settings
N_EPISODES = 2500
MAX_STEPS_PER_EPISODE = 1000
GAMMA = 0.98
LEARNING_RATE = 0.0005

# Exploration settings
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Environment settings
GRID_SIZE = 20
FOV_RANGE = 2
EARLY_STOP_COVERAGE = 0.92

# Checkpoint settings
CHECKPOINT_INTERVAL = 500
PRINT_EVERY = 50

# Visualization
ANIMATE_TRAINING = False

# Obstacle settings
NUM_OBSTACLE_PATTERNS = 3
MIN_OBSTACLE_LENGTH = 3
MAX_OBSTACLE_LENGTH = 10

# ===== LOOP DETECTION SETTINGS (PERFORMANCE OPTIMIZED) =====
LOOP_CHECK_INTERVAL = 50  # Check every 50 steps (not every step)
LOOP_PENALTY = -3.0
LOOP_MIN_REPETITIONS = 3  # Require 3+ repetitions (fewer false positives)
ENTROPY_THRESHOLD = 0.30  # Low entropy = stuck
HISTORY_SIZE = 50
# ===========================================================

# ===== STUCK ROBOT DETECTION SETTINGS =====
STUCK_CHECK_INTERVAL = 20  # Check for stuck every 20 steps
STUCK_PENALTY = -5.0  # Penalty when stuck in same area
STUCK_RADIUS_THRESHOLD = 3.0  # Max avg distance to be "stuck"
STUCK_POSITION_WINDOW = 30  # Track last 30 positions
STUCK_NO_PROGRESS_STEPS = 40  # Steps without coverage increase = stuck
# =========================================

# ===== A* STRATEGY SETTINGS =====
USE_ASTAR_ESCAPE = True  # Use A* to navigate to unexplored areas

# PERIODIC A* (NEW! - from test file)
USE_PERIODIC_ASTAR = True  # Enable periodic A* guidance during training
ASTAR_PERIODIC_INTERVAL = 25  # Use A* every N steps
PERIODIC_COVERAGE_THRESHOLD = 0.85  # Only use periodic A* if coverage below this

# EMERGENCY A* (when stuck)
ASTAR_SEARCH_RADIUS = 20  # Search radius for nearest unexplored cell
ASTAR_FOLLOW_STEPS = 10  # Number of steps to follow A* path
# ======================================


# --------------------------
# OPTIMIZED LOOP DETECTOR
# --------------------------
class OptimizedLoopDetector:
    """
    Performance-optimized loop detection:
    1. Fast entropy check (O(n))
    2. Pattern detection only when needed (O(n*m))
    3. Checks every 50 steps, not every step
    """
    
    def __init__(self, history_size=50, loop_sizes=[2, 3, 4, 5]):
        self.history_size = history_size
        self.loop_sizes = loop_sizes
        self.position_history = deque(maxlen=history_size)
        
        # Statistics
        self.loops_detected_episode = 0
        self.entropy_detections = 0
        self.pattern_detections = 0
        
    def add_position(self, position):
        """Add position to history - O(1)"""
        self.position_history.append(tuple(position))
    
    def check_entropy(self, window_size=20):
        """
        Fast entropy-based detection - O(n)
        Checks if robot is stuck in small area
        
        Returns: (is_stuck, entropy_score)
        """
        if len(self.position_history) < window_size:
            return False, 1.0
        
        # Get recent positions
        recent = list(self.position_history)[-window_size:]
        unique_positions = len(set(recent))
        entropy = unique_positions / window_size
        
        # Low entropy = stuck in small area
        is_stuck = entropy < ENTROPY_THRESHOLD
        
        if is_stuck:
            self.entropy_detections += 1
        
        return is_stuck, entropy
    
    def check_pattern(self, min_repetitions=3):
        """
        Detailed pattern detection - O(n*m)
        Only used when entropy suggests a problem
        
        Returns: (is_loop, loop_size, confidence)
        """
        if len(self.position_history) < min(self.loop_sizes) * min_repetitions:
            return False, 0, 0
        
        positions = list(self.position_history)
        
        # Check patterns from smallest to largest
        for loop_size in sorted(self.loop_sizes):
            if len(positions) < loop_size * min_repetitions:
                continue
            
            # Get recent pattern
            recent_pattern = positions[-loop_size:]
            repetition_count = 1
            
            # Count how many times this pattern repeats
            for i in range(loop_size, len(positions), loop_size):
                check_start = len(positions) - loop_size - i
                check_end = len(positions) - i
                
                if check_start < 0:
                    break
                
                check_pattern = positions[check_start:check_end]
                
                if check_pattern == recent_pattern:
                    repetition_count += 1
                else:
                    break
            
            # Loop confirmed if repeated enough times
            if repetition_count >= min_repetitions:
                self.pattern_detections += 1
                return True, loop_size, repetition_count
        
        return False, 0, 0
    
    def comprehensive_check(self):
        """
        Two-stage detection:
        1. Fast entropy check (always)
        2. Detailed pattern check (only if entropy low)
        
        Returns: (is_loop, method, details)
        """
        # Stage 1: Fast entropy check
        is_stuck_entropy, entropy = self.check_entropy()
        
        if not is_stuck_entropy:
            # Not stuck, no need for expensive pattern check
            return False, "none", {"entropy": entropy}
        
        # Stage 2: Confirm with pattern detection
        is_pattern, loop_size, reps = self.check_pattern(
            min_repetitions=LOOP_MIN_REPETITIONS
        )
        
        if is_pattern:
            # Confident loop detection
            self.loops_detected_episode += 1
            return True, "entropy+pattern", {
                "entropy": entropy,
                "loop_size": loop_size,
                "repetitions": reps
            }
        
        # Entropy low but no clear pattern
        return False, "entropy_only", {"entropy": entropy}
    
    def reset_episode_stats(self):
        """Reset per-episode counters"""
        self.loops_detected_episode = 0
    
    def get_stats(self):
        """Get detection statistics"""
        return {
            "total_entropy_detections": self.entropy_detections,
            "total_pattern_detections": self.pattern_detections
        }


# --------------------------
# STUCK ROBOT DETECTOR
# --------------------------
class StuckRobotDetector:
    """
    Detects when robot is stuck in small area:
    1. Position clustering - staying within small radius
    2. Coverage stagnation - no new cells explored
    """
    
    def __init__(self, window_size=30, radius_threshold=3.0, no_progress_steps=40):
        self.window_size = window_size
        self.radius_threshold = radius_threshold
        self.no_progress_steps = no_progress_steps
        
        self.position_window = deque(maxlen=window_size)
        self.last_coverage = 0
        self.steps_since_coverage_increase = 0
        self.stuck_detections = 0
    
    def add_position(self, position, current_coverage):
        """Update position history and coverage tracking"""
        self.position_window.append(tuple(position))
        
        # Track coverage progress
        if current_coverage > self.last_coverage:
            self.last_coverage = current_coverage
            self.steps_since_coverage_increase = 0
        else:
            self.steps_since_coverage_increase += 1
    
    def check_position_clustering(self):
        """Check if positions are clustered in small area"""
        if len(self.position_window) < self.window_size:
            return False
        
        positions = np.array(list(self.position_window))
        centroid = positions.mean(axis=0)
        
        # Calculate average distance from centroid
        distances = np.linalg.norm(positions - centroid, axis=1)
        avg_distance = distances.mean()
        
        return avg_distance < self.radius_threshold
    
    def check_coverage_stagnation(self):
        """Check if coverage hasn't increased recently"""
        return self.steps_since_coverage_increase >= self.no_progress_steps
    
    def is_stuck(self):
        """Combined stuck detection"""
        clustered = self.check_position_clustering()
        stagnant = self.check_coverage_stagnation()
        
        is_stuck = clustered or stagnant
        
        if is_stuck:
            self.stuck_detections += 1
        
        return is_stuck, {
            "clustered": clustered,
            "stagnant": stagnant,
            "steps_no_progress": self.steps_since_coverage_increase
        }
    
    def reset_episode_stats(self):
        """Reset for new episode"""
        self.position_window.clear()
        self.last_coverage = 0
        self.steps_since_coverage_increase = 0
    
    def get_stats(self):
        """Get detection statistics"""
        return {
            "stuck_detections": self.stuck_detections
        }


# --------------------------
# A* ESCAPE HELPER
# --------------------------
class AStarEscapeHelper:
    """
    Uses A* to navigate to nearest unexplored area.
    Now supports TWO modes:
    1. PERIODIC - guide exploration every N steps
    2. EMERGENCY - escape when stuck
    """
    
    def __init__(self, grid_size=20, search_radius=20):
        self.grid_size = grid_size
        self.search_radius = search_radius
        
        # Action mapping
        self.action_to_direction = {
            0: (-1, 0), 1: (-1, 1), 2: (0, 1), 3: (1, 1),
            4: (1, 0), 5: (1, -1), 6: (0, -1), 7: (-1, -1), 8: (0, 0)
        }
        self.direction_to_action = {v: k for k, v in self.action_to_direction.items()}
        
        # Statistics - separate tracking for periodic vs emergency
        self.paths_computed = 0
        self.paths_followed = 0
        self.periodic_uses = 0  # NEW
        self.emergency_uses = 0  # NEW
    
    def find_nearest_unexplored(self, env, current_pos):
        """BFS to find nearest unexplored cell"""
        from collections import deque
        
        queue = deque([current_pos])
        explored = {current_pos}
        
        while queue:
            pos = queue.popleft()
            
            # Found unexplored cell
            if pos not in env.visited_cells and env.is_free(pos[0], pos[1]):
                return pos
            
            # Expand to neighbors
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                next_pos = (pos[0] + dr, pos[1] + dc)
                
                # Boundary check
                if not (0 <= next_pos[0] < self.grid_size and 0 <= next_pos[1] < self.grid_size):
                    continue
                if next_pos in explored:
                    continue
                if not env.is_free(next_pos[0], next_pos[1]):
                    continue
                
                # Limit search radius
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
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            # Explore neighbors
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                # Check bounds and obstacles
                if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                    continue
                if not env.can_move_to(neighbor[0], neighbor[1]):
                    continue
                
                # Calculate cost (diagonal moves cost more)
                move_cost = 1.4 if abs(dr) + abs(dc) == 2 else 1.0
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, neighbor))
        
        return None
    
    def get_escape_actions(self, env, current_pos, is_periodic=False):
        """
        Get action sequence to nearest unexplored area
        
        Args:
            is_periodic: True if called from periodic A*, False if emergency
        """
        target = self.find_nearest_unexplored(env, current_pos)
        if target is None:
            return []
        
        path = self.astar_path(env, current_pos, target)
        if path is None or len(path) < 2:
            return []
        
        # Convert path to actions
        actions = []
        for i in range(len(path) - 1):
            dr = path[i+1][0] - path[i][0]
            dc = path[i+1][1] - path[i][1]
            # Clamp to valid direction
            dr = max(-1, min(1, dr))
            dc = max(-1, min(1, dc))
            direction = (dr, dc)
            action = self.direction_to_action.get(direction, 8)
            actions.append(action)
        
        # Update statistics
        self.paths_computed += 1
        if is_periodic:
            self.periodic_uses += 1
        else:
            self.emergency_uses += 1
        
        return actions
    
    def get_stats(self):
        """Get A* usage statistics"""
        return {
            "paths_computed": self.paths_computed,
            "paths_followed": self.paths_followed,
            "periodic_uses": self.periodic_uses,
            "emergency_uses": self.emergency_uses
        }


# --------------------------
# ENVIRONMENT
# --------------------------
class ExplorationGridWorld:
    """Grid world for exploration with obstacles and FOV"""
    
    def __init__(self, rows=20, cols=20, fov_range=2, fov_enabled=True):
        self.rows = rows
        self.cols = cols
        self.fov_range = fov_range
        self.fov_enabled = fov_enabled
        
        self.grid = np.zeros((rows, cols), dtype=int)
        self.robot_pos = None
        self.obstacles = set()
        self.visited_cells = set()
        self.total_explorable = 0
        
        # Multi-channel state representation - NOW 6 CHANNELS!
        self.n_channels = 6
        
        # NEW: A* guidance signal tracking
        self.astar_active = False
        self.steps_since_astar_complete = 0
        self.astar_signal_decay_steps = 20
    
    def set_robot0_start(self, row, col):
        """Set starting position"""
        self.robot_pos = np.array([row, col])
        self.visited_cells.add(tuple(self.robot_pos))
    
    def add_obstacle_pattern(self, start_r, start_c, length, direction):
        """Add line obstacle"""
        dr, dc = direction
        for i in range(length):
            r, c = start_r + i * dr, start_c + i * dc
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.obstacles.add((r, c))
                self.grid[r, c] = 1
    
    def finalize_obstacles(self):
        """Calculate total explorable cells"""
        self.total_explorable = self.rows * self.cols - len(self.obstacles)
    
    def is_free(self, row, col):
        """Check if cell is free"""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        return (row, col) not in self.obstacles
    
    def can_move_to(self, row, col):
        """Check if robot can move to position"""
        return self.is_free(row, col)
    
    def get_fov_visible_cells(self):
        """Get cells visible from current position"""
        visible = set()
        r, c = self.robot_pos
        
        for dr in range(-self.fov_range, self.fov_range + 1):
            for dc in range(-self.fov_range, self.fov_range + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    visible.add((nr, nc))
        
        return visible
    
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
    
    def get_state_tensor(self):
        """
        Generate 6-channel state tensor:
        Channel 0: Robot position
        Channel 1: Known obstacles
        Channel 2: Visited cells
        Channel 3: FOV visibility mask
        Channel 4: Unknown cells (not visited, not in FOV)
        Channel 5: A* guidance signal (NEW!)
        """
        state = np.zeros((self.n_channels, self.rows, self.cols), dtype=np.float32)
        
        # Robot position
        r, c = self.robot_pos
        state[0, r, c] = 1.0
        
        # Known obstacles
        for (obs_r, obs_c) in self.obstacles:
            state[1, obs_r, obs_c] = 1.0
        
        # Visited cells
        for (vis_r, vis_c) in self.visited_cells:
            state[2, vis_r, vis_c] = 1.0
        
        # FOV visibility
        if self.fov_enabled:
            visible_cells = self.get_fov_visible_cells()
            for (fov_r, fov_c) in visible_cells:
                state[3, fov_r, fov_c] = 1.0
        else:
            state[3, :, :] = 1.0
        
        # Unknown cells
        for row in range(self.rows):
            for col in range(self.cols):
                pos = (row, col)
                if pos not in self.visited_cells and pos not in self.obstacles:
                    if not self.fov_enabled or pos in visible_cells:
                        state[4, row, col] = 1.0
        
        # NEW: A* guidance signal (Channel 5)
        astar_signal = self.get_astar_signal_value()
        if astar_signal > 0:
            # Fill entire channel with signal value
            # This gives NN clear global signal about A* state
            state[5, :, :] = astar_signal
        
        return state
    
    def step(self, action):
        """Execute action and return (state, reward, done, info)"""
        # Action mapping: 0-7 = 8 directions, 8 = stay
        actions = [
            (-1, 0), (-1, 1), (0, 1), (1, 1),
            (1, 0), (1, -1), (0, -1), (-1, -1), (0, 0)
        ]
        
        dr, dc = actions[action]
        new_pos = self.robot_pos + np.array([dr, dc])
        
        # Check if move is valid
        if self.can_move_to(new_pos[0], new_pos[1]):
            self.robot_pos = new_pos
        
        # Update visited cells
        pos_tuple = tuple(self.robot_pos)
        newly_explored = pos_tuple not in self.visited_cells
        self.visited_cells.add(pos_tuple)
        
        # Calculate reward
        reward = 0.0
        if newly_explored:
            reward = 10.0  # Big reward for new cell
        else:
            reward = -0.5  # Small penalty for revisiting
        
        # Calculate coverage
        coverage = len(self.visited_cells) / self.total_explorable if self.total_explorable > 0 else 0
        
        # Get new state
        state = self.get_state_tensor()
        
        done = (coverage >= 0.99)
        
        info = {
            'coverage': coverage,
            'newly_explored': newly_explored,
            'position': tuple(self.robot_pos)
        }
        
        return state, reward, done, info
    
    def reset(self):
        """Reset environment"""
        self.visited_cells = set()
        self.robot_pos = np.array([0, 0])
        self.visited_cells.add(tuple(self.robot_pos))
        return self.get_state_tensor()


# --------------------------
# NEURAL NETWORK
# --------------------------
class QNetwork(nn.Module):
    """CNN-based Q-Network for grid exploration - NOW WITH 6 CHANNELS!"""
    
    def __init__(self, n_channels=6, grid_size=20, n_actions=9):  # Changed default to 6
        super(QNetwork, self).__init__()
        
        # Convolutional layers - now accept 6 input channels
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Calculate flattened size after convolutions
        conv_output_size = 32 * grid_size * grid_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_actions)
    
    def forward(self, x):
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


# --------------------------
# NEURAL Q-LEARNING AGENT
# --------------------------
class NeuralQLearningAgent:
    """Neural Q-Learning agent with exploration strategies"""
    
    def __init__(self, grid_size=20, n_actions=9, n_channels=6):  # Changed to 6
        self.grid_size = grid_size
        self.n_actions = n_actions
        self.n_channels = n_channels
        
        # Q-Network
        self.q_network = QNetwork(n_channels, grid_size, n_actions).to(DEVICE)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        
        # Exploration
        self.epsilon = EPSILON_START
        
        # Training stats
        self.training_stats = {
            'episodes': [],
            'steps': [],
            'coverage': [],
            'rewards': [],
            'losses': [],
            'epsilon': [],
            'loops_detected': [],
            'stuck_detected': [],
            'astar_periodic_uses': [],  # NEW
            'astar_emergency_uses': [],  # NEW
            'avg_coverage_50': [],
            'avg_steps_50': []
        }
        
        self.best_coverage = 0.0
        self.best_episode = 0
    
    def select_action(self, state, epsilon=None):
        """Epsilon-greedy action selection"""
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def train_step(self, state, action, reward, next_state, done):
        """Single training step"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)
        action_tensor = torch.LongTensor([action]).to(DEVICE)
        reward_tensor = torch.FloatTensor([reward]).to(DEVICE)
        done_tensor = torch.FloatTensor([done]).to(DEVICE)
        
        # Current Q-value
        current_q = self.q_network(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        
        # Target Q-value
        with torch.no_grad():
            next_q = self.q_network(next_state_tensor).max(1)[0]
            target_q = reward_tensor + GAMMA * next_q * (1 - done_tensor)
        
        # Compute loss
        loss = self.criterion(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_stats': self.training_stats
        }, path)
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']


# --------------------------
# OBSTACLE GENERATORS
# --------------------------
def generate_obstacles(env, n_patterns=3):
    """Generate random obstacle patterns"""
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    
    for _ in range(n_patterns):
        start_r = random.randint(0, env.rows - 1)
        start_c = random.randint(0, env.cols - 1)
        length = random.randint(MIN_OBSTACLE_LENGTH, MAX_OBSTACLE_LENGTH)
        direction = random.choice(directions)
        
        env.add_obstacle_pattern(start_r, start_c, length, direction)
    
    env.finalize_obstacles()


# --------------------------
# TRAINING FUNCTION
# --------------------------
def train_neural_qlearning(env, agent, n_episodes=1000, max_steps=500):
    """
    Train neural Q-learning agent with periodic A* guidance
    """
    print(f"\n{'='*70}")
    print(f"🚀 STARTING NEURAL Q-LEARNING WITH PERIODIC A* + 6-CHANNEL STATE")
    print(f"{'='*70}\n")
    
    # Create save directory
    save_dir = '../../checkpoints/periodic_astar'
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize detectors and helpers
    loop_detector = OptimizedLoopDetector(history_size=HISTORY_SIZE)
    stuck_detector = StuckRobotDetector(
        window_size=STUCK_POSITION_WINDOW,
        radius_threshold=STUCK_RADIUS_THRESHOLD,
        no_progress_steps=STUCK_NO_PROGRESS_STEPS
    )
    
    astar_helper = None
    if USE_ASTAR_ESCAPE:
        astar_helper = AStarEscapeHelper(grid_size=GRID_SIZE, search_radius=ASTAR_SEARCH_RADIUS)
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        # Reset environment with new obstacles
        env.obstacles = set()
        env.grid = np.zeros((env.rows, env.cols), dtype=int)
        generate_obstacles(env, NUM_OBSTACLE_PATTERNS)
        state = env.reset()
        
        # Reset detectors
        loop_detector.position_history.clear()
        loop_detector.reset_episode_stats()
        stuck_detector.reset_episode_stats()
        
        episode_reward = 0.0
        episode_losses = []
        
        # A* tracking for this episode
        astar_path_queue = []
        astar_periodic_count = 0  # NEW
        astar_emergency_count = 0  # NEW
        
        for step in range(max_steps):
            current_pos = tuple(env.robot_pos)
            current_coverage = len(env.visited_cells) / env.total_explorable
            
            # Decay A* signal if not active
            if not env.astar_active:
                env.step_astar_signal_decay()
            
            # Add position to detectors
            loop_detector.add_position(current_pos)
            stuck_detector.add_position(current_pos, current_coverage)
            
            # ===== PERIODIC A* (NEW!) =====
            if (USE_PERIODIC_ASTAR and 
                step > 0 and 
                step % ASTAR_PERIODIC_INTERVAL == 0 and
                current_coverage < PERIODIC_COVERAGE_THRESHOLD and
                len(astar_path_queue) == 0):
                
                # Use periodic A* guidance
                escape_actions = astar_helper.get_escape_actions(env, current_pos, is_periodic=True)
                if escape_actions:
                    astar_path_queue = escape_actions  # Follow COMPLETE path
                    env.set_astar_active(True)  # Signal A* is active
                    astar_periodic_count += 1
            
            # ===== LOOP DETECTION =====
            if step % LOOP_CHECK_INTERVAL == 0 and step > 0:
                is_loop, method, details = loop_detector.comprehensive_check()
                if is_loop:
                    episode_reward += LOOP_PENALTY
            
            # ===== STUCK DETECTION + EMERGENCY A* =====
            if step % STUCK_CHECK_INTERVAL == 0 and step > 0:
                is_stuck, stuck_info = stuck_detector.is_stuck()
                
                if is_stuck:
                    episode_reward += STUCK_PENALTY
                    
                    # Emergency A* escape
                    if USE_ASTAR_ESCAPE and len(astar_path_queue) == 0:
                        escape_actions = astar_helper.get_escape_actions(env, current_pos, is_periodic=False)
                        if escape_actions:
                            astar_path_queue = escape_actions  # Follow COMPLETE path
                            env.set_astar_active(True)  # Signal A* is active
                            astar_emergency_count += 1
            
            # ===== ACTION SELECTION =====
            if len(astar_path_queue) > 0:
                # Follow A* path
                action = astar_path_queue.pop(0)
                if astar_helper:
                    astar_helper.paths_followed += 1
                
                # Check if A* path is complete
                if len(astar_path_queue) == 0:
                    env.set_astar_active(False)  # A* complete, signal starts decaying
            else:
                # Normal Q-learning action
                action = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Train
            loss = agent.train_step(state, action, reward, next_state, done)
            episode_losses.append(loss)
            
            state = next_state
            
            # Check termination
            if done or info['coverage'] >= EARLY_STOP_COVERAGE:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record statistics
        final_coverage = len(env.visited_cells) / env.total_explorable
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        loops_detected = loop_detector.loops_detected_episode
        stuck_detections = stuck_detector.stuck_detections
        
        agent.training_stats['episodes'].append(episode)
        agent.training_stats['steps'].append(step + 1)
        agent.training_stats['coverage'].append(final_coverage)
        agent.training_stats['rewards'].append(episode_reward)
        agent.training_stats['losses'].append(avg_loss)
        agent.training_stats['epsilon'].append(agent.epsilon)
        agent.training_stats['loops_detected'].append(loops_detected)
        agent.training_stats['stuck_detected'].append(stuck_detections)
        agent.training_stats['astar_periodic_uses'].append(astar_periodic_count)
        agent.training_stats['astar_emergency_uses'].append(astar_emergency_count)
        
        # Rolling averages
        if len(agent.training_stats['coverage']) >= 50:
            avg_cov = np.mean(agent.training_stats['coverage'][-50:])
            avg_steps = np.mean(agent.training_stats['steps'][-50:])
        else:
            avg_cov = np.mean(agent.training_stats['coverage'])
            avg_steps = np.mean(agent.training_stats['steps'])
        
        agent.training_stats['avg_coverage_50'].append(avg_cov)
        agent.training_stats['avg_steps_50'].append(avg_steps)
        
        # Track best
        if final_coverage > agent.best_coverage:
            agent.best_coverage = final_coverage
            agent.best_episode = episode
        
        # Print progress
        if episode % PRINT_EVERY == 0:
            elapsed = time.time() - start_time
            remaining = (elapsed / (episode + 1)) * (n_episodes - episode - 1)
            
            # Detection info
            detection_info = ""
            if loops_detected > 0:
                detection_info += f" | Loops: {loops_detected}"
            if stuck_detections > 0:
                detection_info += f" | Stuck: {stuck_detections}"
            if astar_periodic_count > 0:
                detection_info += f" | A*-P: {astar_periodic_count}"
            if astar_emergency_count > 0:
                detection_info += f" | A*-E: {astar_emergency_count}"
            
            print(f"Ep {episode:5d}/{n_episodes} | "
                  f"Steps: {step+1:3d} | "
                  f"Cov: {final_coverage:5.1%} | "
                  f"Avg50: {avg_cov:5.1%} | "
                  f"Reward: {episode_reward:7.1f} | "
                  f"Loss: {avg_loss:6.3f} | "
                  f"ε: {agent.epsilon:.3f}{detection_info} | "
                  f"Time: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m")
        
        # Save checkpoints
        if episode % CHECKPOINT_INTERVAL == 0 and episode > 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_ep{episode}.pt')
            agent.save(checkpoint_path)
            print(f"  ✓ Checkpoint saved at episode {episode}")
            
            # Print detection stats
            loop_stats = loop_detector.get_stats()
            stuck_stats = stuck_detector.get_stats()
            print(f"  📊 Loop Stats: Entropy={loop_stats['total_entropy_detections']}, "
                  f"Pattern={loop_stats['total_pattern_detections']}")
            print(f"  🎯 Stuck Stats: Detections={stuck_stats['stuck_detections']}")
            if USE_ASTAR_ESCAPE and astar_helper is not None:
                astar_stats = astar_helper.get_stats()
                print(f"  🗺️  A* Stats: Periodic={astar_stats['periodic_uses']}, "
                      f"Emergency={astar_stats['emergency_uses']}, "
                      f"Paths={astar_stats['paths_computed']}")
    
    # Save final model
    agent.save(os.path.join(save_dir, 'final_model.pt'))
    
    total_time = time.time() - start_time
    final_loop_stats = loop_detector.get_stats()
    final_stuck_stats = stuck_detector.get_stats()
    
    print(f"\n{'='*70}")
    print(f"✅ NEURAL Q-LEARNING COMPLETE!")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Best coverage: {agent.best_coverage:.2%} (Episode {agent.best_episode})")
    print(f"Final avg (last 100): {np.mean(agent.training_stats['coverage'][-100:]):.2%}")
    print(f"\n📊 Detection Summary:")
    print(f"  Loop Detection:")
    print(f"    - Total entropy detections: {final_loop_stats['total_entropy_detections']}")
    print(f"    - Total pattern detections: {final_loop_stats['total_pattern_detections']}")
    print(f"  Stuck Robot Detection:")
    print(f"    - Total stuck detections: {final_stuck_stats['stuck_detections']}")
    if USE_ASTAR_ESCAPE and astar_helper is not None:
        final_astar_stats = astar_helper.get_stats()
        print(f"  A* Strategy:")
        print(f"    - Periodic uses: {final_astar_stats['periodic_uses']}")
        print(f"    - Emergency uses: {final_astar_stats['emergency_uses']}")
        print(f"    - Total paths computed: {final_astar_stats['paths_computed']}")
        print(f"    - Paths followed: {final_astar_stats['paths_followed']}")
    print(f"{'='*70}\n")
    
    return agent


# --------------------------
# Main
# --------------------------
def main():
    print("\n" + "="*70)
    print("🧠 NEURAL Q-LEARNING WITH PERIODIC A* + 6-CHANNEL STATE")
    print("="*70)
    print("\nAlgorithm: Classic Q-Learning with Neural Network")
    print("               + 6th channel for A* guidance signal")
    print("\n⚡ Systems:")
    print("  🔄 Loop Detection (every 50 steps):")
    print("     - Fast entropy check before expensive pattern check")
    print("     - min_repetitions = 3 (fewer false positives)")
    print("  🎯 Stuck Robot Detection (every 20 steps):")
    print("     - Position clustering detection")
    print("     - Coverage progress monitoring")
    if USE_PERIODIC_ASTAR:
        print(f"  🗺️  Periodic A* (NEW! - every {ASTAR_PERIODIC_INTERVAL} steps):")
        print(f"     - Guides exploration when coverage < {PERIODIC_COVERAGE_THRESHOLD:.0%}")
        print(f"     - Helps reach unexplored areas")
    print("  🚨 Emergency A* (when stuck):")
    print("     - Escapes from stuck situations")
    print("\nNetwork Architecture:")
    print("  📊 Input: Multi-channel grid (6 channels × 20 × 20)")  # Updated to 6
    print("  🔷 CNN Layers: 6 → 16 → 32 filters")  # Updated to 6
    print("  🔶 FC Layers: 12,800 → 256 → 128 → 9")
    print("  🆕 Channel 5: A* guidance signal (NEW!)")  # Added
    print("="*70 + "\n")
    
    # Create environment
    env = ExplorationGridWorld(rows=GRID_SIZE, cols=GRID_SIZE, 
                               fov_range=FOV_RANGE, fov_enabled=True)
    env.set_robot0_start(0, 0)
    
    # Create agent with 6 channels
    agent = NeuralQLearningAgent(grid_size=GRID_SIZE, n_actions=9, n_channels=6)  # Changed to 6
    
    # Print model summary
    print("Model Summary:")
    print(agent.q_network)
    total_params = sum(p.numel() for p in agent.q_network.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print()
    
    # Train
    trained_agent = train_neural_qlearning(env, agent, 
                                          n_episodes=N_EPISODES, 
                                          max_steps=MAX_STEPS_PER_EPISODE)
    
    # Plot results
    print("Generating training plots...")
    stats = trained_agent.training_stats
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    # Coverage
    axes[0, 0].plot(stats['episodes'], stats['coverage'], 'b-', linewidth=0.5, alpha=0.3)
    axes[0, 0].plot(stats['episodes'], stats['avg_coverage_50'], 'b-', linewidth=2, label='Avg (50)')
    axes[0, 0].axhline(y=0.90, color='g', linestyle='--', label='Target')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Coverage')
    axes[0, 0].set_title('Coverage Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Steps
    axes[0, 1].plot(stats['episodes'], stats['steps'], 'g-', linewidth=0.5, alpha=0.3)
    axes[0, 1].plot(stats['episodes'], stats['avg_steps_50'], 'g-', linewidth=2, label='Avg (50)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Steps per Episode')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Rewards
    axes[0, 2].plot(stats['episodes'], stats['rewards'], 'r-', linewidth=0.8, alpha=0.5)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Total Reward')
    axes[0, 2].set_title('Episode Rewards')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Loss
    axes[1, 0].plot(stats['episodes'], stats['losses'], 'purple', linewidth=1)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss (MSE)')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Epsilon
    axes[1, 1].plot(stats['episodes'], stats['epsilon'], 'orange', linewidth=1.5)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].set_title('Exploration Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Loops detected
    axes[1, 2].plot(stats['episodes'], stats['loops_detected'], 'red', linewidth=1, alpha=0.6)
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Loops Detected')
    axes[1, 2].set_title('Loop Detection per Episode')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Stuck detected
    axes[2, 0].plot(stats['episodes'], stats['stuck_detected'], 'brown', linewidth=1, alpha=0.6)
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Stuck Detections')
    axes[2, 0].set_title('Stuck Detection per Episode')
    axes[2, 0].grid(True, alpha=0.3)
    
    # Periodic A* uses (NEW!)
    axes[2, 1].plot(stats['episodes'], stats['astar_periodic_uses'], 'cyan', linewidth=1, alpha=0.6)
    axes[2, 1].set_xlabel('Episode')
    axes[2, 1].set_ylabel('Periodic A* Uses')
    axes[2, 1].set_title(f'Periodic A* Uses (every {ASTAR_PERIODIC_INTERVAL} steps)')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Emergency A* uses
    axes[2, 2].plot(stats['episodes'], stats['astar_emergency_uses'], 'magenta', linewidth=1, alpha=0.6)
    axes[2, 2].set_xlabel('Episode')
    axes[2, 2].set_ylabel('Emergency A* Uses')
    axes[2, 2].set_title('Emergency A* Uses (when stuck)')
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = '../../results/periodic_astar_training_curves.png'
    plt.savefig(plot_path, dpi=150)
    print(f"✓ Training curves saved to {plot_path}")
    
    plt.show()


if __name__ == "__main__":
    main()