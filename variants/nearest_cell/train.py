"""
Neural Network Q-Learning with Dual Detection System
=====================================================
Addresses TWO problems:
1. LOOP DETECTION - Robot repeating position patterns
2. STUCK DETECTION - Robot staying in same small area

Performance-optimized:
- Loop check every 50 steps (not every step)
- Stuck check every 20 steps
- Entropy check for quick detection
- Pattern detection for deep analysis
- min_repetitions = 3 (fewer false positives)

Stuck Detection specifically solves the "robot at (11,0)" problem
from user's visualization where robot stayed in corner too long.
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

# ===== A* ESCAPE STRATEGY SETTINGS =====
USE_ASTAR_ESCAPE = True  # Use A* to navigate to unexplored areas when stuck
ASTAR_PATH_MAX_LENGTH = 50  # Max path length to follow
ASTAR_SEARCH_RADIUS = 20  # Search radius for nearest unexplored cell
FOLLOW_PATH_STEPS = 10  # Number of steps to follow A* path before re-evaluating
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
        # Could be stuck but not looping
        return False, "entropy_only", {"entropy": entropy}
    
    def get_escape_actions(self, current_pos, valid_actions):
        """
        Prioritize actions leading to less-visited positions
        Returns: List of actions sorted by visit frequency (best first)
        """
        if len(self.position_history) < 5:
            return valid_actions
        
        # Count visits to recent positions
        recent = list(self.position_history)[-20:]
        position_counts = {}
        for pos in recent:
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        # Action directions (adjust to your action space)
        action_to_direction = {
            0: (-1, 0),   # N
            1: (-1, 1),   # NE
            2: (0, 1),    # E
            3: (1, 1),    # SE
            4: (1, 0),    # S
            5: (1, -1),   # SW
            6: (0, -1),   # W
            7: (-1, -1),  # NW
            8: (0, 0)     # WAIT
        }
        
        # Score each valid action
        action_scores = []
        for action in valid_actions:
            if action not in action_to_direction:
                action_scores.append((action, 0))
                continue
            
            dr, dc = action_to_direction[action]
            next_pos = (current_pos[0] + dr, current_pos[1] + dc)
            
            # Lower score for frequently visited positions
            visits = position_counts.get(next_pos, 0)
            score = -visits  # Negative: fewer visits = higher priority
            
            action_scores.append((action, score))
        
        # Sort by score (highest first)
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [action for action, _ in action_scores]
    
    def reset_episode(self):
        """Reset for new episode"""
        self.position_history.clear()
        self.loops_detected_episode = 0
    
    def get_stats(self):
        """Get detection statistics"""
        return {
            'loops_this_episode': self.loops_detected_episode,
            'total_entropy_detections': self.entropy_detections,
            'total_pattern_detections': self.pattern_detections,
            'history_length': len(self.position_history),
            'unique_positions': len(set(self.position_history))
        }


# --------------------------
# STUCK ROBOT DETECTOR
# --------------------------
class SimpleStuckDetector:
    """
    Detects when robot stays in same small area too long
    Addresses the "robot stuck at (11,0)" problem from user's image
    
    Detection criteria:
    1. Positions clustered in small radius (< 3 cells average distance)
    2. AND coverage not increasing for 40+ steps
    """
    
    def __init__(self, window_size=30, radius_threshold=3.0, no_progress_threshold=40):
        self.position_history = deque(maxlen=window_size)
        self.window_size = window_size
        self.radius_threshold = radius_threshold
        self.no_progress_threshold = no_progress_threshold
        
        # Coverage tracking
        self.last_coverage = 0.0
        self.steps_no_progress = 0
        
        # Statistics
        self.stuck_detections = 0
        
    def add_position(self, position):
        """Track position - call every step"""
        self.position_history.append(tuple(position))
    
    def is_stuck(self, current_coverage):
        """
        Check if robot is stuck
        
        Returns: (is_stuck, reason, details)
        """
        if len(self.position_history) < self.window_size:
            return False, "not_enough_history", {}
        
        # Check 1: Are positions clustered in small area?
        positions = list(self.position_history)
        center_row = np.mean([p[0] for p in positions])
        center_col = np.mean([p[1] for p in positions])
        
        # Calculate average distance from center
        distances = [
            abs(p[0] - center_row) + abs(p[1] - center_col)
            for p in positions
        ]
        avg_distance = np.mean(distances)
        
        in_small_area = avg_distance < self.radius_threshold
        
        # Check 2: Is coverage increasing?
        if current_coverage > self.last_coverage + 0.005:
            # Coverage increasing - reset counter
            self.steps_no_progress = 0
            self.last_coverage = current_coverage
            coverage_stagnant = False
        else:
            # No progress
            self.steps_no_progress += 1
            coverage_stagnant = self.steps_no_progress >= self.no_progress_threshold
        
        # Robot is stuck if BOTH conditions are true
        is_stuck = in_small_area and coverage_stagnant
        
        if is_stuck:
            self.stuck_detections += 1
            reason = "stuck_in_small_area"
            details = {
                'avg_distance': avg_distance,
                'center': (center_row, center_col),
                'steps_no_progress': self.steps_no_progress,
                'coverage': current_coverage
            }
        else:
            reason = "moving_normally"
            details = {'avg_distance': avg_distance}
        
        return is_stuck, reason, details
    
    def get_escape_direction(self, current_pos):
        """
        Calculate direction to escape stuck area
        
        Returns: (escape_row_dir, escape_col_dir)
        """
        if len(self.position_history) < 5:
            return (0, 0)
        
        # Calculate center of position cluster
        positions = list(self.position_history)
        center_row = np.mean([p[0] for p in positions])
        center_col = np.mean([p[1] for p in positions])
        
        # Move AWAY from center
        escape_row = 1 if current_pos[0] > center_row else -1
        escape_col = 1 if current_pos[1] > center_col else -1
        
        return (escape_row, escape_col)
    
    def get_escape_actions(self, current_pos, valid_actions):
        """
        Prioritize actions that move away from stuck area
        
        Returns: List of actions sorted by escape score (best first)
        """
        if not valid_actions or len(self.position_history) < 5:
            return valid_actions
        
        # Get escape direction
        escape_row, escape_col = self.get_escape_direction(current_pos)
        
        # Action to direction mapping
        action_to_direction = {
            0: (-1, 0),   # N
            1: (-1, 1),   # NE
            2: (0, 1),    # E
            3: (1, 1),    # SE
            4: (1, 0),    # S
            5: (1, -1),   # SW
            6: (0, -1),   # W
            7: (-1, -1),  # NW
            8: (0, 0)     # WAIT
        }
        
        # Score each action based on alignment with escape direction
        action_scores = []
        for action in valid_actions:
            if action not in action_to_direction:
                action_scores.append((action, 0))
                continue
            
            dr, dc = action_to_direction[action]
            
            # Score: how well does this action match escape direction?
            score = 0
            if np.sign(dr) == np.sign(escape_row):
                score += 1
            if np.sign(dc) == np.sign(escape_col):
                score += 1
            
            # Penalty for WAIT action
            if action == 8:
                score -= 3
            
            action_scores.append((action, score))
        
        # Sort by score (higher = better escape action)
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [action for action, _ in action_scores]
    
    def reset(self):
        """Reset for new episode"""
        self.position_history.clear()
        self.last_coverage = 0.0
        self.steps_no_progress = 0
    
    def get_stats(self):
        """Get detection statistics"""
        return {
            'stuck_detections': self.stuck_detections,
            'steps_no_progress': self.steps_no_progress,
            'history_length': len(self.position_history)
        }


# --------------------------
# A* ESCAPE STRATEGY HELPER
# --------------------------
class AStarEscapeHelper:
    """
    Uses A* pathfinding to navigate to nearest unexplored area
    When stuck/loop detected with no coverage improvement, plans path to exploration
    """
    
    def __init__(self, grid_size=20, max_path_length=50, search_radius=20):
        self.grid_size = grid_size
        self.max_path_length = max_path_length
        self.search_radius = search_radius
        
        # Action to direction mapping
        self.action_to_direction = {
            0: (-1, 0),   # N
            1: (-1, 1),   # NE
            2: (0, 1),    # E
            3: (1, 1),    # SE
            4: (1, 0),    # S
            5: (1, -1),   # SW
            6: (0, -1),   # W
            7: (-1, -1),  # NW
            8: (0, 0)     # WAIT
        }
        
        # Reverse mapping: direction to action
        self.direction_to_action = {v: k for k, v in self.action_to_direction.items()}
        
        # Statistics
        self.paths_computed = 0
        self.paths_followed = 0
        
    def find_nearest_unexplored_cell(self, env, current_pos):
        """
        Find the nearest cell that hasn't been visited
        
        Returns: (row, col) of nearest unexplored cell, or None
        """
        visited = env.visited_cells
        
        # BFS to find nearest unexplored cell
        queue = deque([current_pos])
        explored = {current_pos}
        
        while queue:
            pos = queue.popleft()
            
            # Check if this cell is unexplored and free
            if pos not in visited and env.is_free(pos[0], pos[1]):
                return pos
            
            # Expand to neighbors (8-directional)
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                next_pos = (pos[0] + dr, pos[1] + dc)
                
                # Check bounds
                if not (0 <= next_pos[0] < self.grid_size and 0 <= next_pos[1] < self.grid_size):
                    continue
                
                # Check if already explored in BFS
                if next_pos in explored:
                    continue
                
                # Check if blocked by obstacle
                if not env.is_free(next_pos[0], next_pos[1]):
                    continue
                
                # Check search radius
                if abs(next_pos[0] - current_pos[0]) + abs(next_pos[1] - current_pos[1]) > self.search_radius:
                    continue
                
                explored.add(next_pos)
                queue.append(next_pos)
        
        return None  # No unexplored cell found in search radius
    
    def astar_path(self, env, start, goal):
        """
        A* pathfinding from start to goal
        Avoids obstacles and convoy robots
        
        Returns: List of positions [(r,c), ...] or None if no path
        """
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
                path.reverse()
                self.paths_computed += 1
                return path
            
            # Check 8 neighbors (all directions)
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                # Check bounds
                if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                    continue
                
                # Check if free (no obstacle, no convoy robot)
                if not env.can_move_to(neighbor[0], neighbor[1]):
                    continue
                
                # Calculate tentative g score
                # Diagonal moves cost slightly more (1.4 vs 1.0)
                move_cost = 1.4 if abs(dr) + abs(dc) == 2 else 1.0
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    # Heuristic: Manhattan distance
                    h = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    f = tentative_g + h
                    
                    heapq.heappush(open_set, (f, neighbor))
        
        return None  # No path found
    
    def path_to_actions(self, path):
        """
        Convert a path to a sequence of actions
        
        Args:
            path: List of (row, col) positions
            
        Returns: List of action indices
        """
        if not path or len(path) < 2:
            return []
        
        actions = []
        for i in range(len(path) - 1):
            current = path[i]
            next_pos = path[i + 1]
            
            # Calculate direction
            dr = next_pos[0] - current[0]
            dc = next_pos[1] - current[1]
            
            # Normalize to -1, 0, 1
            dr = max(-1, min(1, dr))
            dc = max(-1, min(1, dc))
            
            direction = (dr, dc)
            
            # Convert to action
            if direction in self.direction_to_action:
                action = self.direction_to_action[direction]
                actions.append(action)
            else:
                # Fallback: WAIT
                actions.append(8)
        
        return actions
    
    def get_escape_path_actions(self, env, current_pos, max_actions=None):
        """
        Get actions to navigate to nearest unexplored area
        
        Args:
            env: Environment
            current_pos: Current robot position
            max_actions: Maximum number of actions to return
            
        Returns: List of actions, or [] if no path found
        """
        # Find nearest unexplored cell
        target = self.find_nearest_unexplored_cell(env, current_pos)
        
        if target is None:
            return []
        
        # Plan path using A*
        path = self.astar_path(env, current_pos, target)
        
        if path is None or len(path) < 2:
            return []
        
        # Limit path length
        if len(path) > self.max_path_length:
            path = path[:self.max_path_length + 1]
        
        # Convert to actions
        actions = self.path_to_actions(path)
        
        # Limit number of actions if specified
        if max_actions is not None and len(actions) > max_actions:
            actions = actions[:max_actions]
        
        self.paths_followed += 1
        
        return actions
    
    def get_stats(self):
        """Get statistics"""
        return {
            'paths_computed': self.paths_computed,
            'paths_followed': self.paths_followed
        }


# --------------------------
# Neural Network Q-Function
# --------------------------
class QNetwork(nn.Module):
    """
    Neural Network for Q-function approximation
    """
    
    def __init__(self, grid_size=20, n_actions=9, n_channels=5):
        super(QNetwork, self).__init__()
        
        self.grid_size = grid_size
        self.n_actions = n_actions
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Calculate flattened size
        conv_output_size = grid_size * grid_size * 32
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_actions)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through network"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


# --------------------------
# Neural Q-Learning Agent
# --------------------------
class NeuralQLearningAgent:
    """Classic Q-Learning with Neural Network"""
    
    def __init__(self, grid_size=20, n_actions=9, n_channels=5):
        self.grid_size = grid_size
        self.n_actions = n_actions
        self.n_channels = n_channels
        
        # Single Q-network
        self.q_network = QNetwork(grid_size, n_actions, n_channels).to(DEVICE)
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        
        # Exploration
        self.epsilon = EPSILON_START
        
        # Statistics
        self.training_stats = {
            'episodes': [],
            'steps': [],
            'coverage': [],
            'rewards': [],
            'losses': [],
            'epsilon': [],
            'avg_coverage_50': [],
            'avg_steps_50': [],
            'loops_detected': []  # NEW: Track loops per episode
        }
        
        self.best_coverage = 0.0
        self.best_episode = 0
        self.steps_done = 0
    
    def get_state_representation(self, env):
        """Convert environment to multi-channel grid"""
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
        
        # Channel 4: FOV coverage
        if env.robot0_pos:
            fov_cells = env.get_fov_cells(env.robot0_pos)
            for r, c in fov_cells:
                state[4, r, c] = 1.0
        
        return state
    
    def select_action(self, state, valid_actions, eval_mode=False):
        """Epsilon-greedy action selection"""
        if not valid_actions:
            return 0
        
        epsilon = 0.0 if eval_mode else self.epsilon
        
        if random.random() < epsilon:
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                q_values = self.q_network(state_tensor)
                q_values_np = q_values.cpu().numpy()[0]
                
                # Mask invalid actions
                masked_q = np.full(self.n_actions, -np.inf)
                masked_q[valid_actions] = q_values_np[valid_actions]
                
                return int(np.argmax(masked_q))
    
    def update(self, state, action, reward, next_state, next_valid_actions, done):
        """Classic Q-Learning update with neural network"""
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)
        
        # Get current Q-value
        q_values = self.q_network(state_tensor)
        current_q = q_values[0, action].unsqueeze(0)
        
        # Compute target Q-value
        with torch.no_grad():
            next_q_values = self.q_network(next_state_tensor)[0]
            
            if done or not next_valid_actions:
                max_next_q = 0.0
            else:
                valid_next_q = [next_q_values[a].item() for a in next_valid_actions]
                max_next_q = max(valid_next_q)
            
            target_q = reward + GAMMA * max_next_q
        
        # Compute loss
        target_q_tensor = torch.FloatTensor([target_q]).to(DEVICE)
        loss = F.mse_loss(current_q, target_q_tensor)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_stats': self.training_stats,
            'best_coverage': self.best_coverage,
            'best_episode': self.best_episode
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=DEVICE)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_stats = checkpoint['training_stats']
        self.best_coverage = checkpoint.get('best_coverage', 0.0)
        self.best_episode = checkpoint.get('best_episode', 0)


# --------------------------
# Environment Classes
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
        
        done = (coverage >= EARLY_STOP_COVERAGE)
        
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


# --------------------------
# Environment Generation
# --------------------------
def generate_random_obstacles(env, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    env.clear_obstacles()
    
    for _ in range(NUM_OBSTACLE_PATTERNS):
        start_row = np.random.randint(1, env.rows - 1)
        start_col = np.random.randint(1, env.cols - 1)
        length = np.random.randint(MIN_OBSTACLE_LENGTH, MAX_OBSTACLE_LENGTH + 1)
        
        if np.random.random() < 0.5:
            for i in range(length):
                col = min(start_col + i, env.cols - 2)
                env.set_obstacle(start_row, col)
        else:
            for i in range(length):
                row = min(start_row + i, env.rows - 2)
                env.set_obstacle(row, start_col)
    
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


# --------------------------
# OPTIMIZED TRAINING LOOP WITH LOOP DETECTION
# --------------------------
def train_neural_qlearning(env, agent, n_episodes, max_steps):
    """
    Enhanced training with:
    1. Performance-optimized loop detection (every 50 steps)
    2. Stuck robot detection (every 20 steps)
    """
    
    print("\n" + "="*70)
    print("Neural Network Q-Learning with DUAL DETECTION")
    print("="*70)
    print(f"Episodes: {n_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Device: {DEVICE}")
    print("\n⚡ Loop Detection Settings (Performance Optimized):")
    print(f"  ✓ Check interval: Every {LOOP_CHECK_INTERVAL} steps")
    print(f"  ✓ Min repetitions: {LOOP_MIN_REPETITIONS}x (fewer false positives)")
    print(f"  ✓ Entropy threshold: {ENTROPY_THRESHOLD} (fast check)")
    print(f"  ✓ Loop penalty: {LOOP_PENALTY}")
    print(f"  ✓ History size: {HISTORY_SIZE}")
    print("\n🎯 Stuck Robot Detection Settings:")
    print(f"  ✓ Check interval: Every {STUCK_CHECK_INTERVAL} steps")
    print(f"  ✓ Radius threshold: {STUCK_RADIUS_THRESHOLD} cells")
    print(f"  ✓ No progress threshold: {STUCK_NO_PROGRESS_STEPS} steps")
    print(f"  ✓ Stuck penalty: {STUCK_PENALTY}")
    if USE_ASTAR_ESCAPE:
        print("\n🗺️  A* Escape Strategy: ENABLED")
        print(f"  ✓ Navigate to nearest unexplored area when stuck")
        print(f"  ✓ Max path length: {ASTAR_PATH_MAX_LENGTH}")
        print(f"  ✓ Follow path for: {FOLLOW_PATH_STEPS} steps")
    print("="*70 + "\n")
    
    save_dir = '../../checkpoints/nearest_cell'
    os.makedirs(save_dir, exist_ok=True)
    
    start_time = time.time()
    
    # Create optimized loop detector
    loop_detector = OptimizedLoopDetector(
        history_size=HISTORY_SIZE,
        loop_sizes=[2, 3, 4, 5]
    )
    
    # Create stuck robot detector
    stuck_detector = SimpleStuckDetector(
        window_size=STUCK_POSITION_WINDOW,
        radius_threshold=STUCK_RADIUS_THRESHOLD,
        no_progress_threshold=STUCK_NO_PROGRESS_STEPS
    )
    
    # Create A* escape helper
    astar_helper = None
    if USE_ASTAR_ESCAPE:
        astar_helper = AStarEscapeHelper(
            grid_size=GRID_SIZE,
            max_path_length=ASTAR_PATH_MAX_LENGTH,
            search_radius=ASTAR_SEARCH_RADIUS
        )
    
    for episode in range(n_episodes):
        # Generate random environment
        generate_random_obstacles(env, seed=episode)
        setup_random_convoy_robots(env, seed=episode)
        env.reset()
        
        state = agent.get_state_representation(env)
        episode_reward = 0
        episode_losses = []
        
        # Reset detectors for new episode
        loop_detector.reset_episode()
        stuck_detector.reset()
        last_coverage = 0
        
        # A* path following state
        astar_path_actions = []  # Queue of actions from A* path
        astar_path_step_count = 0  # How many steps we've been following path
        following_astar_path = False
        
        for step in range(max_steps):
            # Always track position for both detectors (O(1) operations)
            loop_detector.add_position(env.robot0_pos)
            stuck_detector.add_position(env.robot0_pos)
            
            valid_actions = env.get_valid_actions(env.robot0_pos)
            
            if not valid_actions:
                break
            
            # Initialize action selection variables
            action = None
            forced_action_reason = None
            apply_penalty = 0.0
            
            # ===== CHECK IF FOLLOWING A* PATH =====
            if following_astar_path and astar_path_actions:
                # Continue following the A* path
                action = astar_path_actions.pop(0)
                astar_path_step_count += 1
                forced_action_reason = "astar_path"
                
                # Stop following path if:
                # 1. Path exhausted (handled by pop above)
                # 2. Reached step limit
                # 3. Made coverage progress (let agent learn)
                current_coverage = len(env.visited_cells) / env.total_explorable
                if not astar_path_actions or astar_path_step_count >= FOLLOW_PATH_STEPS:
                    following_astar_path = False
                    astar_path_step_count = 0
                elif (current_coverage - last_coverage) > 0.02:
                    # Good progress, stop following path and let agent learn
                    following_astar_path = False
                    astar_path_actions = []
                    astar_path_step_count = 0
            
            # ===== STUCK ROBOT CHECK (Every 20 steps) =====
            # Check FIRST - higher priority than loop detection
            if action is None and step > 30 and step % STUCK_CHECK_INTERVAL == 0:
                current_coverage = len(env.visited_cells) / env.total_explorable
                is_stuck, stuck_reason, stuck_details = stuck_detector.is_stuck(current_coverage)
                
                if is_stuck:
                    print(f"  🚨 Step {step}: Robot STUCK at {env.robot0_pos}!")
                    print(f"     Reason: {stuck_reason}")
                    print(f"     Details: avg_dist={stuck_details.get('avg_distance', 0):.2f}, "
                          f"no_progress={stuck_details.get('steps_no_progress', 0)} steps")
                    
                    # Try A* escape if enabled
                    if USE_ASTAR_ESCAPE and astar_helper is not None:
                        astar_actions = astar_helper.get_escape_path_actions(
                            env, env.robot0_pos, max_actions=FOLLOW_PATH_STEPS
                        )
                        
                        if astar_actions:
                            print(f"     🗺️  Using A* path to nearest unexplored area ({len(astar_actions)} steps)")
                            astar_path_actions = astar_actions
                            following_astar_path = True
                            astar_path_step_count = 0
                            action = astar_path_actions.pop(0)
                            forced_action_reason = "astar_escape"
                            apply_penalty = STUCK_PENALTY
                        else:
                            print(f"     ⚠️  A* path not found, using directional escape")
                            # Fallback to old method
                            escape_actions = stuck_detector.get_escape_actions(
                                env.robot0_pos, valid_actions
                            )
                            if escape_actions:
                                action = escape_actions[0]
                                escape_dir = stuck_detector.get_escape_direction(env.robot0_pos)
                                print(f"     🏃 Forcing escape action {action} toward direction {escape_dir}")
                                forced_action_reason = "stuck_escape"
                                apply_penalty = STUCK_PENALTY
                    else:
                        # A* disabled, use old method
                        escape_actions = stuck_detector.get_escape_actions(
                            env.robot0_pos, valid_actions
                        )
                        if escape_actions:
                            action = escape_actions[0]
                            escape_dir = stuck_detector.get_escape_direction(env.robot0_pos)
                            print(f"     🏃 Forcing escape action {action} toward direction {escape_dir}")
                            forced_action_reason = "stuck_escape"
                            apply_penalty = STUCK_PENALTY
            
            # ===== OPTIMIZED LOOP CHECK (Every 50 steps) =====
            # Only check if action not already forced by stuck detection
            if action is None and step > 30 and step % LOOP_CHECK_INTERVAL == 0:
                # Two-stage detection:
                # 1. Fast entropy check (O(n))
                # 2. Pattern check only if entropy suggests problem (O(n*m))
                is_loop, loop_method, loop_details = loop_detector.comprehensive_check()
                
                if is_loop:
                    # Check if coverage is stagnant
                    current_coverage = len(env.visited_cells) / env.total_explorable
                    coverage_increasing = (current_coverage - last_coverage) > 0.01
                    
                    if not coverage_increasing:
                        # Try A* escape if enabled
                        if USE_ASTAR_ESCAPE and astar_helper is not None:
                            astar_actions = astar_helper.get_escape_path_actions(
                                env, env.robot0_pos, max_actions=FOLLOW_PATH_STEPS
                            )
                            
                            if astar_actions:
                                print(f"  🔄 Loop detected with no progress at {env.robot0_pos}")
                                print(f"     🗺️  Using A* path to nearest unexplored area ({len(astar_actions)} steps)")
                                astar_path_actions = astar_actions
                                following_astar_path = True
                                astar_path_step_count = 0
                                action = astar_path_actions.pop(0)
                                forced_action_reason = "astar_loop_escape"
                                apply_penalty = LOOP_PENALTY
                            else:
                                # Fallback to old random escape method
                                escape_actions = loop_detector.get_escape_actions(
                                    env.robot0_pos, valid_actions
                                )
                                if escape_actions and len(escape_actions) > 0:
                                    if np.random.random() < 0.7:
                                        action = escape_actions[0]
                                    else:
                                        top_choices = escape_actions[:min(3, len(escape_actions))]
                                        action = np.random.choice(top_choices)
                                    forced_action_reason = "loop_escape"
                                    apply_penalty = LOOP_PENALTY
                        else:
                            # A* disabled, use old method
                            escape_actions = loop_detector.get_escape_actions(
                                env.robot0_pos, valid_actions
                            )
                            if escape_actions and len(escape_actions) > 0:
                                if np.random.random() < 0.7:
                                    action = escape_actions[0]
                                else:
                                    top_choices = escape_actions[:min(3, len(escape_actions))]
                                    action = np.random.choice(top_choices)
                                forced_action_reason = "loop_escape"
                                apply_penalty = LOOP_PENALTY
                    
                    last_coverage = current_coverage
            
            # ===== NORMAL ACTION SELECTION =====
            # If no forced action from detectors, use normal policy
            if action is None:
                action = agent.select_action(state, valid_actions)
            # ================================
            
            # Take action
            _, reward, done, info = env.step_robot0(action)
            env.step_convoy_robots()
            
            # Apply penalty if any detection occurred
            reward += apply_penalty
            
            # Get next state
            next_state = agent.get_state_representation(env)
            next_valid_actions = env.get_valid_actions(env.robot0_pos)
            
            # ONLINE UPDATE
            loss = agent.update(state, action, reward, next_state, 
                              next_valid_actions, done)
            episode_losses.append(loss)
            
            state = next_state
            episode_reward += reward
            
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
            
            # Add detection info if detected
            detection_info = ""
            if loops_detected > 0:
                detection_info += f" | Loops: {loops_detected}"
            if stuck_detections > 0:
                detection_info += f" | Stuck: {stuck_detections}"
            
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
                print(f"  🗺️  A* Stats: Paths computed={astar_stats['paths_computed']}, "
                      f"Paths followed={astar_stats['paths_followed']}")
    
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
        print(f"  A* Escape Strategy:")
        print(f"    - Paths computed: {final_astar_stats['paths_computed']}")
        print(f"    - Paths followed: {final_astar_stats['paths_followed']}")
    print(f"{'='*70}\n")
    
    return agent


# --------------------------
# Main
# --------------------------
def main():
    print("\n" + "="*70)
    print("🧠 NEURAL Q-LEARNING WITH DUAL DETECTION SYSTEM")
    print("="*70)
    print("\nAlgorithm: Classic Q-Learning with Neural Network")
    print("\n⚡ Detection Systems:")
    print("  🔄 Loop Detection (every 50 steps):")
    print("     - Fast entropy check (O(n)) before expensive pattern check")
    print("     - Pattern detection only when entropy suggests problem")
    print("     - min_repetitions = 3 (fewer false positives)")
    print("  🎯 Stuck Robot Detection (every 20 steps):")
    print("     - Position clustering detection")
    print("     - Coverage progress monitoring")
    print("     - Forces escape from stuck areas")
    print("\nNetwork Architecture:")
    print("  📊 Input: Multi-channel grid (5 channels × 20 × 20)")
    print("  🔷 CNN Layers: 5 → 16 → 32 filters")
    print("  🔶 FC Layers: 12,800 → 256 → 128 → 9")
    print("="*70 + "\n")
    
    # Create environment
    env = ExplorationGridWorld(rows=GRID_SIZE, cols=GRID_SIZE, 
                               fov_range=FOV_RANGE, fov_enabled=True)
    env.set_robot0_start(0, 0)
    
    # Create agent
    agent = NeuralQLearningAgent(grid_size=GRID_SIZE, n_actions=9, n_channels=5)
    
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
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Coverage
    axes[0, 0].plot(stats['episodes'], stats['coverage'], 'b-', linewidth=0.5, alpha=0.3)
    axes[0, 0].plot(stats['episodes'], stats['avg_coverage_50'], 'b-', linewidth=2, label='Avg (50)')
    axes[0, 0].axhline(y=0.90, color='g', linestyle='--', label='Target')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Coverage')
    axes[0, 0].set_title('Neural Q-Learning Coverage')
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
    
    # Loops detected per episode
    axes[1, 2].plot(stats['episodes'], stats['loops_detected'], 'red', linewidth=1, alpha=0.6)
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Loops Detected')
    axes[1, 2].set_title('Loop Detection per Episode')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = '../../results/nearest_cell_training_curves.png'
    plt.savefig(plot_path, dpi=150)
    print(f"✓ Training curves saved to {plot_path}")
    
    plt.show()


if __name__ == "__main__":
    main()