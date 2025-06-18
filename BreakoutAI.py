#!/usr/bin/env python3
"""
Breakout AI using NEAT (NeuroEvolution of Augmenting Topologies)

This module implements a Breakout game where AI agents learn to play
using the NEAT algorithm for evolving neural networks.

Author: Fbrrr
Date: 2025
License: MIT
"""

import os
import pickle
import random
import math
import time
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import neat
import pyxel
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Game constants
@dataclass(frozen=True)
class GameConfig:
    """Configuration constants for the game."""
    SCREEN_WIDTH: int = 1000
    SCREEN_HEIGHT: int = 600
    BALL_RADIUS: int = 8
    BALL_SPEED: float = 4.0
    PADDLE_WIDTH: int = 100
    PADDLE_HEIGHT: int = 15
    PADDLE_SPEED: float = 6.0
    BLOCK_WIDTH: int = 70
    BLOCK_HEIGHT: int = 25
    BLOCKS_PER_ROW: int = 10
    BLOCK_ROWS: int = 5
    TIMEOUT_SECONDS: int = 60
    FPS: int = 60
    
    # UI Layout
    GAME_AREA_WIDTH: int = 700
    GAME_AREA_HEIGHT: int = 600
    STATS_PANEL_WIDTH: int = 300
    STATS_PANEL_X: int = 700
    NEURAL_VIZ_WIDTH: int = 250
    NEURAL_VIZ_HEIGHT: int = 200
    NEURAL_VIZ_X: int = 720
    NEURAL_VIZ_Y: int = 300
    
    # Enhanced Colors
    BALL_COLORS: List[int] = None
    PADDLE_COLOR: int = 11  # Light blue
    BLOCK_COLORS: List[int] = None  # Will be set in post_init
    BACKGROUND_COLOR: int = 1  # Dark blue
    GAME_BACKGROUND_COLOR: int = 0  # Black
    TEXT_COLOR: int = 7  # White
    ACCENT_COLOR: int = 10  # Yellow
    SUCCESS_COLOR: int = 11  # Green
    WARNING_COLOR: int = 8   # Red

    def __post_init__(self):
        if self.BALL_COLORS is None:
            object.__setattr__(self, 'BALL_COLORS', [8, 9, 10, 11, 12, 13, 14, 15])
        if self.BLOCK_COLORS is None:
            # Different colors for each row - back rows get more vibrant colors
            object.__setattr__(self, 'BLOCK_COLORS', [8, 9, 10, 11, 12])

config = GameConfig()

class GameState(Enum):
    """Enum for game states."""
    PLAYING = "playing"
    FINISHED = "finished"
    TIMEOUT = "timeout"

class Vector2D:
    """Simple 2D vector class for position and velocity."""
    
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y
    
    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)
    
    def normalize(self) -> 'Vector2D':
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)

class Ball:
    """Represents the game ball with physics."""
    
    def __init__(self, x: float, y: float, color: int = None):
        self.position = Vector2D(x, y)
        self.radius = config.BALL_RADIUS
        self.color = color or random.choice(config.BALL_COLORS)
        
        # Initialize velocity with random angle
        angle = random.uniform(0.25 * math.pi, 0.75 * math.pi)  # 45-135 degrees
        self.velocity = Vector2D(
            config.BALL_SPEED * math.cos(angle),
            -config.BALL_SPEED * math.sin(angle)
        )
    
    def update(self) -> None:
        """Update ball position based on velocity."""
        self.position = self.position + self.velocity
    
    def handle_wall_collision(self) -> None:
        """Handle collision with walls."""
        # Left and right walls (use game area width)
        if (self.position.x - self.radius <= 0 or 
            self.position.x + self.radius >= config.GAME_AREA_WIDTH):
            self.velocity.x = -self.velocity.x
            # Keep ball in bounds
            self.position.x = max(self.radius, 
                                min(self.position.x, config.GAME_AREA_WIDTH - self.radius))
        
        # Top wall
        if self.position.y - self.radius <= 0:
            self.velocity.y = -self.velocity.y
            self.position.y = self.radius
    
    def hits_paddle(self, paddle: 'Paddle') -> bool:
        """Check and handle collision with paddle."""
        paddle_top = config.SCREEN_HEIGHT - config.PADDLE_HEIGHT
        
        if (self.position.y + self.radius >= paddle_top and
            paddle.position.x - paddle.width/2 <= self.position.x <= 
            paddle.position.x + paddle.width/2):
            
            # Calculate bounce angle based on where ball hits paddle
            hit_pos = (self.position.x - paddle.position.x) / (paddle.width/2)
            bounce_angle = hit_pos * math.pi/3  # Max 60 degrees
            
            speed = self.velocity.magnitude()
            self.velocity = Vector2D(
                speed * math.sin(bounce_angle),
                -abs(speed * math.cos(bounce_angle))  # Always bounce up
            )
            
            # Ensure ball is above paddle
            self.position.y = paddle_top - self.radius
            return True
        return False
    
    def hits_ground(self) -> bool:
        """Check if ball hits the ground."""
        return self.position.y + self.radius >= config.SCREEN_HEIGHT
    
    def draw(self) -> None:
        """Draw the ball with enhanced visuals."""
        x, y = int(self.position.x), int(self.position.y)
        # Main ball
        pyxel.circ(x, y, self.radius, self.color)
        # Inner highlight for 3D effect
        pyxel.circ(x - 2, y - 2, self.radius // 2, 7)  # White highlight

class Paddle:
    """AI-controlled paddle."""
    
    def __init__(self, genome, neat_config):
        self.position = Vector2D(config.GAME_AREA_WIDTH // 2, 0)
        self.width = config.PADDLE_WIDTH
        self.height = config.PADDLE_HEIGHT
        self.speed = config.PADDLE_SPEED
        
        # Neural network brain
        self.brain = neat.nn.FeedForwardNetwork.create(genome, neat_config)
        self.genome = genome
        self.genome.fitness = 0
        
        # Statistics
        self.total_distance = 0.0
        self.last_position = self.position.x
        
        # Neural network visualization data
        self.last_inputs = [0.0] * 15  # Updated for new input count
        self.last_outputs = [0.0] * 1
        self.hidden_activations = {}  # Store hidden node activations
    
    def update(self, ball: Ball, blocks: List = None) -> None:
        """Update paddle position based on neural network decision."""
        # Prepare inputs for neural network
        inputs = self._get_inputs(ball, blocks or [])
        
        # Get neural network output and capture hidden activations
        output = self._activate_with_tracking(inputs)
        decision = output[0]
        
        # Store for visualization
        self.last_inputs = inputs.copy()
        self.last_outputs = output.copy()
        
        # Move paddle based on decision
        old_x = self.position.x
        if decision > 0.1:  # Move right
            self.position.x += self.speed
        elif decision < -0.1:  # Move left
            self.position.x -= self.speed
        
        # Keep paddle in bounds
        half_width = self.width / 2
        self.position.x = max(half_width, 
                            min(self.position.x, config.GAME_AREA_WIDTH - half_width))
        
        # Track movement for fitness calculation
        self.total_distance += abs(self.position.x - old_x)
    
    def _activate_with_tracking(self, inputs: List[float]) -> List[float]:
        """Activate the network while tracking hidden node activations."""
        # Reset hidden activations
        self.hidden_activations = {}
        
        # Manual forward pass to capture hidden activations
        if hasattr(self.brain, 'nodes') and hasattr(self.brain, 'connections'):
            # Initialize node values
            node_values = {}
            
            # Set input values (NEAT uses negative keys for inputs)
            input_keys = [key for key in self.brain.nodes.keys() if key < 0]
            for i, key in enumerate(sorted(input_keys)):
                if i < len(inputs):
                    node_values[key] = inputs[i]
                else:
                    node_values[key] = 0.0
            
            # Add bias node (key 0)
            if 0 in self.brain.nodes:
                node_values[0] = 1.0
            
            # Process nodes in topological order (inputs -> hidden -> outputs)
            processed = set(node_values.keys())
            
            # Get all node keys sorted (inputs are negative, outputs/hidden are positive)
            all_keys = sorted(self.brain.nodes.keys())
            
            # Process remaining nodes
            for key in all_keys:
                if key in processed:
                    continue
                    
                # Calculate activation for this node
                activation = 0.0
                node = self.brain.nodes[key]
                
                # Sum weighted inputs from connected nodes
                for conn_key, connection in self.brain.connections.items():
                    if connection.key[1] == key and connection.enabled:
                        input_key = connection.key[0]
                        if input_key in node_values:
                            activation += node_values[input_key] * connection.weight
                
                # Apply bias
                activation += node.bias
                
                # Apply activation function (tanh by default)
                if hasattr(node, 'activation'):
                    if node.activation == 'tanh':
                        activation = math.tanh(activation)
                    elif node.activation == 'sigmoid':
                        activation = 1.0 / (1.0 + math.exp(-activation))
                    elif node.activation == 'relu':
                        activation = max(0, activation)
                else:
                    activation = math.tanh(activation)  # Default to tanh
                
                node_values[key] = activation
                
                # Store hidden node activations (positive keys that aren't the final output)
                if key > 0:
                    self.hidden_activations[key] = activation
                
                processed.add(key)
            
            # Return output values (typically just key 1 for single output)
            output_keys = [key for key in self.brain.nodes.keys() if key > 0]
            outputs = []
            for key in sorted(output_keys):
                if key in node_values:
                    outputs.append(node_values[key])
            
            return outputs if outputs else [0.0]
        
        else:
            # Fallback to standard activation
            return self.brain.activate(inputs)
    
    def _get_inputs(self, ball: Ball, blocks: List) -> List[float]:
        """Get comprehensive strategic inputs for the neural network."""
        inputs = []
        
        # 1. BASIC POSITION & VELOCITY (5 inputs)
        inputs.extend([
            # Paddle position (normalized)
            self.position.x / config.GAME_AREA_WIDTH,
            # Ball position (normalized)
            ball.position.x / config.GAME_AREA_WIDTH,
            ball.position.y / config.GAME_AREA_HEIGHT,
            # Ball velocity (normalized)
            ball.velocity.x / (config.BALL_SPEED * 2),
            ball.velocity.y / (config.BALL_SPEED * 2)
        ])
        
        # 2. BALL TRAJECTORY PREDICTION (3 inputs)
        # Predict where ball will be in next few frames
        future_ball_x = ball.position.x + ball.velocity.x * 10  # 10 frames ahead
        future_ball_y = ball.position.y + ball.velocity.y * 10
        
        # Predict ball intersection with paddle level
        paddle_y = config.SCREEN_HEIGHT - config.PADDLE_HEIGHT
        if ball.velocity.y != 0:
            frames_to_paddle = (paddle_y - ball.position.y) / ball.velocity.y
            predicted_x = ball.position.x + ball.velocity.x * frames_to_paddle
        else:
            predicted_x = ball.position.x
            
        inputs.extend([
            future_ball_x / config.GAME_AREA_WIDTH,  # Future ball X
            future_ball_y / config.GAME_AREA_HEIGHT,  # Future ball Y  
            predicted_x / config.GAME_AREA_WIDTH     # Predicted paddle intersection
        ])
        
        # 3. SPATIAL RELATIONSHIPS (2 inputs)
        # Distance from paddle to ball
        paddle_ball_distance = math.sqrt(
            (self.position.x - ball.position.x) ** 2 + 
            (paddle_y - ball.position.y) ** 2
        )
        max_distance = math.sqrt(config.GAME_AREA_WIDTH**2 + config.GAME_AREA_HEIGHT**2)
        
        # Relative position (is ball left/right of paddle)
        relative_position = (ball.position.x - self.position.x) / config.GAME_AREA_WIDTH
        
        inputs.extend([
            paddle_ball_distance / max_distance,    # Normalized distance
            relative_position                       # Relative position (-1 to 1)
        ])
        
        # 4. BLOCK INFORMATION (3 inputs)
        if blocks:
            # Nearest block to ball
            nearest_block = min(blocks, key=lambda b: 
                math.sqrt((b.position.x - ball.position.x)**2 + (b.position.y - ball.position.y)**2))
            
            nearest_distance = math.sqrt(
                (nearest_block.position.x - ball.position.x)**2 + 
                (nearest_block.position.y - ball.position.y)**2
            )
            
            # Count blocks in each row
            row_counts = [0] * config.BLOCK_ROWS
            for block in blocks:
                if 0 <= block.row < config.BLOCK_ROWS:
                    row_counts[block.row] += 1
            
            # Focus on most valuable row (back rows worth more)
            most_valuable_row = 0
            max_value = 0
            for row in range(config.BLOCK_ROWS):
                row_value = row_counts[row] * (2 ** row)  # Exponential value
                if row_value > max_value:
                    max_value = row_value
                    most_valuable_row = row
                    
            inputs.extend([
                nearest_distance / max_distance,                    # Distance to nearest block
                len(blocks) / (config.BLOCK_ROWS * config.BLOCKS_PER_ROW),  # Completion progress
                most_valuable_row / config.BLOCK_ROWS              # Most valuable row target
            ])
        else:
            inputs.extend([0.0, 1.0, 0.0])  # No blocks left
            
        # 5. STRATEGIC CONTEXT (2 inputs) 
        # Ball direction relative to blocks
        ball_moving_up = 1.0 if ball.velocity.y < 0 else 0.0
        
        # Urgency factor (how close to bottom)
        urgency = ball.position.y / config.GAME_AREA_HEIGHT
        
        inputs.extend([
            ball_moving_up,    # Is ball moving toward blocks
            urgency           # How urgent is the situation
        ])
        
        return inputs
    
    def draw(self) -> None:
        """Draw the paddle with enhanced visuals."""
        x = int(self.position.x - self.width // 2)
        y = config.SCREEN_HEIGHT - self.height
        # Main paddle
        pyxel.rect(x, y, self.width, self.height, config.PADDLE_COLOR)
        # Top highlight
        pyxel.rect(x + 2, y + 2, self.width - 4, 3, 7)  # White highlight
        # Border
        pyxel.rectb(x, y, self.width, self.height, 7)  # White border

class Block:
    """Represents a breakable block."""
    
    def __init__(self, x: int, y: int, row: int = 0, width: int = None, height: int = None):
        self.position = Vector2D(x, y)
        self.width = width or config.BLOCK_WIDTH
        self.height = height or config.BLOCK_HEIGHT
        self.row = row
        self.color = config.BLOCK_COLORS[row % len(config.BLOCK_COLORS)]
        self.destroyed = False
    
    def check_collision(self, ball: Ball) -> bool:
        """Check and handle collision with ball using proper AABB collision detection."""
        if self.destroyed:
            return False
        
        # Expanded AABB collision detection accounting for ball radius
        ball_left = ball.position.x - ball.radius
        ball_right = ball.position.x + ball.radius
        ball_top = ball.position.y - ball.radius
        ball_bottom = ball.position.y + ball.radius
        
        block_left = self.position.x
        block_right = self.position.x + self.width
        block_top = self.position.y
        block_bottom = self.position.y + self.height
        
        # Check if ball and block are overlapping
        if (ball_right >= block_left and ball_left <= block_right and
            ball_bottom >= block_top and ball_top <= block_bottom):
            
            # Calculate overlap distances
            overlap_left = ball_right - block_left
            overlap_right = block_right - ball_left
            overlap_top = ball_bottom - block_top
            overlap_bottom = block_bottom - ball_top
            
            # Find minimum overlap to determine collision side
            min_overlap_x = min(overlap_left, overlap_right)
            min_overlap_y = min(overlap_top, overlap_bottom)
            
            # Determine collision response based on smallest overlap
            if min_overlap_x < min_overlap_y:
                # Horizontal collision (left or right side)
                ball.velocity.x = -ball.velocity.x
                
                # Push ball out of block
                if overlap_left < overlap_right:
                    # Ball hit left side of block
                    ball.position.x = block_left - ball.radius
                else:
                    # Ball hit right side of block
                    ball.position.x = block_right + ball.radius
            else:
                # Vertical collision (top or bottom side)
                ball.velocity.y = -ball.velocity.y
                
                # Push ball out of block
                if overlap_top < overlap_bottom:
                    # Ball hit top side of block
                    ball.position.y = block_top - ball.radius
                else:
                    # Ball hit bottom side of block
                    ball.position.y = block_bottom + ball.radius
            
            # Handle corner case: if overlaps are very similar, it's a corner hit
            if abs(min_overlap_x - min_overlap_y) < ball.radius * 0.5:
                # Corner collision - reflect both components
                ball.velocity.x = -ball.velocity.x
                ball.velocity.y = -ball.velocity.y
                
                # Push ball to nearest corner
                if ball.position.x < self.position.x + self.width / 2:
                    ball.position.x = block_left - ball.radius
                else:
                    ball.position.x = block_right + ball.radius
                    
                if ball.position.y < self.position.y + self.height / 2:
                    ball.position.y = block_top - ball.radius
                else:
                    ball.position.y = block_bottom + ball.radius
            
            self.destroyed = True
            return True
        return False
    
    def draw(self) -> None:
        """Draw the block if not destroyed with enhanced visuals."""
        if not self.destroyed:
            # Main block
            pyxel.rect(int(self.position.x), int(self.position.y), 
                      self.width, self.height, self.color)
            # Border for 3D effect
            pyxel.rectb(int(self.position.x), int(self.position.y), 
                       self.width, self.height, 7)  # White border

class Game:
    """Individual game instance for one AI agent."""
    
    def __init__(self, genome, neat_config):
        self.genome = genome
        self.config = neat_config
        self.state = GameState.PLAYING
        
        # Game objects
        self.ball = Ball(config.GAME_AREA_WIDTH // 2, config.GAME_AREA_HEIGHT // 2)
        self.paddle = Paddle(genome, neat_config)
        self.blocks = self._create_blocks()
        
        # Game statistics
        self.ticks = 0
        self.paddle_hits = 0
        self.blocks_destroyed = 0
        self.initial_blocks = len(self.blocks)
        self.blocks_by_row = [0] * config.BLOCK_ROWS  # Track blocks destroyed per row
        self.consecutive_hits = 0  # Track consecutive paddle hits
        self.max_consecutive_hits = 0
        self.blocks_destroyed_this_hit = 0  # Track blocks destroyed in current paddle hit
        self.combo_scores = []  # Track all combo scores for analysis
        self.best_combo = 0  # Track best combo achieved
    
    def _create_blocks(self) -> List[Block]:
        """Create the grid of blocks."""
        blocks = []
        for row in range(config.BLOCK_ROWS):
            for col in range(config.BLOCKS_PER_ROW):
                x = col * config.BLOCK_WIDTH
                y = row * config.BLOCK_HEIGHT + 50  # Offset from top
                blocks.append(Block(x, y, row))
        return blocks
    
    def _calculate_combo_score(self, blocks_in_combo: int) -> int:
        """Calculate bonus score for multi-block combos."""
        if blocks_in_combo <= 1:
            return 0
        
        # Exponential combo scoring: 2 blocks = 200pts, 3 blocks = 600pts, 4 blocks = 1200pts, etc.
        # Formula: base_combo_score * (combo_size - 1) * combo_size
        base_combo_score = 100
        combo_bonus = base_combo_score * (blocks_in_combo - 1) * blocks_in_combo
        
        # Extra bonus for exceptional combos
        if blocks_in_combo >= 5:
            combo_bonus += 2000  # Massive bonus for 5+ block combos
        elif blocks_in_combo >= 4:
            combo_bonus += 800   # Big bonus for 4 block combos
        elif blocks_in_combo >= 3:
            combo_bonus += 300   # Good bonus for 3 block combos
        
        return combo_bonus
    
    def _finalize_combo_and_calculate_fitness(self) -> None:
        """Finalize any ongoing combo and calculate fitness."""
        # Handle final combo if there's one in progress
        if self.blocks_destroyed_this_hit > 0:
            combo_score = self._calculate_combo_score(self.blocks_destroyed_this_hit)
            self.combo_scores.append(combo_score)
            self.best_combo = max(self.best_combo, self.blocks_destroyed_this_hit)
        
        self._calculate_fitness()
    
    def update(self) -> None:
        """Update game state for one frame."""
        if self.state != GameState.PLAYING:
            return
        
        self.ticks += 1
        
        # Update game objects
        self.ball.update()
        self.ball.handle_wall_collision()
        self.paddle.update(self.ball, self.blocks)
        
        # Check paddle collision
        paddle_hit_this_frame = False
        if self.ball.hits_paddle(self.paddle):
            paddle_hit_this_frame = True
            self.paddle_hits += 1
            self.consecutive_hits += 1
            self.max_consecutive_hits = max(self.max_consecutive_hits, self.consecutive_hits)
            
            # Reset combo counter for new paddle hit
            if self.blocks_destroyed_this_hit > 0:
                # Calculate combo score for previous hit
                combo_score = self._calculate_combo_score(self.blocks_destroyed_this_hit)
                self.combo_scores.append(combo_score)
                self.best_combo = max(self.best_combo, self.blocks_destroyed_this_hit)
            
            self.blocks_destroyed_this_hit = 0  # Reset for new hit
        else:
            self.consecutive_hits = 0
        
        # Check block collisions
        blocks_destroyed_this_frame = 0
        for block in self.blocks:
            if block.check_collision(self.ball):
                self.blocks_destroyed += 1
                blocks_destroyed_this_frame += 1
                # Track which row the block was in (use the block's row attribute)
                if 0 <= block.row < config.BLOCK_ROWS:
                    self.blocks_by_row[block.row] += 1
        
        # Track blocks destroyed in current paddle hit sequence
        if not paddle_hit_this_frame and blocks_destroyed_this_frame > 0:
            self.blocks_destroyed_this_hit += blocks_destroyed_this_frame
        
        # Remove destroyed blocks
        self.blocks = [block for block in self.blocks if not block.destroyed]
        
        # Check game end conditions
        if self.ball.hits_ground():
            self.state = GameState.FINISHED
            self._finalize_combo_and_calculate_fitness()
        elif self.ticks >= config.TIMEOUT_SECONDS * config.FPS:
            self.state = GameState.TIMEOUT
            self._finalize_combo_and_calculate_fitness()
        elif len(self.blocks) == 0:
            self.state = GameState.FINISHED
            self._finalize_combo_and_calculate_fitness()
    
    def _calculate_fitness(self) -> None:
        """Calculate advanced fitness score encouraging full board completion."""
        
        # 1. PROGRESSIVE BLOCK SCORING - Later blocks worth exponentially more
        base_block_score = 0
        for row in range(config.BLOCK_ROWS):
            blocks_in_row = self.blocks_by_row[row]
            # Each row is worth more: row 0 = 50pts, row 1 = 100pts, row 2 = 200pts, etc.
            row_multiplier = 50 * (2 ** row)
            base_block_score += blocks_in_row * row_multiplier
        
        # 2. COMPLETION MULTIPLIER - Exponential bonus based on completion percentage
        completion_percentage = self.blocks_destroyed / self.initial_blocks
        if completion_percentage >= 1.0:
            # MASSIVE bonus for 100% completion
            completion_multiplier = 10.0
            completion_bonus = 10000  # 10,000 point bonus for clearing board
        elif completion_percentage >= 0.9:
            # Big bonus for 90%+ completion
            completion_multiplier = 3.0
            completion_bonus = 2000
        elif completion_percentage >= 0.75:
            # Good bonus for 75%+ completion
            completion_multiplier = 2.0
            completion_bonus = 500
        elif completion_percentage >= 0.5:
            # Small bonus for 50%+ completion
            completion_multiplier = 1.5
            completion_bonus = 100
        else:
            completion_multiplier = 1.0
            completion_bonus = 0
        
        block_score = base_block_score * completion_multiplier
        
        # 3. BALL CONTROL MASTERY - Reward consistent paddle hits
        ball_control_score = 0
        if self.paddle_hits > 0:
            # Base survival score
            ball_control_score = self.paddle_hits * 20
            
            # Bonus for consecutive hits (ball control skill)
            consistency_bonus = self.max_consecutive_hits * 50
            
            # Endurance bonus - keeping ball alive longer
            if self.paddle_hits >= 20:
                endurance_bonus = (self.paddle_hits - 20) * 10
            else:
                endurance_bonus = 0
                
            ball_control_score += consistency_bonus + endurance_bonus
        
        # 3.5. COMBO MASTERY - Reward multi-block destruction per paddle hit
        combo_score = sum(self.combo_scores)  # Total of all combo bonuses
        
        # Additional combo achievement bonuses
        combo_achievement_bonus = 0
        if self.best_combo >= 5:
            combo_achievement_bonus += 5000  # Master combo achievement
        elif self.best_combo >= 4:
            combo_achievement_bonus += 2000  # Expert combo achievement
        elif self.best_combo >= 3:
            combo_achievement_bonus += 500   # Good combo achievement
        
        # Combo consistency bonus - reward players who get multiple combos
        if len(self.combo_scores) >= 3:
            combo_consistency_bonus = len(self.combo_scores) * 200
        else:
            combo_consistency_bonus = 0
        
        total_combo_score = combo_score + combo_achievement_bonus + combo_consistency_bonus
        
        # 4. EFFICIENCY vs THOROUGHNESS BALANCE
        # Remove time pressure for learning, focus on completion
        time_factor = 1.0
        if len(self.blocks) == 0:  # Only give time bonus if completed
            time_efficiency = max(0, (config.TIMEOUT_SECONDS * config.FPS - self.ticks) * 0.5)
        else:
            time_efficiency = 0
        
        # 5. STRATEGIC PLAY REWARDS
        strategic_bonus = 0
        
        # Reward for clearing back rows first (harder strategy)
        back_row_bonus = 0
        for row in range(config.BLOCK_ROWS - 1, -1, -1):  # Start from back
            if self.blocks_by_row[row] > 0:
                # Bonus for reaching difficult back rows
                back_row_bonus += self.blocks_by_row[row] * (row + 1) * 100
        
        # Reward for row completion (clearing entire rows)
        row_completion_bonus = 0
        blocks_per_row = config.BLOCKS_PER_ROW
        for row in range(config.BLOCK_ROWS):
            if self.blocks_by_row[row] == blocks_per_row:
                # Bonus for completely clearing a row
                row_completion_bonus += 1000 * (row + 1)  # Back rows worth more
        
        strategic_bonus = back_row_bonus + row_completion_bonus
        
        # 6. MOVEMENT EFFICIENCY
        # Reduce movement penalty and make it smarter
        if self.paddle_hits > 0:
            # Only penalize if not hitting ball (encourage active play)
            movement_penalty = max(0, self.paddle.total_distance * 0.05)
        else:
            # Bigger penalty for excessive movement without ball contact
            movement_penalty = self.paddle.total_distance * 0.2
        
        # 7. FINAL FITNESS CALCULATION
        total_fitness = (
            block_score +                    # Progressive block scoring with multipliers
            ball_control_score +             # Ball control and consistency
            total_combo_score +              # Multi-block combo rewards
            completion_bonus +               # Major completion bonuses
            strategic_bonus +                # Strategic play rewards
            time_efficiency -                # Time efficiency (only if completed)
            movement_penalty                 # Smart movement penalty
        )
        
        # 8. LEARNING STAGE BONUS - Extra reward for first-time achievements
        learning_bonus = 0
        if completion_percentage >= 0.25:  # First quarter completion
            learning_bonus += 500
        if completion_percentage >= 0.5:   # Half completion
            learning_bonus += 1000
        if completion_percentage >= 0.75:  # Three quarters
            learning_bonus += 2000
        if completion_percentage >= 0.9:   # Nearly complete
            learning_bonus += 5000
        
        self.genome.fitness = total_fitness + learning_bonus
    
    def draw(self) -> None:
        """Draw all game objects."""
        self.ball.draw()
        self.paddle.draw()
        for block in self.blocks:
            block.draw()

class NeuralNetworkVisualizer:
    """Visualizes neural network structure and real-time activations."""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.input_labels = [
            "Paddle X", "Ball X", "Ball Y", "Ball Vel X", "Ball Vel Y",           # Basic (5)
            "Future X", "Future Y", "Predict X",                                  # Trajectory (3) 
            "Distance", "Relative",                                               # Spatial (2)
            "Nearest Block", "Progress", "Target Row",                            # Blocks (3)
            "Moving Up", "Urgency"                                               # Context (2)
        ]
        self.output_labels = ["Move"]
        
    def draw_network(self, brain, inputs: List[float], outputs: List[float], hidden_activations: dict = None) -> None:
        """Draw the neural network with real-time activations."""
        # Background
        pyxel.rect(self.x, self.y, self.width, self.height, config.GAME_BACKGROUND_COLOR)
        pyxel.rectb(self.x, self.y, self.width, self.height, config.TEXT_COLOR)
        
        # Title
        pyxel.text(self.x + 5, self.y + 5, "NEURAL NETWORK", config.ACCENT_COLOR)
        
        # Get network structure
        input_nodes = []
        hidden_nodes = []
        output_nodes = []
        
        # Categorize nodes based on their keys (NEAT uses negative keys for inputs, 0 for bias, positive for hidden/output)
        if hasattr(brain, 'nodes'):
            # Determine which nodes are actually connected and used
            connected_nodes = set()
            if hasattr(brain, 'connections'):
                for conn in brain.connections.values():
                    if conn.enabled:
                        connected_nodes.add(conn.key[0])
                        connected_nodes.add(conn.key[1])
            
            for key, node in brain.nodes.items():
                if key < 0:  # Input nodes
                    input_nodes.append((key, node))
                elif key == 0:  # Bias node - treat as input
                    input_nodes.append((key, node))
                else:  # Positive keys - hidden and output nodes
                    # Check if this is the output node (usually the highest key or specifically key 1)
                    max_key = max([k for k in brain.nodes.keys() if k > 0])
                    if key == 1 or key == max_key:  # Output node
                        output_nodes.append((key, node))
                    else:  # Hidden node
                        if key in connected_nodes:  # Only show connected hidden nodes
                            hidden_nodes.append((key, node))
        
        # If we can't determine structure, use simple layout
        if not input_nodes and not output_nodes:
            self._draw_simple_network(inputs, outputs)
            return
            
        # Calculate positions for different layers
        layer_width = self.width - 20
        layer_spacing = layer_width // 3
        
        input_x = self.x + 20
        hidden_x = self.x + 20 + layer_spacing
        output_x = self.x + 20 + 2 * layer_spacing
        
        # Draw connections first (so they appear behind nodes)
        if hasattr(brain, 'connections'):
            self._draw_connections(brain.connections, input_nodes, hidden_nodes, output_nodes,
                                 input_x, hidden_x, output_x)
        
        # Draw input nodes
        input_y_start = self.y + 30
        input_spacing = min(25, (self.height - 60) // max(len(input_nodes), 1))
        for i, (key, node) in enumerate(input_nodes):
            y_pos = input_y_start + i * input_spacing
            activation = inputs[i] if i < len(inputs) else 0.0
            self._draw_node(input_x, y_pos, activation, "input", 
                          self.input_labels[i] if i < len(self.input_labels) else f"In{i}")
        
        # Draw hidden nodes
        if hidden_nodes:
            hidden_y_start = self.y + 30
            hidden_spacing = min(25, (self.height - 60) // max(len(hidden_nodes), 1))
            for i, (key, node) in enumerate(hidden_nodes):
                y_pos = hidden_y_start + i * hidden_spacing
                # Use tracked activation if available
                activation = hidden_activations.get(key, 0.0) if hidden_activations else 0.0
                self._draw_node(hidden_x, y_pos, activation, "hidden", f"H{key}")
        
        # Draw output nodes
        output_y_start = self.y + 30
        output_spacing = min(25, (self.height - 60) // max(len(output_nodes), 1))
        for i, (key, node) in enumerate(output_nodes):
            y_pos = output_y_start + i * output_spacing
            activation = outputs[i] if i < len(outputs) else 0.0
            self._draw_node(output_x, y_pos, activation, "output",
                          self.output_labels[i] if i < len(self.output_labels) else f"Out{i}")
    
    def _draw_simple_network(self, inputs: List[float], outputs: List[float]) -> None:
        """Draw a simplified network visualization when structure is unknown."""
        # Input layer
        input_x = self.x + 20
        input_y_start = self.y + 30
        input_spacing = min(25, (self.height - 60) // max(len(inputs), 1))
        
        for i, activation in enumerate(inputs):
            y_pos = input_y_start + i * input_spacing
            self._draw_node(input_x, y_pos, activation, "input",
                          self.input_labels[i] if i < len(self.input_labels) else f"In{i}")
        
        # Output layer
        output_x = self.x + self.width - 40
        output_y_start = self.y + 30
        output_spacing = min(25, (self.height - 60) // max(len(outputs), 1))
        
        for i, activation in enumerate(outputs):
            y_pos = output_y_start + i * output_spacing
            self._draw_node(output_x, y_pos, activation, "output",
                          self.output_labels[i] if i < len(self.output_labels) else f"Out{i}")
        
        # Draw connections (simplified)
        for i, input_activation in enumerate(inputs):
            input_y = input_y_start + i * input_spacing
            for j, output_activation in enumerate(outputs):
                output_y = output_y_start + j * output_spacing
                # Color connection based on activation strength
                connection_strength = abs(input_activation * output_activation)
                color = self._get_activation_color(connection_strength)
                pyxel.line(input_x + 8, input_y, output_x - 8, output_y, color)
    
    def _draw_connections(self, connections, input_nodes, hidden_nodes, output_nodes,
                         input_x, hidden_x, output_x) -> None:
        """Draw connections between nodes."""
        # Create position lookup
        node_positions = {}
        
        # Input positions
        input_y_start = self.y + 30
        input_spacing = min(25, (self.height - 60) // max(len(input_nodes), 1))
        for i, (key, node) in enumerate(input_nodes):
            node_positions[key] = (input_x, input_y_start + i * input_spacing)
        
        # Hidden positions
        if hidden_nodes:
            hidden_y_start = self.y + 30
            hidden_spacing = min(25, (self.height - 60) // max(len(hidden_nodes), 1))
            for i, (key, node) in enumerate(hidden_nodes):
                node_positions[key] = (hidden_x, hidden_y_start + i * hidden_spacing)
        
        # Output positions
        output_y_start = self.y + 30
        output_spacing = min(25, (self.height - 60) // max(len(output_nodes), 1))
        for i, (key, node) in enumerate(output_nodes):
            node_positions[key] = (output_x, output_y_start + i * output_spacing)
        
        # Draw connections
        for conn in connections.values():
            if not conn.enabled:
                continue
                
            if conn.key[0] in node_positions and conn.key[1] in node_positions:
                x1, y1 = node_positions[conn.key[0]]
                x2, y2 = node_positions[conn.key[1]]
                
                # Color and thickness based on weight
                weight_strength = abs(conn.weight)
                if conn.weight > 0:
                    color = config.SUCCESS_COLOR  # Green for positive weights
                else:
                    color = config.WARNING_COLOR   # Red for negative weights
                
                # Draw thicker lines for stronger connections
                if weight_strength > 2.0:
                    # Very strong connection - draw multiple lines
                    pyxel.line(x1 + 8, y1, x2 - 8, y2, color)
                    pyxel.line(x1 + 8, y1 + 1, x2 - 8, y2 + 1, color)
                    pyxel.line(x1 + 8, y1 - 1, x2 - 8, y2 - 1, color)
                elif weight_strength > 1.0:
                    # Strong connection - draw double line
                    pyxel.line(x1 + 8, y1, x2 - 8, y2, color)
                    pyxel.line(x1 + 8, y1 + 1, x2 - 8, y2 + 1, color)
                elif weight_strength > 0.5:
                    # Normal connection
                    pyxel.line(x1 + 8, y1, x2 - 8, y2, color)
                else:
                    # Weak connection - dimmed color
                    pyxel.line(x1 + 8, y1, x2 - 8, y2, 5)  # Dark gray
    
    def _draw_node(self, x: int, y: int, activation: float, node_type: str, label: str) -> None:
        """Draw a single neuron node with enhanced activation visualization."""
        # Node size based on activation
        base_radius = 8
        activation_radius = int(base_radius + abs(activation) * 4)
        
        # Color based on activation and type
        if node_type == "input":
            base_color = config.ACCENT_COLOR
        elif node_type == "output":
            base_color = config.SUCCESS_COLOR
        else:  # hidden
            base_color = 14  # Light gray for hidden nodes
        
        # Draw node with activation intensity
        activation_color = self._get_activation_color(abs(activation))
        
        # Draw pulsing effect for high activation
        if abs(activation) > 0.7:
            # Draw outer glow for high activation
            pyxel.circ(x, y, activation_radius + 2, activation_color)
            pyxel.circ(x, y, activation_radius, 7)  # White core
        else:
            pyxel.circ(x, y, activation_radius, activation_color)
        
        # Draw border
        pyxel.circb(x, y, base_radius, base_color)
        
        # Draw activation value as small text
        activation_text = f"{activation:.2f}"
        text_x = x - len(activation_text) * 2
        text_color = config.TEXT_COLOR if abs(activation) < 0.5 else 7  # White for high activation
        pyxel.text(text_x, y - 18, activation_text, text_color)
        
        # Draw label (shorter for space)
        short_label = label[:6] if len(label) > 6 else label
        label_x = x - len(short_label) * 2
        pyxel.text(label_x, y + 12, short_label, config.TEXT_COLOR)
        
        # Draw activation direction indicator for hidden/output nodes
        if node_type != "input" and abs(activation) > 0.1:
            if activation > 0:
                # Positive activation - small up arrow
                pyxel.pset(x, y - 3, 7)
                pyxel.pset(x - 1, y - 2, 7)
                pyxel.pset(x + 1, y - 2, 7)
            else:
                # Negative activation - small down arrow
                pyxel.pset(x, y + 3, 8)
                pyxel.pset(x - 1, y + 2, 8)
                pyxel.pset(x + 1, y + 2, 8)
    
    def _get_activation_color(self, activation: float) -> int:
        """Get color based on activation strength."""
        if activation > 0.9:
            return 8   # Bright red - very high activation
        elif activation > 0.7:
            return 9   # Orange - high activation
        elif activation > 0.5:
            return 10  # Yellow - medium-high activation
        elif activation > 0.3:
            return 11  # Light green - medium activation
        elif activation > 0.1:
            return 12  # Blue - low activation
        else:
            return 5   # Dark gray - very low activation

class GameVisualizer:
    """Handles enhanced visualization of multiple games running simultaneously."""
    
    def __init__(self, genomes, neat_config, generation: int):
        self.games = [Game(genome, neat_config) for genome_id, genome in genomes]
        self.generation = generation
        self.finished_count = 0
        self.start_time = time.time()
        self.fitness_history = []
        
        # Neural network visualizer
        self.neural_viz = NeuralNetworkVisualizer(
            config.NEURAL_VIZ_X, config.NEURAL_VIZ_Y,
            config.NEURAL_VIZ_WIDTH, config.NEURAL_VIZ_HEIGHT
        )
        
    def update(self) -> None:
        """Update all active games."""
        self.finished_count = 0
        for game in self.games:
            game.update()
            if game.state != GameState.PLAYING:
                self.finished_count += 1
    
    def draw(self) -> None:
        """Draw enhanced visualization with game area and detailed statistics."""
        # Clear screen with background
        pyxel.cls(config.BACKGROUND_COLOR)
        
        # Draw game area background
        pyxel.rect(0, 0, config.GAME_AREA_WIDTH, config.GAME_AREA_HEIGHT, 
                  config.GAME_BACKGROUND_COLOR)
        
        # Draw game area border
        pyxel.rectb(0, 0, config.GAME_AREA_WIDTH, config.GAME_AREA_HEIGHT, 
                   config.TEXT_COLOR)
        
        # Find and draw the best performing game
        active_games = [g for g in self.games if g.state == GameState.PLAYING]
        if active_games:
            best_game = max(active_games, 
                          key=lambda g: g.blocks_destroyed * 100 + g.paddle_hits)
            best_game.draw()
        elif self.games:  # If no active games, show the best finished game
            best_game = max(self.games, key=lambda g: g.genome.fitness or 0)
            best_game.draw()
        
        # Draw comprehensive statistics panel
        self._draw_enhanced_statistics()
        
        # Draw performance graphs
        self._draw_fitness_graph()
        
        # Draw neural network legend
        self._draw_neural_legend()
        
        # Draw neural network visualization for the best game
        if active_games or self.games:
            best_game = None
            if active_games:
                best_game = max(active_games, 
                              key=lambda g: g.blocks_destroyed * 100 + g.paddle_hits)
            else:
                best_game = max(self.games, key=lambda g: g.genome.fitness or 0)
            
            if best_game:
                self.neural_viz.draw_network(
                    best_game.paddle.brain,
                    best_game.paddle.last_inputs,
                    best_game.paddle.last_outputs,
                    best_game.paddle.hidden_activations
                )
    
    def _draw_enhanced_statistics(self) -> None:
        """Draw comprehensive statistics panel."""
        x_start = config.STATS_PANEL_X + 10
        y = 20
        line_height = 20
        
        # Title
        pyxel.text(x_start, y, "=== AI TRAINING STATS ===", config.ACCENT_COLOR)
        y += line_height + 10
        
        # Generation info
        pyxel.text(x_start, y, f"Generation: {self.generation}", config.TEXT_COLOR)
        y += line_height
        
        # Time elapsed
        elapsed = time.time() - self.start_time
        pyxel.text(x_start, y, f"Time: {elapsed:.1f}s", config.TEXT_COLOR)
        y += line_height + 5
        
        # Calculate current statistics
        fitnesses = [game.genome.fitness or 0 for game in self.games]
        max_fitness = max(fitnesses) if fitnesses else 0
        avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0
        
        # Fitness statistics
        pyxel.text(x_start, y, "FITNESS SCORES:", config.ACCENT_COLOR)
        y += line_height
        pyxel.text(x_start, y, f"Max: {max_fitness:.1f}", 
                  config.SUCCESS_COLOR if max_fitness > 10000 else config.TEXT_COLOR)
        y += line_height
        pyxel.text(x_start, y, f"Avg: {avg_fitness:.1f}", config.TEXT_COLOR)
        y += line_height + 5
        
        # Progress statistics
        pyxel.text(x_start, y, "PROGRESS:", config.ACCENT_COLOR)
        y += line_height
        active_count = len([g for g in self.games if g.state == GameState.PLAYING])
        pyxel.text(x_start, y, f"Active: {active_count}", config.TEXT_COLOR)
        y += line_height
        pyxel.text(x_start, y, f"Finished: {self.finished_count}/{len(self.games)}", 
                  config.TEXT_COLOR)
        y += line_height + 5
        
        # Best game statistics
        if self.games:
            best_game = max(self.games, key=lambda g: g.genome.fitness or 0)
            pyxel.text(x_start, y, "BEST PERFORMER:", config.ACCENT_COLOR)
            y += line_height
            pyxel.text(x_start, y, f"Blocks: {best_game.blocks_destroyed}/{best_game.initial_blocks}", 
                      config.TEXT_COLOR)
            y += line_height
            completion = (best_game.blocks_destroyed / best_game.initial_blocks) * 100
            color = config.SUCCESS_COLOR if completion >= 90 else config.WARNING_COLOR if completion < 50 else config.TEXT_COLOR
            pyxel.text(x_start, y, f"Complete: {completion:.1f}%", color)
            y += line_height
            pyxel.text(x_start, y, f"Paddle Hits: {best_game.paddle_hits}", config.TEXT_COLOR)
            y += line_height
            pyxel.text(x_start, y, f"Max Chain: {best_game.max_consecutive_hits}", config.TEXT_COLOR)
            y += line_height
            pyxel.text(x_start, y, f"Best Combo: {best_game.best_combo}", config.SUCCESS_COLOR)
            y += line_height
            pyxel.text(x_start, y, f"Combos: {len(best_game.combo_scores)}", config.SUCCESS_COLOR)
            y += line_height + 5
            
            # Network complexity
            if hasattr(best_game.paddle.brain, 'nodes'):
                node_count = len(best_game.paddle.brain.nodes)
                conn_count = len(best_game.paddle.brain.connections)
                pyxel.text(x_start, y, "NETWORK:", config.ACCENT_COLOR)
                y += line_height
                pyxel.text(x_start, y, f"Nodes: {node_count}", config.TEXT_COLOR)
                y += line_height
                pyxel.text(x_start, y, f"Connections: {conn_count}", config.TEXT_COLOR)
                y += line_height + 5
        
        # Row completion analysis (compact version)
        if self.games and y < config.NEURAL_VIZ_Y - 20:  # Make sure we don't overlap with neural viz
            best_game = max(self.games, key=lambda g: g.genome.fitness or 0)
            pyxel.text(x_start, y, "ROW PROGRESS:", config.ACCENT_COLOR)
            y += line_height
            for row in range(min(3, config.BLOCK_ROWS)):  # Show only first 3 rows to save space
                destroyed = best_game.blocks_by_row[row]
                total = config.BLOCKS_PER_ROW
                percentage = (destroyed / total) * 100 if total > 0 else 0
                color = config.SUCCESS_COLOR if percentage >= 100 else config.TEXT_COLOR
                pyxel.text(x_start, y, f"R{row}: {destroyed}/{total}", color)
                y += line_height
    
    def _draw_fitness_graph(self) -> None:
        """Draw a simple fitness progression graph."""
        if len(self.fitness_history) < 2:
            return
            
        graph_x = config.STATS_PANEL_X + 10
        graph_y = 450
        graph_width = 200
        graph_height = 80
        
        # Draw graph background
        pyxel.rect(graph_x, graph_y, graph_width, graph_height, config.GAME_BACKGROUND_COLOR)
        pyxel.rectb(graph_x, graph_y, graph_width, graph_height, config.TEXT_COLOR)
        
        # Draw fitness progression line
        if len(self.fitness_history) > 1:
            max_fitness = max(self.fitness_history) if self.fitness_history else 1
            for i in range(1, len(self.fitness_history)):
                x1 = graph_x + int((i-1) * graph_width / len(self.fitness_history))
                y1 = graph_y + graph_height - int((self.fitness_history[i-1] / max_fitness) * graph_height)
                x2 = graph_x + int(i * graph_width / len(self.fitness_history))
                y2 = graph_y + graph_height - int((self.fitness_history[i] / max_fitness) * graph_height)
                pyxel.line(x1, y1, x2, y2, config.ACCENT_COLOR)
        
        # Graph title
        pyxel.text(graph_x, graph_y - 15, "FITNESS HISTORY", config.ACCENT_COLOR)
    
    def _draw_neural_legend(self) -> None:
        """Draw legend for neural network visualization."""
        legend_x = config.STATS_PANEL_X + 10
        legend_y = 520
        
        pyxel.text(legend_x, legend_y, "NEURAL LEGEND:", config.ACCENT_COLOR)
        legend_y += 15
        
        # Connection colors
        pyxel.text(legend_x, legend_y, "Connections:", config.TEXT_COLOR)
        legend_y += 10
        pyxel.line(legend_x, legend_y, legend_x + 15, legend_y, config.SUCCESS_COLOR)
        pyxel.text(legend_x + 20, legend_y - 2, "Positive", config.TEXT_COLOR)
        legend_y += 8
        pyxel.line(legend_x, legend_y, legend_x + 15, legend_y, config.WARNING_COLOR)
        pyxel.text(legend_x + 20, legend_y - 2, "Negative", config.TEXT_COLOR)
        legend_y += 12
        
        # Activation levels
        pyxel.text(legend_x, legend_y, "Activation:", config.TEXT_COLOR)
        legend_y += 10
        colors = [8, 10, 11, 12, 5]
        labels = ["High", "Med", "Low", "VLow", "Off"]
        for i, (color, label) in enumerate(zip(colors, labels)):
            pyxel.circ(legend_x + i * 25, legend_y + 5, 3, color)
            pyxel.text(legend_x + i * 25 - 6, legend_y + 12, label, config.TEXT_COLOR)
    
    def all_finished(self) -> bool:
        """Check if all games are finished."""
        if self.finished_count >= len(self.games):
            # Record fitness for history
            fitnesses = [game.genome.fitness or 0 for game in self.games]
            max_fitness = max(fitnesses) if fitnesses else 0
            self.fitness_history.append(max_fitness)
            return True
        return False

def evaluate_genomes(genomes, neat_config, generation: int) -> None:
    """Evaluate a generation of genomes."""
    visualizer = GameVisualizer(genomes, neat_config, generation)
    
    # Run until all games are finished
    while not visualizer.all_finished():
        visualizer.update()
        visualizer.draw()
        pyxel.flip()

def run_neat_algorithm(config_file: str) -> None:
    """Run the NEAT algorithm to evolve Breakout AI."""
    try:
        # Load NEAT configuration
        neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
        
        # Create population
        population = neat.Population(neat_config)
        
        # Add reporters for logging
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)
        population.add_reporter(neat.Checkpointer(5))
        
        # Run evolution
        def eval_with_generation(genomes, config):
            evaluate_genomes(genomes, config, population.generation)
        
        winner = population.run(eval_with_generation, 200)
        
        # Save the winner
        with open("winner.pkl", "wb") as f:
            pickle.dump(winner, f)
        
        logger.info(f"Evolution completed. Winner saved to winner.pkl")
        logger.info(f"Winner fitness: {winner.fitness}")
        
    except Exception as e:
        logger.error(f"Error running NEAT algorithm: {e}")
        raise

def main():
    """Main entry point."""
    try:
        # Get configuration file path
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-feedforward.txt')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Initialize Pyxel with enhanced title
        pyxel.init(config.SCREEN_WIDTH, config.SCREEN_HEIGHT, 
                  title="🎮 Breakout AI - Advanced NEAT Evolution", fps=config.FPS)
        
        logger.info("Starting Breakout AI evolution...")
        run_neat_algorithm(config_path)
        
    except KeyboardInterrupt:
        logger.info("Evolution interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
