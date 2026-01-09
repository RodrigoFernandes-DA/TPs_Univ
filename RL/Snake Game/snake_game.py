import numpy as np
from collections import deque
import pygame
import cv2
import math
import os

import gymnasium as gym
from gymnasium import spaces

# Simple 2D point structure for coordinates
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def copy(self, xincr=0, yincr=0):
        return Point(self.x + xincr, self.y + yincr)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

# Action direction maps from codeA.py
action_dir_map = {
    'right': Point(1, 0),
    'left': Point(-1, 0),
    'up': Point(0, -1),
    'down': Point(0, 1),
}

# Maps direction vector to a rotation angle for sprite drawing
dir_map_to_angle = {
    Point(0, -1): 0,    # left
    Point(0, 1): 180,   # right
    Point(-1, 0): -90,   # up
    Point(1, 0): 90,   # down
}

dir_map_to_angle_head = {
    Point(0, -1): -90,    # left
    Point(0, 1): 90,   # right
    Point(-1, 0): 0,   # up
    Point(1, 0): 180,   # down
}

class SnakeGameEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 20,
    }

    def __init__(self, render_mode=None, n_channel=1, board_size=15, n_target=1):
        assert board_size >= 5
        assert n_target > 0
        assert n_channel in (1, 2, 4)

        self.BLANK = 0
        self.ITEM = board_size**2 + 1
        self.HEAD = 1
        self.n_channel = n_channel

        self.color_gradient = (255 - 100) / (board_size**2)

        self.board_size = board_size  # The size of the square grid
        self.window_width = 600  # The size of the PyGame window
        self.window_height = 700
        self.window_diff = self.window_height - self.window_width
        self.n_target = n_target
        
        # space
        self.observation_space = spaces.Box(
            low=0,
            high=self.ITEM,
            shape=(self.n_channel, board_size, board_size),
            dtype=np.uint32,
        )
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        
        # initialize
        self.snake = deque()
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.uint32)

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Background color 
        self.COLOR_BG = (255, 211, 140)  
        self.COLOR_GRID = (80, 80, 80)  # Slightly lighter gray for grid
        self.COLOR_TEXT = (240, 240, 240)
        
        # === Load sprites ===
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sprite_dir = os.path.join(base_dir, "sprites")
        
        # Try to load sprites
        sprite_paths = {
            'head': os.path.join(base_dir, "RL/Snake Game/codes/sprites/head.png"),
            'body': os.path.join(base_dir, "RL/Snake Game/codes/sprites/body.png"),
            'turn': os.path.join(base_dir, "RL/Snake Game/codes/sprites/turn.png"),
            'fruit': os.path.join(base_dir, "RL/Snake Game/codes/sprites/fruit.png"),
            'tail': os.path.join(base_dir, "RL/Snake Game/codes/sprites/tail.png"),
        }
        
        # Fallback to local sprites directory
        for key in sprite_paths:
            if not os.path.exists(sprite_paths[key]):
                sprite_paths[key] = os.path.join(sprite_dir, f"{key}.png")
        
        # Load sprites 
        self.sprites = {}
        for key, path in sprite_paths.items():
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.sprites[key] = img
                else:
                    print(f"[SPRITE WARNING] Could not load {path}, using fallback")
                    self.sprites[key] = self._create_fallback_sprite(key, color=True)
            else:
                print(f"[SPRITE WARNING] File not found: {path}")
                self.sprites[key] = self._create_fallback_sprite(key, color=True)
        
        # Convert sprites to pygame 
        self.pygame_sprites = {}
        for key, img in self.sprites.items():
            if img is not None:
                if len(img.shape) == 2:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    rgb_img = img
                
                # Convert to pygame surface
                self.pygame_sprites[key] = pygame.surfarray.make_surface(
                    np.transpose(rgb_img, (1, 0, 2))
                )
        
        # Initialize snake direction tracking for sprite rotation
        self.snake_direction = Point(1, 0)  # Start moving right as in codeA.py
        self.prev_action = 1  # Start facing right
        
        # Snake segments for rendering (head + tail positions)
        self.snake_segments = []

    def _create_fallback_sprite(self, sprite_type):
        """Create a simple fallback sprite if the image file is not found"""
        size = 32
        img = np.zeros((size, size), dtype=np.uint8)
        
        if sprite_type == 'head':
            # Create a simple head shape
            cv2.circle(img, (size//2, size//2), size//3, 255, -1)
        elif sprite_type == 'body':
            # Create a simple body segment
            cv2.rectangle(img, (size//4, size//4), (3*size//4, 3*size//4), 200, -1)
        elif sprite_type == 'turn':
            # Create a corner piece
            cv2.ellipse(img, (size//2, size//2), (size//3, size//3), 0, 0, 90, 220, -1)
        elif sprite_type == 'fruit':
            # Create a fruit/apple shape
            cv2.circle(img, (size//2, size//2), size//3, 150, -1)
        elif sprite_type == 'tail':
            # Create a tail piece
            cv2.rectangle(img, (size//3, size//3), (2*size//3, 2*size//3), 180, -1)
        
        return img

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset
        self.board.fill(0)
        self.snake.clear()
        for i in range(3):
            self.snake.appendleft(np.array([self.board_size // 2, self.board_size // 2 - i]))
        for i, (x, y) in enumerate(self.snake):
            self.board[x, y] = len(self.snake) - i

        self._place_target(initial=True)

        # update iteration
        self._n_step = 0
        self._score = 0
        self.prev_action = 1
        self.snake_direction = Point(1, 0)  # Start moving right

        # Build snake segments for rendering
        self._update_snake_segments()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _update_snake_segments(self):
        """Convert snake deque to list of Points for rendering"""
        self.snake_segments = []
        for segment in self.snake:
            self.snake_segments.append(Point(segment[1], segment[0]))  # Note: x,y swap for rendering
        
        # Reverse to get head first as in codeA.py
        self.snake_segments = list(reversed(self.snake_segments))

    def _place_target(self, initial: bool = False) -> None:
        target_candidate = np.argwhere(self.board == self.BLANK)
        if initial:
            target_list = target_candidate[self.np_random.choice(len(target_candidate), self.n_target)]
            for x, y in target_list:
                self.board[x, y] = self.ITEM
        else:
            if target_candidate.size == 0:
                return
            else:
                new_target = target_candidate[self.np_random.choice(len(target_candidate))]
                self.board[new_target[0], new_target[1]] = self.ITEM

    def _get_obs(self):
        if self.n_channel == 1:
            return self.board[np.newaxis, :, :]
        else:
            return self._split_channel(self.n_channel)

    def _split_channel(self, n_channel):
        if n_channel == 2:
            mask = self.board == self.ITEM
            snake_obs = np.where(mask, 0, self.board)
            target_obs = np.where(mask, self.board, 0)
            return np.array([snake_obs, target_obs])
        # n_channel == 4
        else:
            channels = []
            # body
            mask = (1 < self.board) & (self.board < len(self.snake))
            channel = np.where(mask, self.board, 0)
            channels.append(channel)

            # head, tail, target
            without_body = (1, len(self.snake), self.ITEM)
            for element in without_body:
                mask = self.board == element
                channel = np.where(mask, self.board, 0)
                channels.append(channel)

            return np.array(channels)

    def _get_info(self):
        return {"snake_length": len(self.snake), "prev_action": self.prev_action}

    # def step(self, action: int):
    #     direction = self._action_to_direction[action]
        
    #     # Update snake direction for sprite rotation
    #     dir_mapping = {
    #         0: Point(1, 0),   # right
    #         1: Point(0, 1),   # down
    #         2: Point(-1, 0),  # left
    #         3: Point(0, -1),  # up
    #     }
    #     self.snake_direction = dir_mapping[action]

    #     # update iteration
    #     self._n_step += 1

    #     current_head = self.snake[-1]
    #     current_tail = self.snake[0]
    #     next_head = current_head + direction

    #     if np.array_equal(next_head, self.snake[-2]):
    #         next_head = current_head - direction

    #     # get out the board
    #     if not (0 <= next_head[0] < self.board_size and 0 <= next_head[1] < self.board_size):
    #         reward = -10
    #         terminated = True
    #     # hit the snake
    #     elif 0 < self.board[next_head[0], next_head[1]] < self.ITEM:
    #         reward = -10
    #         terminated = True
    #     else:
    #         # blank
    #         if self.board[next_head[0], next_head[1]] == self.BLANK:
    #             self.board[current_tail[0], current_tail[1]] = self.BLANK
    #             self.snake.popleft()
    #             reward = 0
    #             terminated = False
    #         # target
    #         # self.board[next_head[0], next_head[1]] == self.ITEM
    #         else:
    #             self._score += 1
    #             reward = 10
    #             self._place_target()
    #             self.board[next_head[0], next_head[1]] = 0
    #             if len(self.snake) == self.board_size**2:
    #                 terminated = True
    #             else:
    #                 terminated = False
    #         self.snake.append(next_head)
    #         for x, y in self.snake:
    #             self.board[x][y] += 1

    #     # Update snake segments for rendering
    #     self._update_snake_segments()

    #     observation = self._get_obs()
    #     info = self._get_info()

    #     if self.render_mode == "human":
    #         self.render()

    #     self.prev_action = action

    #     return observation, reward, terminated, False, info
    def step(self, action: int):
        direction = self._action_to_direction[action]
        
        # Update snake direction for sprite rotation
        dir_mapping = {
            0: Point(1, 0),   # right
            1: Point(0, 1),   # down
            2: Point(-1, 0),  # left
            3: Point(0, -1),  # up
        }
        self.snake_direction = dir_mapping[action]

        # update iteration
        self._n_step += 1

        current_head = self.snake[-1]
        current_tail = self.snake[0]
        next_head = current_head + direction

        if np.array_equal(next_head, self.snake[-2]):
            next_head = current_head - direction

        # Get the current head position and find target position
        current_head_pos = Point(current_head[1], current_head[0])
        target_positions = np.argwhere(self.board == self.ITEM)
        
        # Calculate current distance to nearest target (before moving)
        current_min_distance = float('inf')
        if len(target_positions) > 0:
            for target in target_positions:
                target_pos = Point(target[1], target[0])
                # Euclidean distance
                distance = math.sqrt((current_head_pos.x - target_pos.x)**2 + 
                                   (current_head_pos.y - target_pos.y)**2)
                current_min_distance = min(current_min_distance, distance)

        # get out the board
        if not (0 <= next_head[0] < self.board_size and 0 <= next_head[1] < self.board_size):
            reward = -10
            terminated = True
        # hit the snake
        elif 0 < self.board[next_head[0], next_head[1]] < self.ITEM:
            reward = -10
            terminated = True
        else:
            # Calculate new distance to nearest target (after moving)
            next_head_pos = Point(next_head[1], next_head[0])
            new_min_distance = float('inf')
            if len(target_positions) > 0:
                for target in target_positions:
                    target_pos = Point(target[1], target[0])
                    distance = math.sqrt((next_head_pos.x - target_pos.x)**2 + 
                                       (next_head_pos.y - target_pos.y)**2)
                    new_min_distance = min(new_min_distance, distance)
            
            # Base reward
            base_reward = 0
            terminated = False
            
            # blank
            if self.board[next_head[0], next_head[1]] == self.BLANK:
                self.board[current_tail[0], current_tail[1]] = self.BLANK
                self.snake.popleft()
                base_reward = 0
                terminated = False
            # target
            else:
                self._score += 1
                base_reward = 10
                self._place_target()
                self.board[next_head[0], next_head[1]] = 0
                if len(self.snake) == self.board_size**2:
                    terminated = True
                else:
                    terminated = False
            
            # Calculate distance-based reward/penalty
            distance_reward = 0
            if len(target_positions) > 0 and current_min_distance < float('inf'):
                if new_min_distance < current_min_distance:
                    # Moved closer to target - positive reward
                    distance_reward = 0.1 * (current_min_distance - new_min_distance)
                elif new_min_distance > current_min_distance:
                    # Moved away from target - negative reward (penalty)
                    distance_reward = -0.1 * (new_min_distance - current_min_distance)
                # If equal distance, no additional reward/penalty
            
            # Total reward
            reward = base_reward + distance_reward
            
            # Update snake position
            self.snake.append(next_head)
            for x, y in self.snake:
                self.board[x][y] += 1

        # Update snake segments for rendering
        self._update_snake_segments()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        self.prev_action = action

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode in {"rgb_array", "human"}:
            return self._render_frame()

    def _render_frame(self):
        pygame.font.init()

        # Initialize pygame window if needed
        if self.window is None:
            pygame.init()
            self.square_size = self.window_width // self.board_size
            self.font_size = self.window_diff // 3
            
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_width, self.window_height))
            else:
                self.window = pygame.Surface((self.window_width, self.window_height))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Create canvas
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill(self.COLOR_BG)

        # Draw score and step text
        myFont = pygame.font.SysFont("consolas", self.font_size, bold=True)
        score_render_text = myFont.render(f"score: {self._score}", True, self.COLOR_TEXT)
        n_step_render_text = myFont.render(f"step: {self._n_step}", True, self.COLOR_TEXT)

        canvas.blit(
            score_render_text,
            (self.window_width // 30 * 1, self.window_diff // 2 - self.font_size // 2),
        )
        canvas.blit(
            n_step_render_text,
            (self.window_width // 30 * 15, self.window_diff // 2 - self.font_size // 2),
        )

        # Draw the game board with sprites
        self._draw_board_with_sprites(canvas)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def _draw_board_with_sprites(self, canvas):
        """Draw the board using sprites from codeA.py"""
        
        # Helper function to draw a sprite with rotation
        def draw_sprite(canvas, grid_x, grid_y, sprite_type, rotation=0):
            """Draw a sprite at the given grid position with rotation"""
            if sprite_type not in self.pygame_sprites:
                return
                
            sprite_surface = self.pygame_sprites[sprite_type]
            
            # Apply rotation if needed
            if rotation != 0:
                sprite_surface = pygame.transform.rotate(sprite_surface, rotation)
            
            # Calculate screen position
            screen_x = grid_x * self.square_size
            screen_y = self.window_diff + grid_y * self.square_size
            
            # Scale sprite to fit the cell
            scaled_sprite = pygame.transform.smoothscale(
                sprite_surface, 
                (self.square_size, self.square_size)
            )
            
            # Draw the sprite
            canvas.blit(scaled_sprite, (screen_x, screen_y))

        # Draw grid background
        for y in range(self.board_size):
            for x in range(self.board_size):
                cell_rect = pygame.Rect(
                    x * self.square_size,
                    self.window_diff + y * self.square_size,
                    self.square_size,
                    self.square_size,
                )
                pygame.draw.rect(canvas, self.COLOR_GRID, cell_rect, 1)

        # Draw fruits (targets)
        fruit_positions = np.argwhere(self.board == self.ITEM)
        for pos in fruit_positions:
            draw_sprite(canvas, pos[1], pos[0], 'fruit')

        # Draw snake using codeA.py's rendering logic
        if len(self.snake_segments) > 0:
            # Draw head with rotation based on direction
            head_pos = self.snake_segments[0]
            rotation_angle = dir_map_to_angle_head.get(self.snake_direction, 0)
            # Convert rotation angle to pygame rotation (clockwise)
            pygame_rotation = -rotation_angle  # Invert because pygame rotates clockwise
            draw_sprite(canvas, head_pos.x, head_pos.y, 'head', pygame_rotation)

            # Draw body segments
            if len(self.snake_segments) > 1:
                # Build list of limbs as in codeA.py
                limbs = self.snake_segments
                
                # Draw the body and turns
                for i in range(1, len(limbs) - 1):
                    curr = limbs[i]
                    prev = limbs[i-1]
                    nxt = limbs[i+1] if i+1 < len(limbs) else None
                    
                    if nxt is None:
                        continue
                    
                    d2 = curr - prev
                    d1 = nxt - curr
                    
                    if d1.x == d2.x and d1.y == d2.y:
                        # Straight body piece
                        rotation = dir_map_to_angle.get(d2, 0)
                        pygame_rotation = -rotation
                        draw_sprite(canvas, curr.x, curr.y, 'body', pygame_rotation)
                    else:
                        # Turning piece - simplified logic
                        # Determine corner orientation
                        rotation = 0
                        
                        if (d1.x > 0 and d2.y < 0) or (d1.y > 0 and d2.x < 0):
                            rotation = 0
                        elif (d1.y > 0 and d2.x > 0) or (d1.x < 0 and d2.y < 0):
                            rotation = 90
                        elif (d1.x > 0 and d2.y > 0) or (d1.y < 0 and d2.x < 0):
                            rotation = -90
                        elif (d1.y < 0 and d2.x > 0) or (d1.x < 0 and d2.y > 0):
                            rotation = 180
                        
                        pygame_rotation = -rotation
                        draw_sprite(canvas, curr.x, curr.y, 'turn', pygame_rotation)
                
                # Draw tail piece
                if len(limbs) > 1:
                    tail_pos = limbs[-1]
                    prev_segment = limbs[-2] if len(limbs) > 1 else limbs[0]
                    tail_dir = prev_segment - tail_pos
                    rotation = dir_map_to_angle.get(tail_dir, 0)
                    pygame_rotation = -rotation
                    draw_sprite(canvas, tail_pos.x, tail_pos.y, 'tail', pygame_rotation)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            
            
            