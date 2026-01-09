import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from random import choice, randint, sample, seed
from dataclasses import dataclass
from enum import Enum
import math
import time

# Enum representing the result of a snake movement step
class SnakeState(Enum):
    OK = 1
    ATE = 2
    DED = 3
    WON = 4

# Utility to rotate sprites by simple numpy operations
# Only handles 90 / -90 / 180 rotations
def _rotate_image(cv_image, _rotation_angle):
    if _rotation_angle == -90:
        return np.transpose(cv_image, (1, 0, 2))[:, ::-1]
    if _rotation_angle == 90:
        return np.transpose(cv_image, (1, 0, 2))[::-1, :]
    if _rotation_angle in [-180, 180]:
        return cv_image[::-1, ::-1]
    return cv_image

# Simple 2D point structure for coordinates
@dataclass(eq=True, frozen=True)
class Point:
    x: int
    y: int

    def copy(self, xincr, yincr):
        return Point(self.x + xincr, self.y + yincr)

    def to_dict(self):
        return {'x': self.x, 'y': self.y}

    @classmethod
    def from_dict(cls, d):
        return cls(d['x'], d['y'])

    def __repr__(self):
        return f"(x: {self.x}, y: {self.y})"

    def __sub__(self, other):
        return Point(self.x-other.x, self.y-other.y)

    def dist(self, other):
        m = Point(self.x-other.x, self.y-other.y)
        return abs(m.x) + abs(m.y)

# Map human-readable direction to coordinate increments
action_dir_map = {
    'up': Point(0, -1),
    'down': Point(0, 1),
    'left': Point(-1, 0),
    'right': Point(1, 0),
}

# Maps direction vector to a rotation angle for sprite drawing
dir_map_to_angle = {
    Point(0, -1): 0,
    Point(0, 1): 180,
    Point(-1, 0): 90,
    Point(1, 0): -90,
}

# Convert background color from hex to BGR
BACKGROUND_COLOR = (140, 211, 255)  # BGR format for OpenCV

action_dir_order = ['right', 'up', 'left', 'down']

# Default initial tail size
INIT_TAIL_SIZE = 4

# Snake object containing body logic
class Snake:
    def __init__(self, x: int = 0, y: int = 0):
        self.head = Point(x, y)
        self.tail = []
        self.tail_size = INIT_TAIL_SIZE
        self.direction = Point(1, 0)  # Starts moving right
        self.dir_idx = 0

    # Check if snake head touches its tail
    def self_collision(self):
        for t in self.tail:
            if self.head.x == t.x and self.head.y == t.y:
                return True
        return False

    def to_dict(self):
        return {
            'head': self.head.to_dict(),
            'tail': [t.to_dict() for t in self.tail],
            'tail_size': self.tail_size,
            'direction': self.direction.to_dict()
        }

    @classmethod
    def from_dict(cls, d):
        s = cls()
        s.head = Point.from_dict(d['head'])
        s.tail = [Point.from_dict(t) for t in d['tail']]
        s.tail_size = d['tail_size']
        s.direction = Point.from_dict(d['direction'])
        return s

    # Move the snake: head moves, old head becomes a tail segment
    def update(self):
        new_head = self.head.copy(self.direction.x, self.direction.y)
        self.tail.append(self.head)  # append old head to tail
        self.head = new_head

    # Trim tail to correct size
    def shed(self):
        if self.tail_size > 0:
            self.tail = self.tail[-self.tail_size:]
        else:
            self.tail = []

    def __repr__(self):
        return f"""Head: {self.head}
        Tail: {self.tail}
        Dir: {self.direction}
        """

    # Apply left/right turn (not used in main loop)
    def apply_turn(self, turn_dir):
        if not turn_dir:
            return
        assert turn_dir in ['left', 'right']
        shift = 1 if turn_dir == 'left' else -1
        self.dir_idx = (self.dir_idx + shift) % 4
        action = action_dir_order[self.dir_idx]
        self.apply_direction(new_dir=action)

    # Apply an absolute direction like 'up', 'down', etc.
    def apply_direction(self, new_dir=None):
        if not new_dir:
            return
        assert new_dir in action_dir_map, f"Unknown direction {new_dir}"
        self.direction = action_dir_map[new_dir]

# The main Gymnasium environment
class SnakeEnv(gym.Env):
    """A Snake game environment for Gymnasium."""
    
    metadata = {'render_modes': ['human', 'rgb_array', 'ansi'], 'render_fps': 10}
    
    def __init__(self, grid_size=10, num_fruits=1, render_mode=None):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_fruits = num_fruits
        self.render_mode = render_mode
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        
        # Image observation space (main_gs=grid_size+2 for border)
        self.main_gs = grid_size + 2  # Add border for visualization
        scale = 8
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(self.main_gs*scale, self.main_gs*scale, 3), 
            dtype=np.uint8
        )
        
        # Load or create sprites
        self.sprites = self._load_sprites()
        
        # Initialize game state
        self.env = EnvCore(grid_size=grid_size, main_gs=self.main_gs, num_fruits=num_fruits)
        
        # For rendering
        if render_mode == 'human':
            cv2.namedWindow('Snake', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Snake', 640, 640)
    
    def _load_sprites(self):
        """Load sprites or create colored placeholders."""
        scale = 8
        sprites = {}
        
        # Try to load actual sprites
        sprite_files = {
            'head': 'sprites/head.png',
            'body': 'sprites/body.png',
            'turn': 'sprites/turn.png',
            'fruit': 'sprites/fruit.png',
            'tail': 'sprites/tail.png',
        }
        
        for name, path in sprite_files.items():
            try:
                sprite = cv2.imread(path)
                if sprite is not None and sprite.shape[0] > 0:
                    # Resize to standard size if needed
                    if sprite.shape[0] != scale or sprite.shape[1] != scale:
                        sprite = cv2.resize(sprite, (scale, scale))
                    sprites[name] = sprite
                else:
                    # Create placeholder
                    sprites[name] = self._create_placeholder_sprite(name, scale)
            except:
                # Create placeholder on any error
                sprites[name] = self._create_placeholder_sprite(name, scale)
        
        return sprites
    
    def _create_placeholder_sprite(self, name, scale):
        """Create a colored placeholder sprite."""
        sprite = np.zeros((scale, scale, 3), dtype='uint8')
        if name == 'head':
            sprite[:] = (0, 255, 0)  # Green head (BGR)
        elif name == 'body':
            sprite[:] = (0, 200, 0)  # Dark green body
        elif name == 'fruit':
            sprite[:] = (0, 0, 255)  # Red fruit
        elif name == 'tail':
            sprite[:] = (0, 150, 0)  # Darker green tail
        elif name == 'turn':
            sprite[:] = (0, 180, 0)  # Medium green turn
        return sprite
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        self.env = EnvCore(grid_size=self.grid_size, main_gs=self.main_gs, num_fruits=self.num_fruits)
        
        # Get initial observation
        observation = self.env.to_image()
        
        info = {
            'score': len(self.env.snake.tail),
            'steps': self.env.step,
            'fruit_eaten': self.grid_size**2 - len(self.env.pos_set) + len(self.env.snake.tail) + 1
        }
        
        return observation, info
    
    def step(self, action):
        """Take a step in the environment."""
        # Convert discrete action to direction string
        action_map = ['up', 'right', 'down', 'left']
        direction = action_map[action]
        
        # Update game state
        result = self.env.update(direction)
        self.env.step += 1
        
        # Calculate reward
        reward = 0
        terminated = False
        truncated = False
        
        if result == SnakeState.ATE:
            reward = 10.0  # Reward for eating fruit
        elif result == SnakeState.DED:
            reward = -10.0  # Penalty for dying
            terminated = True
        elif result == SnakeState.WON:
            reward = 100.0  # Large reward for winning
            terminated = True
        else:
            reward = -0.1  # Small penalty for each step to encourage efficiency
        
        # Get observation
        observation = self.env.to_image()
        
        # Check for truncation (max steps without eating)
        if self.env.last_ate > self.env.stamina:
            truncated = True
        
        info = {
            'score': len(self.env.snake.tail),
            'steps': self.env.step,
            'fruit_eaten': self.grid_size**2 - len(self.env.pos_set) + len(self.env.snake.tail) + 1,
            'state': result.name
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return
        
        img = self.env.to_image()
        
        if self.render_mode == 'rgb_array':
            return img
        elif self.render_mode == 'human':
            # Resize for display
            display_img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('Snake', display_img)
            cv2.waitKey(1)
        elif self.render_mode == 'ansi':
            return self._render_ansi()
    
    def _render_ansi(self):
        """Render as ASCII art (simple representation)."""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Draw snake head
        head = self.env.snake.head
        if 0 <= head.x < self.grid_size and 0 <= head.y < self.grid_size:
            grid[head.y][head.x] = 'H'
        
        # Draw snake tail
        for segment in self.env.snake.tail:
            if 0 <= segment.x < self.grid_size and 0 <= segment.y < self.grid_size:
                grid[segment.y][segment.x] = 'S'
        
        # Draw fruits
        for fruit in self.env.fruit_loc:
            if 0 <= fruit.x < self.grid_size and 0 <= fruit.y < self.grid_size:
                grid[fruit.y][fruit.x] = 'F'
        
        # Create ASCII representation
        lines = []
        lines.append(f"Score: {len(self.env.snake.tail)} | Steps: {self.env.step}")
        lines.append("+" + "-" * self.grid_size + "+")
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-" * self.grid_size + "+")
        
        return "\n".join(lines)
    
    def close(self):
        """Close the environment and any windows."""
        if self.render_mode == 'human':
            cv2.destroyAllWindows()

# The core game logic (renamed from Env to EnvCore to avoid confusion)
class EnvCore:
    def __init__(self, grid_size=10, main_gs=10, num_fruits=1):
        self.gs = grid_size
        self.main_gs = main_gs
        self.num_fruits = num_fruits
        self.reset()

    def reset(self):
        self.step = 0
        self.last_ate = 0
        
        # Subgrid placement
        self.subgrid_loc = Point(0, 0)
        if self.gs in [10, 20, 38]:
            self.subgrid_loc = Point(1, 1)

        # Create a new snake
        self.snake = Snake()
        self.snake.head = Point(self.gs//2, self.gs//2)

        # Precompute all positions
        pos_list = []
        for i in range(self.gs):
            for j in range(self.gs):
                pos_list.append(Point(i, j))

        self.pos_set = set(pos_list)
        self.fruit_locations = []
        self.set_fruits()

    @property
    def stamina(self):
        a = self.gs ** 2
        stamina = a + len(self.snake.tail) + 1
        stamina = min(a * 2, stamina)
        return stamina

    # Main game step
    def update(self, direction=None):
        self.last_ate += 1
        
        # Apply new direction
        self.snake.apply_direction(direction)
        
        # Move the snake
        self.snake.update()
        out_enum = SnakeState.OK

        # Check fruit collision
        if self.snake.head in self.fruit_locations:
            self.fruit_locations.pop(self.fruit_locations.index(self.snake.head))
            self.last_ate = 0

            try:
                self.set_fruits()
                self.snake.tail_size += 1
                out_enum = SnakeState.ATE
            except IndexError:
                out_enum = SnakeState.WON

            if len(self.fruit_locations) == 0:
                out_enum = SnakeState.WON

        # Trim tail
        self.snake.shed()

        # Check wall or self collision
        if not self._bounds_check(self.snake.head) or self.snake.self_collision():
            out_enum = SnakeState.DED
        elif self.last_ate > self.stamina:
            out_enum = SnakeState.DED

        return out_enum

    @property
    def fruit_loc(self):
        return self.fruit_locations

    # Generate or replenish fruits on the grid
    def set_fruits(self):
        snake = self.snake
        snake_locs = set([snake.head] + snake.tail + self.fruit_locations)
        possible_positions = self.pos_set.difference(snake_locs)
        diff = self.num_fruits - len(self.fruit_locations)
        new_locs = sample(list(possible_positions), k=min(diff, len(possible_positions)))
        self.fruit_locations.extend(new_locs)

    # Bounds check for grid
    def _bounds_check(self, pos):
        return pos.x >= 0 and pos.x < self.gs and pos.y >= 0 and pos.y < self.gs

    # Convert the grid into an image
    def to_image(self):
        scale = 8
        
        # Create main canvas
        full_canvas = np.zeros((self.main_gs*scale, self.main_gs*scale, 3), 'uint8')
        h, w = self.gs*scale, self.gs*scale

        # Select the subgrid area
        canvas = full_canvas[self.subgrid_loc.y*scale:self.subgrid_loc.y*scale+h,
                             self.subgrid_loc.x*scale:self.subgrid_loc.x*scale+w]

        # Set background color
        canvas[:] = BACKGROUND_COLOR
        
        # For Gymnasium env, sprites are loaded in the parent class
        # This method is overridden to use the parent's sprites
        return full_canvas

# For backward compatibility
Env = EnvCore

# Example usage
if __name__ == '__main__':
    # Test the environment
    env = SnakeEnv(grid_size=10, render_mode='human')
    
    observation, info = env.reset()
    print(f"Initial observation shape: {observation.shape}")
    print(f"Initial info: {info}")
    
    # Take a few random actions
    for i in range(20):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i}: Action={action}, Reward={reward:.2f}, Terminated={terminated}, Score={info['score']}")
        
        if terminated or truncated:
            print("Episode ended!")
            observation, info = env.reset()
            print(f"Reset. New score: {info['score']}")
        time.sleep(1)
    env.close()