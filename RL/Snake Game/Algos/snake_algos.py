import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from snake_game import SnakeGameEnv 
from collections import defaultdict

env_train = SnakeGameEnv(
    board_size=10,
    n_channel=1,
   # n_target=1,
    render_mode=None
)

env_view = SnakeGameEnv(
    board_size=10,
    n_channel=1,
    #n_target=1,
    render_mode="human"
)

FOOD_VAL = 101
HEAD_VAL = 1

# =========================
# Outils 
# =========================
def find_pos(grid, val):
    pos = np.argwhere(grid == val)
    return tuple(pos[0]) if len(pos) else None

def is_collision(grid, r, c):
    # Hors-grille => collision
    if r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1]:
        return True
    cell = grid[r, c]
    # Collision si touche le corps (tout sauf 0 et nourriture)
    return (cell != 0) and (cell != FOOD_VAL)

def infer_direction(grid, head_pos, prev_dir=3):
    """
    Essaie d'inférer la direction via le segment 2 (cou).
    Si non dispo, on conserve prev_dir.
    Convention: 0=up, 1=down, 2=left, 3=right
    """
    neck = find_pos(grid, 2)
    if head_pos is None or neck is None:
        return prev_dir

    hr, hc = head_pos
    nr, nc = neck
    dr, dc = hr - nr, hc - nc

    if dr == -1 and dc == 0: return 0  # up
    if dr ==  1 and dc == 0: return 1  # down
    if dr ==  0 and dc == -1: return 2  # left
    if dr ==  0 and dc ==  1: return 3  # right
    return prev_dir

def left_dir(d):
    return {0: 2, 2: 1, 1: 3, 3: 0}[d]

def right_dir(d):
    return {0: 3, 3: 1, 1: 2, 2: 0}[d]

def step_from_dir(r, c, d):
    if d == 0: return r - 1, c
    if d == 1: return r + 1, c
    if d == 2: return r, c - 1
    if d == 3: return r, c + 1
    return r, c

def obs_to_state_key(obs, prev_dir=3):
    """
    obs shape: (1, 10, 10)
    Retourne: (state_key, new_prev_dir)
    state_key = tuple discret => OK pour Q-table
    """
    grid = obs[0]
    head = find_pos(grid, HEAD_VAL)
    food = find_pos(grid, FOOD_VAL)

    if head is None or food is None:
        return ("bad_obs",), prev_dir

    direction = infer_direction(grid, head, prev_dir=prev_dir)

    hr, hc = head
    fr, fc = food

    # nourriture relative
    food_dir_x = -1 if fc < hc else (1 if fc > hc else 0)
    food_dir_y = -1 if fr < hr else (1 if fr > hr else 0)

    # danger devant/gauche/droite
    f_r, f_c = step_from_dir(hr, hc, direction)
    l_r, l_c = step_from_dir(hr, hc, left_dir(direction))
    r_r, r_c = step_from_dir(hr, hc, right_dir(direction))

    danger_front = int(is_collision(grid, f_r, f_c))
    danger_left  = int(is_collision(grid, l_r, l_c))
    danger_right = int(is_collision(grid, r_r, r_c))

    state_key = (danger_front, danger_left, danger_right, food_dir_x, food_dir_y, direction)
    return state_key, direction

# =========================
# Politique 
# =========================
def epsilon_greedy_action(Q, state, n_actions, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    qs = [Q.get((state, a), 0.0) for a in range(n_actions)]
    return int(np.argmax(qs))

def greedy_action(Q, state, n_actions):
    qs = [Q.get((state, a), 0.0) for a in range(n_actions)]
    return int(np.argmax(qs))

# =========================
# épisode
# =========================
def generate_episode_mc(env, Q, epsilon, itermax):
    """
    Génère un épisode: liste de (s, a, r)
    """
    episode = []
    obs, info = env.reset()
    prev_dir = 3
    n_actions = env.action_space.n

    for _ in range(itermax):
        s, prev_dir = obs_to_state_key(obs, prev_dir)
        a = epsilon_greedy_action(Q, s, n_actions, epsilon)

        obs2, r, terminated, truncated, info = env.step(a)
        episode.append((s, a, r))
        obs = obs2

        if terminated or truncated:
            break
    # print(episode)
    return episode

def generate_episode_mc_exploring_starts(env, Q, itermax):
    """
    Génère un épisode MC avec Exploring Starts:
    - état initial aléatoire (via reset)
    - action initiale aléatoire
    """
    episode = []
    obs, info = env.reset()
    prev_dir = 3
    n_actions = env.action_space.n

    # Initial state
    s, prev_dir = obs_to_state_key(obs, prev_dir)

    # Exploring start: action initiale aléatoire
    a = np.random.randint(n_actions)

    for _ in range(itermax):
        obs2, r, terminated, truncated, info = env.step(a)
        episode.append((s, a, r))

        if terminated or truncated:
            break

        # Next state
        s_prime, prev_dir = obs_to_state_key(obs2, prev_dir)

        # Politique greedy ensuite
        a = greedy_action(Q, s_prime, n_actions)

        s = s_prime
        obs = obs2

    return episode

# =========================
# Monte Carlo 
# =========================
def mc_control_onpolicy_first_visit(env, gamma=0.99, nb_episodes=1000, itermax=200, epsilon=0.1):
    Q = {}          # Q[(s,a)]
    Returns = {}    # Returns[(s,a)] = list of returns

    returns_curve = []
    lengths_curve = []

    for ep in range(nb_episodes):
        episode = generate_episode_mc(env, Q, epsilon, itermax)
        lengths_curve.append(len(episode))
        total_return = sum([x[2] for x in episode])
        returns_curve.append(total_return)

        # calcul retours G_t en backward + first-visit sur (s,a)
        G = 0.0
        visited = set()

        for t in range(len(episode) - 1, -1, -1):
            s, a, r = episode[t]
            G = r + gamma * G

            if (s, a) not in visited:
                visited.add((s, a))
                Returns.setdefault((s, a), []).append(G)
                Q[(s, a)] = float(np.mean(Returns[(s, a)]))

        # petit log de temps en temps
        if (ep + 1) % 500 == 0:
            print(f"[train] ep={ep+1}/{nb_episodes} | last_return={total_return:.2f} | last_len={len(episode)}")

    return Q, returns_curve, lengths_curve

def mc_control_exploring_starts(env, gamma=0.99, nb_episodes=3000, itermax=200, epsilon=0.1):
    """
    Monte-Carlo Control avec Exploring Starts (adapté Snake) :
    - On ne peut pas imposer un état initial arbitraire dans Snake facilement.
    - On réalise donc ES en forçant la 1ère action a0 aléatoire, puis on suit une policy epsilon-greedy.
    - Mise à jour First-Visit MC sur (s,a).
    """
    Q = {}
    Returns = defaultdict(list)

    returns_curve, lengths_curve, apples_curve = [], [], []
    n_actions = env.action_space.n

    for ep in range(nb_episodes):
        episode = []
        obs, _ = env.reset()
        prev_dir = 3

        # Exploring start: première action aléatoire
        s0, prev_dir = obs_to_state_key(obs, prev_dir)
        a0 = np.random.randint(n_actions)

        obs, r, terminated, truncated, _ = env.step(a0)
        episode.append((s0, a0, r))

        done = terminated or truncated

        # suite: epsilon-greedy
        for _ in range(itermax - 1):
            if done:
                break
            s, prev_dir = obs_to_state_key(obs, prev_dir)
            a = epsilon_greedy_action(Q, s, n_actions, epsilon)
            obs2, r, terminated, truncated, _ = env.step(a)
            episode.append((s, a, r))
            obs = obs2
            done = terminated or truncated

        total_return = sum(r for _, _, r in episode)
        returns_curve.append(total_return)
        lengths_curve.append(len(episode))
        apples_curve.append(sum(1 for _, _, r in episode if r == 1))

        # First-visit MC update
        G = 0.0
        visited = set()
        for t in range(len(episode) - 1, -1, -1):
            s_t, a_t, r_t = episode[t]
            G = r_t + gamma * G
            if (s_t, a_t) not in visited:
                visited.add((s_t, a_t))
                Returns[(s_t, a_t)].append(G)
                Q[(s_t, a_t)] = float(np.mean(Returns[(s_t, a_t)]))

        if (ep + 1) % 500 == 0:
            print(f"[MC-ES] ep={ep+1}/{nb_episodes} | return={total_return:.2f} | Q={len(Q)}")

    return Q, returns_curve, lengths_curve

# =========================
# SARSA 
# =========================
def sarsa_control(env, alpha=0.1, gamma=0.99, nb_episodes=1000, itermax=200, epsilon=0.1):
    """
    SARSA (On-policy TD control)
    """
    Q = {}
    returns_curve = []
    lengths_curve = []
    n_actions = env.action_space.n
    
    for ep in range(nb_episodes):
        obs, info = env.reset()
        prev_dir = 3
        
        # Initialize S
        s, prev_dir = obs_to_state_key(obs, prev_dir)
        # Choose A from S using epsilon-greedy
        a = epsilon_greedy_action(Q, s, n_actions, epsilon)
        
        total_return = 0
        t = 0
        
        for _ in range(itermax):
            # Take action A, observe R, S'
            obs2, r, terminated, truncated, info = env.step(a)
            s_prime, prev_dir = obs_to_state_key(obs2, prev_dir)
            
            total_return += r
            t += 1
            
            # Choose A' from S' using epsilon-greedy
            a_prime = epsilon_greedy_action(Q, s_prime, n_actions, epsilon)
            
            # SARSA update
            Q_current = Q.get((s, a), 0.0)
            Q_next = Q.get((s_prime, a_prime), 0.0)
            Q[(s, a)] = Q_current + alpha * (r + gamma * Q_next - Q_current)
            
            # Update state and action
            s, a = s_prime, a_prime
            obs = obs2
            
            if terminated or truncated:
                break
        
        returns_curve.append(total_return)
        lengths_curve.append(t)
        
        if (ep + 1) % 500 == 0:
            print(f"[SARSA] ep={ep+1}/{nb_episodes} | return={total_return:.2f} | len={t}")
    
    return Q, returns_curve, lengths_curve

# =========================
# Q-learninG
# =========================
def q_learning_control(env, alpha=0.1, gamma=0.99, nb_episodes=1000, itermax=200, epsilon=0.1):
    """
    Q-learning (Off-policy TD control)
    """
    Q = {}
    returns_curve = []
    lengths_curve = []
    n_actions = env.action_space.n
    
    for ep in range(nb_episodes):
        obs, info = env.reset()
        prev_dir = 3
        total_return = 0
        t = 0
        
        for _ in range(itermax):
            s, prev_dir = obs_to_state_key(obs, prev_dir)
            # Choose action using epsilon-greedy
            a = epsilon_greedy_action(Q, s, n_actions, epsilon)
            
            # Take action, observe reward and next state
            obs2, r, terminated, truncated, info = env.step(a)
            s_prime, prev_dir = obs_to_state_key(obs2, prev_dir)
            
            total_return += r
            t += 1
            
            # Q-learning update (off-policy)
            Q_current = Q.get((s, a), 0.0)
            
            # Find max Q for next state (greedy policy)
            q_next_values = [Q.get((s_prime, a_prime), 0.0) for a_prime in range(n_actions)]
            max_q_next = max(q_next_values) if q_next_values else 0.0
            
            Q[(s, a)] = Q_current + alpha * (r + gamma * max_q_next - Q_current)
            
            obs = obs2
            
            if terminated or truncated:
                break
        
        returns_curve.append(total_return)
        lengths_curve.append(t)
        
        if (ep + 1) % 500 == 0:
            print(f"[Q-learning] ep={ep+1}/{nb_episodes} | return={total_return:.2f} | len={t}")
    
    return Q, returns_curve, lengths_curve

# =========================
# n-step
# =========================
def n_step_td_control(env, n=3, alpha=0.1, gamma=0.99, nb_episodes=1000, itermax=200, epsilon=0.1):
    """
    n-step TD control (SARSA style)
    """
    Q = {}
    returns_curve = []
    lengths_curve = []
    n_actions = env.action_space.n
    
    for ep in range(nb_episodes):
        # Initialize and store S0
        obs, info = env.reset()
        prev_dir = 3
        s, prev_dir = obs_to_state_key(obs, prev_dir)
        
        # Choose and store A0
        a = epsilon_greedy_action(Q, s, n_actions, epsilon)
        
        T = float('inf')
        t = 0
        total_return = 0
        
        # Initialize buffers
        states = [s]
        actions = [a]
        rewards = [0]  # R0 = 0
        
        while True:
            if t < T:
                # Take action At
                obs2, r, terminated, truncated, info = env.step(a)
                s_prime, prev_dir = obs_to_state_key(obs2, prev_dir)
                
                total_return += r
                rewards.append(r)
                states.append(s_prime)
                
                obs = obs2
                
                if terminated or truncated:
                    T = t + 1
                else:
                    # Choose and store A_{t+1}
                    a_prime = epsilon_greedy_action(Q, s_prime, n_actions, epsilon)
                    actions.append(a_prime)
                    s, a = s_prime, a_prime
            
            tau = t - n + 1
            if tau >= 0:
                # Calculate G
                G = 0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += gamma ** (i - tau - 1) * rewards[i]
                
                if tau + n < T:
                    s_tau_n = states[tau + n]
                    a_tau_n = actions[tau + n]
                    G += gamma ** n * Q.get((s_tau_n, a_tau_n), 0.0)
                
                # Update Q
                s_tau = states[tau]
                a_tau = actions[tau]
                Q_current = Q.get((s_tau, a_tau), 0.0)
                Q[(s_tau, a_tau)] = Q_current + alpha * (G - Q_current)
            
            t += 1
            if tau == T - 1:
                break
        
        returns_curve.append(total_return)
        lengths_curve.append(t)
        
        if (ep + 1) % 500 == 0:
            print(f"[n-step TD (n={n})] ep={ep+1}/{nb_episodes} | return={total_return:.2f} | len={t}")
    
    return Q, returns_curve, lengths_curve

# =========================
# Visu
# =========================
def watch_agent(env, Q, n=5, itermax=200, sleep=0.15):
    n_actions = env.action_space.n

    for ep in range(n):
        obs, info = env.reset()
        prev_dir = 3
        total_return = 0.0

        for t in range(itermax):
            s, prev_dir = obs_to_state_key(obs, prev_dir)
            a = greedy_action(Q, s, n_actions)

            obs, r, terminated, truncated, info = env.step(a)
            total_return += r

            time.sleep(sleep)

            if terminated or truncated:
                break

        print(f"[view] Episode {ep} | return={total_return:.2f} | length={t+1}")

# =========================
#Plot
# =========================
def moving_average(x, window=50):
    x = np.array(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")

def plot_curves(returns_curve, lengths_curve, algorithm_name="Algorithm"):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(moving_average(returns_curve, 50))
    plt.title(f"{algorithm_name} - Return (moyenne glissante 50)")
    plt.xlabel("Episode")
    plt.ylabel("Return moyen")
    
    plt.subplot(1, 2, 2)
    plt.plot(moving_average(lengths_curve, 50))
    plt.title(f"{algorithm_name} - Longueur (moyenne glissante 50)")
    plt.xlabel("Episode")
    plt.ylabel("Steps moyens")
    
    plt.tight_layout()
    plt.show()

# =========================
# run
# =========================
def run_algorithm(algorithm="mc", **kwargs):
    """
    Main function to run the selected algorithm.
    
    Parameters:
    -----------
    algorithm : str
        One of: "mc" (Monte Carlo), "sarsa", "qlearning" (TD(0)), 
                "nstep" (n-step TD), "expected_sarsa"
    **kwargs : dict
        Parameters passed to the algorithm function
    """
    algorithm = algorithm.lower()
    
    if algorithm == "mc":
        print("Running Monte Carlo Control...")
        Q, returns, lengths = mc_control_onpolicy_first_visit(
            env=env_train,
            gamma=kwargs.get('gamma', 0.99),
            nb_episodes=kwargs.get('nb_episodes', 1000),
            itermax=kwargs.get('itermax', 200),
            epsilon=kwargs.get('epsilon', 0.1)
        )
        name = "Monte Carlo Control"
        
    elif algorithm == "mc_es":
        print("Running Monte Carlo Exploring Starts...")
        Q, returns, lengths = mc_control_exploring_starts(
            env_train,
            gamma=kwargs.get("gamma", 0.99),
            nb_episodes=kwargs.get("nb_episodes", 1000),
            itermax=kwargs.get("itermax", 200),
        )
        name = "Monte Carlo Exploring Starts"
        
    elif algorithm == "sarsa":
        print("Running SARSA Control...")
        Q, returns, lengths = sarsa_control(
            env=env_train,
            alpha=kwargs.get('alpha', 0.1),
            gamma=kwargs.get('gamma', 0.99),
            nb_episodes=kwargs.get('nb_episodes', 1000),
            itermax=kwargs.get('itermax', 200),
            epsilon=kwargs.get('epsilon', 0.1)
        )
        name = "SARSA"
        
    elif algorithm == "td":
        print("Running TD(0)...")
        Q, returns, lengths = q_learning_control(
            env=env_train,
            alpha=kwargs.get('alpha', 0.1),
            gamma=kwargs.get('gamma', 0.99),
            nb_episodes=kwargs.get('nb_episodes', 1000),
            itermax=kwargs.get('itermax', 200),
            epsilon=kwargs.get('epsilon', 0.1)
        )
        name = "TD(0)"
        
    elif algorithm == "nstep":
        print("Running n-step TD Control...")
        Q, returns, lengths = n_step_td_control(
            env=env_train,
            n=kwargs.get('n', 3),
            alpha=kwargs.get('alpha', 0.1),
            gamma=kwargs.get('gamma', 0.99),
            nb_episodes=kwargs.get('nb_episodes', 1000),
            itermax=kwargs.get('itermax', 200),
            epsilon=kwargs.get('epsilon', 0.1)
        )
        name = f"n-step TD (n={kwargs.get('n', 3)})"
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: mc, sarsa, qlearning, nstep, expected_sarsa")
    
    return Q, returns, lengths, name

# =========================
# Main
# =========================
if __name__ == "__main__":
   
    algorithm_choice = "mc"  # "mc", "mc_es", "sarsa", "td", "nstep"
    
    Q, returns_curve, lengths_curve, algo_name = run_algorithm(
        algorithm=algorithm_choice,
        alpha=0.1,        
        gamma=0.95,         
        nb_episodes=20000,   
        itermax=400,       
        epsilon=0.1,        
        n=3                 
    )
    
    plot_curves(returns_curve, lengths_curve, algo_name)
    
    watch_agent(env_view, Q, n=3, itermax=400, sleep=0.1)