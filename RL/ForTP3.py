#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:02:58 2020

@author: berar
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from Maze_generating_interface import App
import time

actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
def get_actions(state,actions):
    # list of possible actions 
    local_actions = list()
    # print("state = ",state)
    for a in actions :
        m = [0,0]
        if a == "UP" :
            m[0] = -1   
        elif a == "DOWN" :
            m[0] = +1
        elif a == "LEFT" :
            m[1] = -1
        else : # a=="RIGHT"
            m[1] = +1
        # if state[0]+m[0] <5 and state[1]+m[1]<5 :
        try:
            if Maze[state[0]+m[0],state[1]+m[1]] == 1.0 :    
                local_actions.append(a)
        except Exception:
            pass 

    return local_actions
            
def next_state(state, action, exit_state, Maze):
    # give the resulting state end reward 
    new_state = np.copy(state)
    if action == "UP" :
        new_state[0] -= 1   
    elif action == "DOWN" :
        new_state[0] += 1
    elif action == "LEFT" :
        new_state[1] -= 1
    else : # a=="RIGHT"
        new_state[1] += 1
    
    if all(new_state == exit_state) :
        reward = 1
    else:
        reward = 0
    return new_state,reward

def generate_episode(init_state, actions, Maze, exit_state, policy=None, epsilon=0.0, itermax=1000):
    """
    Generates an episode using a given (ε-soft) policy.
    If policy is None, uses uniform random actions.
    """
    i = 0
    current_state = np.copy(init_state)
    episode = []

    while i < itermax and not all(current_state == exit_state):
        possible_actions = get_actions(current_state, actions)
        if len(possible_actions) == 0:
            break
        if policy is None or tuple(current_state) not in policy:
            action = np.random.choice(possible_actions)
        else:
            # ε-soft policy
            if np.random.rand() < epsilon:
                action = np.random.choice(possible_actions)
            else:
                action = policy[tuple(current_state)]
                if action not in possible_actions:
                    action = np.random.choice(possible_actions)

        new_state, reward = next_state(current_state, action, exit_state, Maze)
        episode.append((current_state, action, reward, new_state))
        current_state = new_state
        i += 1
    return episode

def get_states_from_Maze(Maze) :
    # return states, exit_state, init_states,  
    states = list(np.argwhere(Maze == 1))
    M = np.zeros(Maze.shape)
    M[:,0]=1
    M[:,-1] = 1
    M[0,:]=1
    M[-1,:]=1
    
    exit_state = np.argwhere(Maze*M)[0]
    init_states = list(np.copy(states))
    init_states.pop(np.argwhere((init_states == exit_state).sum(axis =1) == 2)[0][0])
    return states,exit_state,init_states

################## METHODS #############################

def MC_First_Visit(num_episodes, states, actions, Maze, exit_state, gamma=0.9):

    states,exit_state,init_states = get_states_from_Maze(Maze)
    V = {tuple(s): 0.0 for s in states}
    returns = {tuple(s): [] for s in states}
    
    for it in range(num_episodes):
        init_state = init_states[np.random.randint(len(init_states))]
        episode = generate_episode(init_state,actions,Maze,exit_state, itermax=100*len(states))
        # print(episode)
        states_in_episode = [tuple(x[0]) for x in episode]

        for t in reversed(range(len(episode))):
            state, action, reward, next_state2 = episode[t]
            state_tuple = tuple(state)
            if state_tuple not in states_in_episode[:t]:
                # Compute the return G from time t to the end
                G = 0.0
                discount = 1.0
                for k in range(t, len(episode)):
                    G += discount * episode[k][2]  # reward
                    discount *= gamma
                returns[state_tuple].append(G)
                V[state_tuple] = np.mean(returns[state_tuple])
        
        V[tuple(exit_state)] = 1.0

    return V


def MC_Exploring_Starts(num_episodes, states, actions, Maze, exit_state, gamma=0.9):

    Q = {(tuple(s), a): 0.0 for s in states for a in actions}
    returns = {(tuple(s), a): [] for s in states for a in actions}
    policy = {tuple(s): np.random.choice(actions) for s in states}

    for episode_num in range(num_episodes):
        # Exploring start
        s0 = states[np.random.randint(len(states))]
        possible_actions = get_actions(s0, actions)
        if len(possible_actions) == 0:
            continue
        a0 = np.random.choice(possible_actions)

        # Generate episode following policy 
        episode = []
        current_state = np.copy(s0)
        current_action = a0
        for t in range(1000):
            new_state, reward = next_state(current_state, current_action, exit_state, Maze)
            episode.append((current_state, current_action, reward, new_state))
            if all(new_state == exit_state):
                episode.append((new_state, None, 0, new_state))
                break
            current_state = new_state
            current_action = policy[tuple(current_state)]
            if current_action not in get_actions(current_state, actions):
                break
            
        # episode = generate_episode(s0, actions, Maze, exit_state, policy, itermax=100*len(states))

        # compute returns and update Q and policy
        G = 0.0
        for t in reversed(range(len(episode))):
            state, action, reward, next_state2 = episode[t]
            G = gamma * G + reward
            if action is None:
                continue
            
            if not any((tuple(state) == tuple(x[0]) and action == x[1]) for x in episode[:t]):
                returns[(tuple(state), action)].append(G)
                Q[(tuple(state), action)] = np.mean(returns[(tuple(state), action)])
                # Greedy policy improvement
                possible_actions = get_actions(state, actions)
                if possible_actions:
                    best_a = max(possible_actions, key=lambda a: Q[(tuple(state), a)])
                    policy[tuple(state)] = best_a

    # get V from Q
    V = {tuple(s): max([Q[(tuple(s), a)] for a in actions]) for s in states}
    V[tuple(exit_state)] = 1.0
    return policy, V


def MC_OnPolicy_FirstVisit_Control(num_episodes, states, actions, Maze, exit_state, gamma=0.9, epsilon=0.1):

    Q = {(tuple(s), a): 0.0 for s in states for a in actions}
    returns = {(tuple(s), a): [] for s in states for a in actions}
    policy = {tuple(s): np.random.choice(actions) for s in states}

    for episode_num in range(num_episodes):
        s0 = states[np.random.randint(len(states))]
        init_states, _, _ = get_states_from_Maze(Maze)
        # generete with e-greedy
        episode = generate_episode(s0, actions, Maze, exit_state, policy, epsilon, itermax=100*len(states))

        G = 0.0
        for t in reversed(range(len(episode))):
            state, action, reward, next_state2 = episode[t]
            G = gamma * G + reward
            # check first visit
            if not any((tuple(state) == tuple(x[0]) and action == x[1]) for x in episode[:t]):
                returns[(tuple(state), action)].append(G)
                Q[(tuple(state), action)] = np.mean(returns[(tuple(state), action)])

                # Improve policy 
                possible_actions = get_actions(state, actions)
                if possible_actions:
                    best_a = max(possible_actions, key=lambda a: Q[(tuple(state), a)])
                    policy[tuple(state)] = best_a

    V = {tuple(s): max([Q[(tuple(s), a)] for a in actions]) for s in states}
    V[tuple(exit_state)] = 1.0
    return policy, V


def Temporal_Difference(num_episodes, states, actions, Maze, exit_state, gamma=0.9, alpha=0.1, policy=None):
    V = {tuple(s): 0.0 for s in states}

    for episode in range(num_episodes):
        state = np.copy(states[np.random.randint(len(states))])

        #only stop when in exit
        while not all(state == exit_state):
            possible_actions = get_actions(state, actions)
            if len(possible_actions) == 0:
                break # no move  
            action = np.random.choice(possible_actions)
            next_s, reward = next_state(state, action, exit_state, Maze)

            s_tuple = tuple(state)
            next_tuple = tuple(next_s)

            # TD(0) Update
            V[s_tuple] = V[s_tuple] + alpha * (reward + gamma * V.get(next_tuple, 0.0) - V[s_tuple])

            #move to continue
            state = next_s

    V[tuple(exit_state)] = 1.0
    return V
    

def SARSA(num_episodes, states, actions, Maze, exit_state, gamma=0.9, alpha=0.1, epsilon=0.1):

    Q = {(tuple(s), a): 0.0 for s in states for a in actions}
    policy = {tuple(s): np.random.choice(actions) for s in states}

    for episode in range(num_episodes):
        state = states[np.random.randint(len(states))]
        possible_actions = get_actions(state, actions)

        if len(possible_actions) == 0:
            continue

        # first action using ε-greedy 
        if np.random.rand() < epsilon:
            action = np.random.choice(possible_actions)
        else:
            action = max(possible_actions, key=lambda a: Q[(tuple(state), a)])

        while not all(state == exit_state):

            next_s, reward = next_state(state, action, exit_state, Maze)

            # If terminal state, update Q and stop episode
            if all(next_s == exit_state):
                Q[(tuple(state), action)] += alpha * (reward - Q[(tuple(state), action)])
                break

            next_possible = get_actions(next_s, actions)
            if len(next_possible) == 0:
                break

            # next action using Q
            if np.random.rand() < epsilon:
                next_action = np.random.choice(next_possible)
            else:
                next_action = max(next_possible, key=lambda a: Q[(tuple(next_s), a)])

            # Atualização SARSA
            Q[(tuple(state), action)] += alpha * (
                reward + gamma * Q[(tuple(next_s), next_action)] - Q[(tuple(state), action)]
            )

            state = next_s
            action = next_action

    V = {tuple(s): max([Q[(tuple(s), a)] for a in actions]) for s in states}
    V[tuple(exit_state)] = 1.0

    return Q, V


def n_step_TD(num_episodes, states, actions, Maze, exit_state, n=3, gamma=0.9, alpha=0.1, policy=None):
    # Initialize value function as dict
    V = {tuple(s): 0.0 for s in states}

    for episode in range(num_episodes):
        state = np.copy(states[np.random.randint(len(states))])

        S = [tuple(state)]  # States buffer
        R = [0.0]           # Rewards buffer

        T = float('inf')
        t = 0

        while True:
            # ----------- STEP 1: Generate experience -----------
            if t < T:
                possible_actions = get_actions(state, actions)
                if len(possible_actions) == 0:
                    break

                # Choose action
                if policy is None or tuple(state) not in policy:
                    action = np.random.choice(possible_actions)
                else:
                    action = policy[tuple(state)]
                    if action not in possible_actions:
                        action = np.random.choice(possible_actions)

                next_s, reward = next_state(state, action, exit_state, Maze)

                S.append(tuple(next_s))
                R.append(reward)

                if all(next_s == exit_state):
                    T = t + 1
                else:
                    state = next_s

            # ----------- STEP 2: Compute update index tau -----------
            tau = t - n + 1

            if tau >= 0:
                G = 0.0
                # Sum real rewards over n steps (or until T)
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += (gamma ** (i - tau - 1)) * R[i]
                # If episode not finished within n steps, estimate
                if tau + n < T:
                    G += (gamma ** n) * V[S[tau + n]]
                 # Update value
                V[S[tau]] = V[S[tau]] + alpha * (G - V[S[tau]])
             # Stop when final update is done
            if tau == T - 1:
                break
            t += 1

    # Explicit terminal value
    V[tuple(exit_state)] = 1.0
    return V

# # ---------------------------------
# # PLOT and PRINT
# # ---------------------------------

def build_value_matrix(V_dict, states, shape):
    V_matrix = np.zeros(shape)
    for s in states:
        V_matrix[s[0], s[1]] = V_dict[tuple(s)]
    return V_matrix


def evaluate_methods(results):
    scores = {}
    for name, V in results.items():
        positive_values = V[V > 0]
        if positive_values.size > 0:
            scores[name] = np.mean(positive_values)
        else:
                scores[name] = -np.inf # avoid selecting methods with no positive values

    best = max(scores, key=scores.get)
    return best, scores


def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start


def plot_all(value_matrices, titles):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()

    for ax, V, title in zip(axes, value_matrices, titles):
        im = ax.imshow(V, cmap='cividis')
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()



# ################## MAIN EXECUTION #############################
if __name__ == "__main__":
    app = App()
    app.mainloop()
    Maze = app.A

    states, exit_state, init_states = get_states_from_Maze(Maze)
    num_episodes = 5000

    results = {}
    execution_times = {}

    # --- First Visit MC ---
    V_FV, t = timed_call(MC_First_Visit, num_episodes, states, actions, Maze, exit_state, gamma=0.9)
    results['First-Visit MC'] = build_value_matrix(V_FV, states, Maze.shape)
    execution_times['First-Visit MC'] = t

    # --- Exploring Starts ---
    (policy_ES, V_ES), t = timed_call(MC_Exploring_Starts, num_episodes, states, actions, Maze, exit_state, gamma=0.9)
    results['Exploring Starts'] = build_value_matrix(V_ES, states, Maze.shape)
    execution_times['Exploring Starts'] = t

    # --- On-policy MC ---
    (policy_ON, V_ON), t = timed_call(MC_OnPolicy_FirstVisit_Control, num_episodes, states, actions, Maze, exit_state, gamma=0.9, epsilon=0.1)
    results['On-Policy MC'] = build_value_matrix(V_ON, states, Maze.shape)
    execution_times['On-Policy MC'] = t

    # --- TD(0) ---
    V_TD, t = timed_call(Temporal_Difference, num_episodes, states, actions, Maze, exit_state, gamma=0.9, alpha=0.1)
    results['TD(0)'] = build_value_matrix(V_TD, states, Maze.shape)
    execution_times['TD(0)'] = t

    # --- SARSA ---
    (Q_SARSA, V_SARSA), t = timed_call(SARSA, num_episodes, states, actions, Maze, exit_state, gamma=0.9, alpha=0.1, epsilon=0.1)
    results['SARSA'] = build_value_matrix(V_SARSA, states, Maze.shape)
    execution_times['SARSA'] = t

    # --- n-step TD ---
    V_nTD, t = timed_call(n_step_TD, num_episodes, states, actions, Maze, exit_state, n=3, gamma=0.9, alpha=0.1, policy=None)
    results['n-step TD'] = build_value_matrix(V_nTD, states, Maze.shape)
    execution_times['n-step TD'] = t

    # ---- Best method selection ----
    best_method, scores = evaluate_methods(results)

    print("\nExecution Times (seconds):")
    for k, v in execution_times.items():
        print(f"{k}: {v:.4f}s")

    print("\nAverage Value Score per Method:")
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

    print(f"\nBest Method Based on Mean Value: {best_method}")

    # ---- Plot cleaner grid ----
    plot_all(list(results.values()), list(results.keys()))


