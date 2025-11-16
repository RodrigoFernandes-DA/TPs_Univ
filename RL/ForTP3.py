#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:02:58 2020

@author: berar
"""
import numpy as np
import matplotlib.pyplot as plt
from Maze_generating_interface import App

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
            # if state not in states_visited:
            if state_tuple not in states_in_episode[:t]:
                G = 0.0
                discount = 1.0
                for k in range(t, len(episode)):
                    G += discount * episode[k][2]  # reward
                    discount *= gamma
                returns[state_tuple].append(G)
                V[state_tuple] = np.mean(returns[state_tuple])

    return V


def MC_Exploring_Starts(num_episodes, states, actions, Maze, exit_state, gamma=0.9):

    Q = {(tuple(s), a): 0.0 for s in states for a in actions}
    returns = {(tuple(s), a): [] for s in states for a in actions}
    policy = {tuple(s): np.random.choice(actions) for s in states}

    for episode_num in range(num_episodes):
        # Exploring start: choose random state and random action
        s0 = states[np.random.randint(len(states))]
        possible_actions = get_actions(s0, actions)
        if len(possible_actions) == 0:
            continue
        a0 = np.random.choice(possible_actions)

        # Generate episode following current policy after the first action
        episode = []
        current_state = np.copy(s0)
        current_action = a0
        for t in range(1000):
            new_state, reward = next_state(current_state, current_action, exit_state, Maze)
            episode.append((current_state, current_action, reward, new_state))
            if all(new_state == exit_state):
                break
            current_state = new_state
            current_action = policy[tuple(current_state)]
            if current_action not in get_actions(current_state, actions):
                break

        # Compute returns and update Q and policy
        G = 0.0
        for t in reversed(range(len(episode))):
            state, action, reward, next_state2 = episode[t]
            G = gamma * G + reward
            if not any((tuple(state) == tuple(x[0]) and action == x[1]) for x in episode[:t]):
                returns[(tuple(state), action)].append(G)
                Q[(tuple(state), action)] = np.mean(returns[(tuple(state), action)])
                # Greedy policy improvement
                possible_actions = get_actions(state, actions)
                if possible_actions:
                    best_a = max(possible_actions, key=lambda a: Q[(tuple(state), a)])
                    policy[tuple(state)] = best_a

    # Derive V from Q
    V = {tuple(s): max([Q[(tuple(s), a)] for a in actions]) for s in states}
    return policy, V


def MC_OnPolicy_FirstVisit_Control(num_episodes, states, actions, Maze, exit_state, gamma=0.9, epsilon=0.1):
    """
    On-Policy First-Visit MC Control with ε-soft policy
    """
    Q = {(tuple(s), a): 0.0 for s in states for a in actions}
    returns = {(tuple(s), a): [] for s in states for a in actions}
    policy = {tuple(s): np.random.choice(actions) for s in states}

    for episode_num in range(num_episodes):
        s0 = states[np.random.randint(len(states))]
        init_states, _, _ = get_states_from_Maze(Maze)
        episode = generate_episode(s0, actions, Maze, exit_state, policy, epsilon, itermax=100*len(states))

        G = 0.0
        for t in reversed(range(len(episode))):
            state, action, reward, next_state2 = episode[t]
            G = gamma * G + reward
            if not any((tuple(state) == tuple(x[0]) and action == x[1]) for x in episode[:t]):
                returns[(tuple(state), action)].append(G)
                Q[(tuple(state), action)] = np.mean(returns[(tuple(state), action)])

                # Improve policy to be ε-greedy w.r.t. Q
                possible_actions = get_actions(state, actions)
                if possible_actions:
                    best_a = max(possible_actions, key=lambda a: Q[(tuple(state), a)])
                    policy[tuple(state)] = best_a

    V = {tuple(s): max([Q[(tuple(s), a)] for a in actions]) for s in states}
    return policy, V


def Temporal_Difference(states, actions, Maze, exit_state):
    V = {tuple(s): 0.0 for s in states}
    
    for it in range(num_episodes):
        init_state = init_states[np.random.randint(len(init_states))]
        episode = generate_episode(init_state,actions,Maze,exit_state, itermax=100*len(states))
        # print(episode)
        states_in_episode = [tuple(x[0]) for x in episode]
    

################## MAIN EXECUTION #############################

if __name__ == "__main__":
    app = App()
    app.mainloop()
    Maze = app.A

    plt.imshow(Maze, cmap='Blues')
    states, exit_state, init_states = get_states_from_Maze(Maze)
    plt.show()

    num_episodes = 5000

    # --- First Visit MC Prediction ---
    V_FV = MC_First_Visit(num_episodes, states, actions, Maze, exit_state, gamma=0.9)
    V_matrix_FV = np.zeros(Maze.shape)
    for s in states:
        V_matrix_FV[s[0], s[1]] = V_FV[tuple(s)]

    # --- MC Exploring Starts ---
    policy_ES, V_ES = MC_Exploring_Starts(num_episodes, states, actions, Maze, exit_state, gamma=0.9)
    V_matrix_ES = np.zeros(Maze.shape)
    for s in states:
        V_matrix_ES[s[0], s[1]] = V_ES[tuple(s)]

    # --- On-Policy First-Visit MC Control ---
    policy_ON, V_ON = MC_OnPolicy_FirstVisit_Control(num_episodes, states, actions, Maze, exit_state, gamma=0.9, epsilon=0.1)
    V_matrix_ON = np.zeros(Maze.shape)
    for s in states:
        V_matrix_ON[s[0], s[1]] = V_ON[tuple(s)]

    # --- Visualization ---
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("First-Visit MC Prediction")
    plt.imshow(V_matrix_FV, cmap='cividis')
    # plt.imshow(V_matrix_FV, cmap='viridis')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("MC Exploring Starts Control")
    plt.imshow(V_matrix_ES, cmap='cividis')
    # plt.imshow(V_matrix_ES, cmap='plasma')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("On-Policy First-Visit MC Control (ε=0.1)")
    plt.imshow(V_matrix_ON, cmap='cividis')
    # plt.imshow(V_matrix_ON, cmap='cividis')
    plt.colorbar()
    plt.show()

    # --- Comparison ---
    diff_ES_ON = np.abs(V_matrix_ES - V_matrix_ON)
    plt.figure()
    plt.title("Difference Between MC-ES and On-Policy MC Control")
    plt.imshow(diff_ES_ON, cmap='coolwarm')
    plt.colorbar()
    plt.show()