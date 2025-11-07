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

def generate_episode(init_state,actions,Maze,exit_state,itermax=1000) :
     #from init_state to exit_state in maze M
     # list of (state, action, reward, next_state)
     i=0
     current_state = np.copy(init_state)
     episode = list()
     while  i < itermax and not all(current_state == exit_state)  :      
         current_actions = get_actions(current_state,actions) # define available movements
         action = current_actions[np.random.randint(len(current_actions))] # choose a random move
         new_state, reward= next_state(current_state, action, exit_state, Maze) # move
         #
         episode.append((current_state,action, reward, new_state)) #add movement to list
         #
         current_state = new_state 
         i=i+1
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
            state, action, reward, next_state = episode[t]
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



if 1 :
    app = App()
    app.mainloop()
    Maze=app.A

plt.imshow(Maze,cmap='Blues')
states,exit_state,init_states = get_states_from_Maze(Maze)
plt.show()

init_state = init_states[np.random.randint(len(init_states))]
episode = generate_episode(init_state,actions,Maze,exit_state, itermax=100*len(states))

num_episodes = 5000
V = MC_First_Visit(num_episodes, states, actions, Maze, exit_state, gamma=0.9)

# Convert V to array for visualization
V_matrix = np.zeros(Maze.shape)
for s in states:
    V_matrix[s[0], s[1]] = V[tuple(s)]

plt.figure()
plt.title("State Value Function (First-Visit MC Prediction)")
plt.imshow(V_matrix, cmap='viridis')
plt.colorbar()
plt.show()