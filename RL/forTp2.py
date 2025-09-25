#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:39:44 2020

@author: berar
"""
import numpy as np


GRID_SIZE = 4
TERMINAL_STATES = [0, GRID_SIZE*GRID_SIZE-1]
states = np.arange(GRID_SIZE*GRID_SIZE)
actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
discount = 1.

def next_state(grid_size, state, action):
    i,j = np.unravel_index(state, (grid_size, grid_size))
    if action == 'UP':
        i = np.maximum(0,i-1)
    elif action == 'DOWN':
        i = np.minimum(i+1,grid_size-1)
    elif action == 'RIGHT':
        j = np.minimum(j+1,grid_size-1)
    elif action == 'LEFT':
        j = np.maximum(0,j-1)    
    new_state = np.ravel_multi_index((i,j), (grid_size, grid_size))
    return new_state
    
def is_done(state, terminal_states):
    return state in terminal_states


# The unifom policy            
uniform_policy = {s : { a : 1/len(actions) for a in actions } for s in states}

print(f"Uni = {uniform_policy}")

# Transition is coded as a dictionary of dictionary
P = {}
for s in range(len(states)):
    P[s] = {a : () for a in actions}
    if s in TERMINAL_STATES:
        # if terminal state, stay where you are
        # instead of next_state
        reward = 0.
        for action in actions:
            P[s][action] = (s, reward, True)
    else:
        # transition
        reward = -1.
        for action in actions:
            next_s = next_state(GRID_SIZE, s, action)
            P[s][action] = (next_s,reward,is_done(next_s, TERMINAL_STATES))
 



def Policy_Evaluation(K, states, uniform_policy):
                      
    values = np.arange(GRID_SIZE*GRID_SIZE)

    for k in range(K):
        for state in states:
            vnew = 
            values[state] = vnew




for s in range(len(values)):
    
    if s in TERMINAL_STATES:
        reward = 0.
        for action in actions:
            P[s][action] = (s, reward, True)
    else:
        values[i, j]
 
