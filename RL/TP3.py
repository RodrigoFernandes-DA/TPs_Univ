import ForTP3 as ForTP3
import numpy as np
import matplotlib.pyplot as plt
from Maze_generating_interface import App



# def first_visit_mc_prediction(num_episodes, states, actions, Maze, exit_state, gamma=1.0):
#     """
#     Estimate V(s) for all states using First-Visit Monte Carlo Prediction.
#     """
#     # Initialize value function and returns memory
#     V = {tuple(s): 0.0 for s in states}
#     returns = {tuple(s): [] for s in states}

#     for ep in range(num_episodes):
#         # Choose random initial state
#         init_state = states[np.random.randint(len(states))]
#         episode = ForTP3.generate_episode(init_state, actions, Maze, exit_state, itermax=100*len(states))

#         # Extract state sequence
#         states_in_episode = [tuple(x[0]) for x in episode]

#         for t, (state, action, reward, next_state) in enumerate(episode):
#             state_tuple = tuple(state)
#             # If first visit of this state in this episode
#             if state_tuple not in states_in_episode[:t]:
#                 # Compute return G_t
#                 G = 0.0
#                 discount = 1.0
#                 for k in range(t, len(episode)):
#                     G += discount * episode[k][2]  # reward
#                     discount *= gamma
                
#                 returns[state_tuple].append(G)
#                 V[state_tuple] = np.mean(returns[state_tuple])

#     return V


def MC_First_Visit(num_episodes, states, actions, Maze, exit_state, gamma=0.9):

    V = {tuple(s): 0.0 for s in states}
    returns = {tuple(s): [] for s in states}
    states,exit_state,init_states = ForTP3.get_states_from_Maze(Maze)
    for it in range(num_episodes):
        init_state = init_states[np.random.randint(len(init_states))]
        episode = ForTP3.generate_episode(init_state,actions,Maze,exit_state, itermax=100*len(states))

        states_visited = []
        returns = []
        V = []

        for t, (state, action, reward, next_state) in enumerate(episode.reverse()):
            if state not in states_visited:
                states_visited.append(state)
                G = 0.0
                discount = 1.0
                for k in range(t, len(episode)):
                    G += discount * episode[k][2]  # reward
                    discount *= gamma
                print(state_tuple)
                returns[state].append(G)
                V[state,] = np.mean(returns[state])

    return V



if 1 :
    app = App()
    app.mainloop()
    Maze=app.A

plt.imshow(Maze,cmap='Blues')
states,exit_state,init_states = ForTP3.get_states_from_Maze(Maze)

actions = ['UP', 'RIGHT', 'DOWN', 'LEFT']

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
