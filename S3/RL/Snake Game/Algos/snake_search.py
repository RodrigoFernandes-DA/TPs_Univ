import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from snake_game import SnakeGameEnv 
import snake_algos as algos

env_train = SnakeGameEnv(
    board_size=10,
    n_channel=1,
   # n_target=1,
    render_mode=None
)

def snake_search(algorithm="mc", nb_episodes=1000, itermax=200, n_trials=5):
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict
    
    plt.figure(figsize=(15, 10))
    
    ####### mc
    if algorithm == "mc":
        gamma_values = [0.9, 0.95, 0.99]
        results = defaultdict(list)
        
        for gamma in gamma_values:
            print(f"\nTesting Monte Carlo with gamma={gamma}")
            Q, returns, lengths, _ = algos.run_algorithm(
                algorithm="mc",
                gamma=gamma,
                nb_episodes=nb_episodes,
                itermax=itermax,
                epsilon=0.1
            )
            results[gamma] = returns
            print(gamma, " Average = ", np.average(returns[-1000:]))
            
        plt.subplot(2, 2, 1)
        for gamma, returns in results.items():
            plt.plot(algos.moving_average(returns, 50), label=f'γ={gamma}')
        plt.title('MC On Policy Control - Returns (different γ)')
        plt.xlabel('Episode')
        plt.ylabel('Return (moving avg 50)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        final_returns = [np.mean(returns[-100:]) for returns in results.values()]
        plt.bar([str(g) for g in gamma_values], final_returns)
        plt.title('MC On Policy Control - Final Performance (last 100 eps)')
        plt.xlabel('Gamma (γ)')
        plt.ylabel('Average Return')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        for gamma, returns in results.items():
            plt.plot(algos.moving_average(lengths, 50), label=f'γ={gamma}')
        plt.title('Monte Carlo - Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length (moving avg 50)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    ####### mc_es
    elif algorithm == "mc_es":
        gamma_values = [0.9, 0.95, 0.99]
        results = defaultdict(list)
        
        for gamma in gamma_values:
            print(f"\nTesting MC Exploring Starts with gamma={gamma}")
            Q, returns, lengths, _ = algos.run_algorithm(
                algorithm="mc_es",
                gamma=gamma,
                nb_episodes=nb_episodes,
                itermax=itermax
            )
            results[gamma] = returns
            print(gamma, " Average = ", np.average(returns[-1000:]))
            
        plt.subplot(2, 2, 1)
        for gamma, returns in results.items():
            plt.plot(algos.moving_average(returns, 50), label=f'γ={gamma}')
        plt.title('MC Exploring Starts - Returns')
        plt.xlabel('Episode')
        plt.ylabel('Return (moving avg 50)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        final_returns = [np.mean(returns[-100:]) for returns in results.values()]
        plt.bar([str(g) for g in gamma_values], final_returns)
        plt.title('MC Exploring Starts (last 100 eps)')
        plt.xlabel('Gamma (γ)')
        plt.ylabel('Average Return')
        plt.grid(True, alpha=0.3)
        
    ####### sarsa
    elif algorithm == "sarsa":
        param_combinations = [
            {"alpha": 0.01, "gamma": 0.9},
            {"alpha": 0.05, "gamma": 0.9},
            {"alpha": 0.1, "gamma": 0.95},
            {"alpha": 0.2, "gamma": 0.99},
            {"alpha": 0.1, "gamma": 0.99}
        ]
        
        results = {}
        
        for i, params in enumerate(param_combinations[:n_trials]):
            alpha = params["alpha"]
            gamma = params["gamma"]
            print(f"\nTesting SARSA with alpha={alpha}, gamma={gamma}")
            Q, returns, lengths, _ = algos.run_algorithm(
                algorithm="sarsa",
                alpha=alpha,
                gamma=gamma,
                nb_episodes=nb_episodes,
                itermax=itermax,
                epsilon=0.1
            )
            results[f"α={alpha}, γ={gamma}"] = returns
            print(" Average = ", np.average(returns[-1000:]))
            
        plt.subplot(2, 2, 1)
        for label, returns in results.items():
            plt.plot(algos.moving_average(returns, 50), label=label)
        plt.title('SARSA - Returns (different α/γ)')
        plt.xlabel('Episode')
        plt.ylabel('Return (moving avg 50)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        final_returns = [np.mean(returns[-100:]) for returns in results.values()]
        plt.bar(range(len(results)), final_returns)
        plt.xticks(range(len(results)), results.keys(), rotation=45, ha='right')
        plt.title('SARSA - Final Performance (last 100 eps)')
        plt.ylabel('Average Return')
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        
    ####### qlearnig
    elif algorithm == "td" or algorithm == "qlearning":
        param_combinations = [
            {"alpha": 0.3, "gamma": 0.9},
            {"alpha": 0.4, "gamma": 0.9},
            {"alpha": 0.3, "gamma": 0.95},
            {"alpha": 0.4, "gamma": 0.99},
            {"alpha": 0.1, "gamma": 0.99}
        ]
        
        results = {}
        
        for i, params in enumerate(param_combinations[:n_trials]):
            alpha = params["alpha"]
            gamma = params["gamma"]
            print(f"\nTesting Q-learning with alpha={alpha}, gamma={gamma}")
            Q, returns, lengths, _ = algos.run_algorithm(
                algorithm="td",
                alpha=alpha,
                gamma=gamma,
                nb_episodes=nb_episodes,
                itermax=itermax,
                epsilon=0.1
            )
            results[f"α={alpha}, γ={gamma}"] = returns
            print(" Average = ", np.average(returns[-1000:]))
            
        plt.subplot(2, 2, 1)
        for label, returns in results.items():
            plt.plot(algos.moving_average(returns, 50), label=label)
        plt.title('Q-learning - Returns (different α/γ)')
        plt.xlabel('Episode')
        plt.ylabel('Return (moving avg 50)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        final_returns = [np.mean(returns[-100:]) for returns in results.values()]
        plt.bar(range(len(results)), final_returns)
        plt.xticks(range(len(results)), results.keys(), rotation=45, ha='right')
        plt.title('Q-learning - Final Performance (last 100 eps)')
        plt.ylabel('Average Return')
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        
    ####### nstep
    elif algorithm == "nstep":
        param_combinations = [
            {"alpha": 0.2, "n": 1},
            {"alpha": 0.3, "n": 3},
            {"alpha": 0.2, "n": 5},
            {"alpha": 0.1, "n": 7},
            {"alpha": 0.1, "n": 9}
        ]
        results = {}
        
        for i, params in enumerate(param_combinations[:n_trials]):
            alpha = params["alpha"]
            n = params["n"]
            print(f"\nTesting n-step with alpha={alpha}, n={n}")
            Q, returns, lengths, _ = algos.run_algorithm(
                algorithm="nstep",
                n=n,
                alpha=alpha,
                gamma=0.9,
                nb_episodes=nb_episodes,
                itermax=itermax,
                epsilon=0.1
            )
            results[f"α={alpha}, n={n}"] = returns
            print(" Average = ", np.average(returns[-1000:]))
            
        plt.subplot(2, 2, 1)
        for label, returns in results.items():
            plt.plot(algos.moving_average(returns, 50), label=label)
        plt.title(f'n-step TD - Returns (different n values)')
        plt.xlabel('Episode')
        plt.ylabel('Return (moving avg 50)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        final_returns = [np.mean(returns[-100:]) for returns in results.values()]
        plt.bar(range(len(results)), final_returns)
        plt.xticks(range(len(results)), results.keys())
        plt.title('n-step TD - Final Performance (last 100 eps)')
        plt.ylabel('Average Return')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Parameter Search for {algorithm.upper()} Algorithm', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return results

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    obs, info = env_train.reset()
    
    print("\n" + "="*50)
    
    results = snake_search(
        algorithm="nstep", # "mc", "mc_es", "sarsa", "td", "nstep"
        nb_episodes=500, 
        itermax=200,
        n_trials=5
    )
