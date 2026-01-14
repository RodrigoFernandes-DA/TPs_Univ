import time
import numpy as np
import matplotlib.pyplot as plt
from snake_algos import *

# =========================
# Environment 
# =========================
env_train = SnakeGameEnv(
    board_size=10,
    n_channel=1,
    render_mode=None
)

env_view = SnakeGameEnv(
    board_size=10,
    n_channel=1,
    render_mode="human"
)

# =========================
# Functions
# =========================
def moving_average(x, window=50):
    x = np.array(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")

def run_algorithm_multiple_times(algo, params, n_runs=10, nb_episodes=2000, itermax=200):
    all_returns = []
    all_lengths = []
    all_times = []
    
    print(f"  Running {n_runs} times...")
    
    for run in range(n_runs):
        print(f"    Run {run + 1}/{n_runs}", end="\r")
        start_time = time.time()
        
        Q, returns, lengths, _ = run_algorithm(
            algorithm=algo,
            nb_episodes=nb_episodes,
            itermax=itermax,
            **params
        )
        
        training_time = time.time() - start_time
        all_returns.append(returns)
        all_lengths.append(lengths)
        all_times.append(training_time)
    
    print(f"    Completed {n_runs} runs            ")
    
    return all_returns, all_lengths, all_times

def compare_algorithms(n_runs=10, nb_episodes=2000, itermax=200, window=50):
    algorithms = ["mc", "mc_es", "sarsa", "td", "nstep"]
    
    algo_display_names = {
        "mc": "MC On Policy ",
        "mc_es": "MC Exploring Starts",
        "sarsa": "SARSA",
        "td": "Q-learning",
        "nstep": "n-step TD"
    }
    
    algo_params = {
        "mc": {"gamma": 0.95, "epsilon": 0.1},
        "mc_es": {"gamma": 0.95},
        "sarsa": {"alpha": 0.1, "gamma": 0.95, "epsilon": 0.1},
        "td": {"alpha": 0.3, "gamma": 0.95, "epsilon": 0.1},
        "nstep": {"n": 7, "alpha": 0.1, "gamma": 0.9, "epsilon": 0.1}
    }

    results = {}
    
    print("=" * 70)
    print(f"COMPARING 5 REINFORCEMENT LEARNING ALGORITHMS")
    print(f"Training each algorithm {n_runs} times with {nb_episodes} episodes per run")
    print("=" * 70)
    
    for algo in algorithms:
        display_name = algo_display_names[algo]
        params = algo_params[algo]
        
        print(f"\nTraining {display_name}...")
        
        all_returns, all_lengths, all_times = run_algorithm_multiple_times(
            algo, params, n_runs, nb_episodes, itermax
        )
        
        avg_returns = np.mean(all_returns, axis=0)
        std_returns = np.std(all_returns, axis=0)
        
        avg_lengths = np.mean(all_lengths, axis=0)
        
        # Calculate final performance 
        if len(avg_returns) >= 100:
            final_perf_mean = np.mean(avg_returns[-100:])
            # Calculate final performance for each run and get stats
            run_final_perfs = []
            for run_returns in all_returns:
                if len(run_returns) >= 100:
                    run_final_perfs.append(np.mean(run_returns[-100:]))
            final_perf_std = np.std(run_final_perfs) if run_final_perfs else 0
        else:
            final_perf_mean = np.mean(avg_returns)
            final_perf_std = 0
        
        # Store results
        results[display_name] = {
            'all_returns': all_returns,
            'all_lengths': all_lengths,
            'avg_returns': avg_returns,
            'std_returns': std_returns,
            'avg_lengths': avg_lengths,
            'avg_time': np.mean(all_times),
            'final_perf_mean': final_perf_mean,
            'final_perf_std': final_perf_std,
            'max_return': np.max(avg_returns)
        }
        
        print(f"  Average training time: {np.mean(all_times):.1f}s")
        print(f"  Final performance (last 100 eps): {final_perf_mean:.2f} ± {final_perf_std:.2f}")
        print(f"  Max average return: {np.max(avg_returns):.2f}")
        print(f"  Average episode length: {np.mean(avg_lengths):.1f}")
    
    for display_name, data in results.items():
        print(f"{display_name:<25} {data['avg_time']:<12.1f} "
              f"{data['final_perf_mean']:.2f} ± {data['final_perf_std']:.2f} "
              f"{data['max_return']:<12.2f}")
    
    best_algo = None
    best_perf = -float('inf')
    
    for display_name, data in results.items():
        if data['final_perf_mean'] > best_perf:
            best_perf = data['final_perf_mean']
            best_algo = display_name
    
    print(f"\nBest Algorithm: {best_algo} (avg return: {best_perf:.2f})")
    
    plt.figure(figsize=(14, 6))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    plt.subplot(1, 2, 1)
    
    for (display_name, data), color in zip(results.items(), colors):
        avg_returns = data['avg_returns']
        std_returns = data['std_returns']
        
        smoothed_avg = moving_average(avg_returns, window)
        smoothed_std = moving_average(std_returns, window)
        
        plt.plot(smoothed_avg, label=display_name, color=color, linewidth=2)
    
    plt.title(f'Returns (Moving Average, window={window})\nMean ± Std over {n_runs} runs')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    
    algo_names = []
    final_performances = []
    final_errors = []
    
    for display_name, data in results.items():
        algo_names.append(display_name)
        final_performances.append(data['final_perf_mean'])
        final_errors.append(data['final_perf_std'])
    
    x_pos = np.arange(len(algo_names))
    bars = plt.bar(x_pos, final_performances, 
                   color=colors[:len(algo_names)],
                   yerr=final_errors,
                   capsize=5,
                   error_kw={'elinewidth': 2, 'capthick': 2})
    
    plt.title('Final Performance (Last 100 Episodes)\nMean ± Std over 10 runs')
    plt.ylabel('Average Return')
    plt.xticks(x_pos, algo_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    for bar, value, error in zip(bars, final_performances, final_errors):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + error + 0.5,
                f'{value:.1f} ± {error:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Algorithm Comparison ({n_runs} runs, {nb_episodes} episodes each)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    
    print(f"\nTraining {best_algo} agent for viewing...")
    
    algo_key = next(key for key, name in algo_display_names.items() if name == best_algo)
    params = algo_params[algo_key]
    
    Q, returns, lengths, _ = run_algorithm(
        algorithm=algo_key,
        nb_episodes=nb_episodes,
        itermax=itermax,
        **params
    )
    
    print(f"Watching {best_algo} agent...")
    watch_agent(env_view, Q, n=3, itermax=itermax, sleep=0.1)
    
    return results, best_algo

# =========================
# Main 
# =========================
if __name__ == "__main__":
    obs, info = env_train.reset()
    
    print("\n" + "="*70)
    print("STARTING COMPARISON")
    print("="*70)
    
    results, best_algo = compare_algorithms(
        n_runs=10,
        nb_episodes=2000,
        itermax=200,
        window=50
    )