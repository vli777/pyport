import random

import numpy as np

from utils.performance_metrics import sharpe_ratio

def simulated_annealing(mu, cov, initial_temp=100, min_temp=1e-4, alpha=0.99, max_iters=1000):
    """
    Simulated Annealing for portfolio optimization to maximize Sharpe ratio.
    
    Parameters:
    - mu: Expected returns (np.array)
    - cov: Covariance matrix (np.array)
    - initial_temp: Initial temperature (higher = more exploration)
    - min_temp: Stopping temperature
    - alpha: Cooling rate (0.99 means 1% decrease per step)
    - max_iters: Max number of iterations
    
    Returns:
    - Optimal portfolio weights
    """
    n = len(mu)  # Number of assets
    w = np.ones(n) / n  # Start with equal weights
    best_w = w.copy()
    best_sharpe = sharpe_ratio(w, mu, cov)
    
    temp = initial_temp
    
    for i in range(max_iters):
        # Generate a new candidate solution (perturb weights slightly)
        new_w = w + np.random.uniform(-0.05, 0.05, size=n)  # Small weight shifts
        new_w = np.clip(new_w, 0, 0.2)  # Ensure weights stay within [0, 0.2]
        new_w /= new_w.sum()  # Normalize to sum to 1

        new_sharpe = sharpe_ratio(new_w, mu, cov)

        # Accept new solution if it's better
        if new_sharpe > best_sharpe:
            best_w = new_w.copy()
            best_sharpe = new_sharpe
        else:
            # Accept worse solutions with probability P = exp(-Δ/T)
            delta = new_sharpe - best_sharpe
            acceptance_prob = np.exp(delta / temp)
            if random.random() < acceptance_prob:
                best_w = new_w.copy()
                best_sharpe = new_sharpe

        # Reduce temperature
        temp *= alpha
        if temp < min_temp:
            break  # Stop when temperature is low

    return best_w
