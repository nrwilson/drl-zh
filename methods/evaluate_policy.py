from tkinter import Grid
from classes.grid_env import GridEnv
from classes.q_table import QTable
from classes.action import Action
from classes.state import State
from methods.policies import Policy
from methods.monte_carlo import monte_carlo, generate_episode, unpack_episode
from methods.greedy_policy import epsilon_greedy_policy
from methods.q_learning import q_learning
from methods.sarsa import sarsa
import numpy as np
import random


def sweep_epsilon(env: GridEnv, n_seeds=10):
    # Sweep epsilon start, averaging each value over `n_seeds` independent training runs.
    # A single MC run is high-variance (it lands on Glory or Target almost like a coin flip),
    # so we average to recover the underlying trend. Seed once up front, not per run.
    random.seed(0)
    np.random.seed(0)
    print(f"{'eps_start':>10} | {'avg_return':>10} | {'Glory':>6} | {'Target':>6} | {'trunc':>6}")
    print("-" * 55)
    for eps_start in [0.0, 0.05, 0.3, 1.0]:
        avgs, counts = [], {"Glory": 0, "Target": 0, "truncated": 0}
        for _ in range(n_seeds):
            policy, _ = monte_carlo(env, num_episodes=20000, alpha=0.02, eps_start=eps_start)
            avgs.append(evaluate_policy(env, policy))
            for k, v in terminal_breakdown(env, policy).items():
                counts[k] += v
        # Counts are summed over n_seeds * n_eval rollouts; report the per-run average.
        print(
            f"\r{eps_start:>10.2f} | {np.mean(avgs):>10.3f} | "
            f"{counts['Glory'] / n_seeds:>6.0f} | {counts['Target'] / n_seeds:>6.0f} | "
            f"{counts['truncated'] / n_seeds:>6.0f}"
        )


def evaluate_policy(env: GridEnv, policy: Policy, n_eval=200, max_t=20) -> float:
    """Rollout `n_eval` episodes under `policy` and return the average (undiscounted) return."""
    returns = []
    for _ in range(n_eval):
        # Generate an episode following `policy` and sum its rewards into `total_return`.
        episode = generate_episode(env, policy, noise=0, max_t=max_t)
        _, _, rewards = unpack_episode(episode)
        total_return = np.sum(rewards)
        returns.append(total_return)
    return float(np.mean(returns))


def terminal_breakdown(env: GridEnv, policy: Policy, n_eval=200, max_t=20) -> dict[str, int]:
    """Roll out `n_eval` episodes and tally which terminal (if any) they land on.

    The rewards are fixed (Target=+1.0, Glory=+10.0, Nuke=-10.0, else 0) and the episode breaks
    as soon as the environment reports `done`, so the *last* reward in each episode is the
    terminal reward -- or ~0 if the episode was truncated by `max_t` without terminating.
    """
    counts = {"Glory": 0, "Target": 0, "truncated": 0}
    for _ in range(n_eval):
        episode = generate_episode(env, policy, max_t=max_t)
        _, _, rewards = unpack_episode(episode)
        # Look at the last reward of the episode and increment the matching bucket in
        #       `counts`. Use abs(last_reward - 10.0) < 0.1 to detect Glory, similarly for Target
        #       with 1.0; everything else is a truncated/no-terminal episode.
        last_reward = rewards[-1]
        if abs(last_reward - 10.0) < 0.1:
            counts["Glory"] += 1
        elif abs(last_reward - 1.0) < 0.1:
            counts["Target"] += 1
        else:
            counts["truncated"] += 1
    return counts


def compare_algorithms(env: GridEnv):

    NUM_EPISODES = 30_000
    results = []
    # TODO: replace each `None` with a lambda that trains the corresponding algorithm on ENV for
    #       NUM_EPISODES. For MC, pass `start_q=biased_q` to reuse the starting Q-table from earlier.
    biased_q = QTable(env.mdp.all_states, env.mdp.all_actions)
    biased_q[State(0, 0), Action.RIGHT] = 0.1
    biased_q[State(1, 0), Action.RIGHT] = 0.1
    biased_q[State(2, 0), Action.UP] = 0.1
    for name, fn in [
        ("Monte Carlo", monte_carlo(env, NUM_EPISODES, 0.02, 1.0, biased_q)),
        ("Q-Learning", q_learning(env, NUM_EPISODES)),
        ("SARSA", sarsa(env, NUM_EPISODES)),
    ]:
        policy, _ = fn()
        results.append((name, evaluate_policy(env, policy), terminal_breakdown(env, policy)))

    print()
    print(f"{'Algorithm':<15} | {'Avg. Return':>12} | {'Glory':>6} | {'Target':>6} | {'trunc':>6}")
    print("-" * 60)
    for name, ret, br in results:
        print(
            f"{name:<15} | {ret:>12.3f} | "
            f"{br['Glory']:>6d} | {br['Target']:>6d} | {br['truncated']:>6d}"
        )