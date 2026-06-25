from classes.grid_env import GridEnv
from classes.state import State
from classes.action import Action
from classes.q_table import QTable
from methods.monte_carlo import monte_carlo
from util.gridworld import run_simulation
from methods.evaluate_policy import sweep_epsilon
from methods.q_learning import q_learning
from methods.sarsa import sarsa


def test_monte_carlo(mdp):

    NUM_EPISODES = 100000
    ENV = GridEnv(mdp)

    # With minimal exploration
    biased_q = QTable(ENV.mdp.all_states, ENV.mdp.all_actions)
    biased_q[State(0, 0), Action.RIGHT] = 0.1
    biased_q[State(1, 0), Action.RIGHT] = 0.1
    biased_q[State(2, 0), Action.UP] = 0.1

    # minimal_exploration_policy, Q = monte_carlo(ENV, NUM_EPISODES, eps_start=0.05, start_q=biased_q)
    # run_simulation(ENV.mdp, minimal_exploration_policy)

    # # With exploration
    # optimal_policy, Q = monte_carlo(ENV, NUM_EPISODES, eps_start=1.0, start_q=biased_q)
    # run_simulation(ENV.mdp, optimal_policy)

    # sweep_epsilon(ENV)
    # opt_policy, Q = q_learning(ENV, 100_000)
    # run_simulation(ENV.mdp, opt_policy)

    sarsa_policy, Q = sarsa(ENV, 100_000)
    run_simulation(ENV.mdp, sarsa_policy)