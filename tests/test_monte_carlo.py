from classes.grid_env import GridEnv
from classes.state import State
from classes.action import Action
from classes.q_table import QTable
from methods.monte_carlo import monte_carlo
from util.gridworld import run_simulation


def test_monte_carlo(mdp):

    NUM_EPISODES = 10000
    ENV = GridEnv(mdp)

    # With minimal exploration
    biased_q = QTable(ENV.mdp.all_states, ENV.mdp.all_actions)
    biased_q[State(0, 0), Action.RIGHT] = 0.1
    biased_q[State(1, 0), Action.RIGHT] = 0.1
    biased_q[State(2, 0), Action.UP] = 0.1

    minimal_exploration_policy, Q = monte_carlo(ENV, NUM_EPISODES, eps_start=0.05, start_q=biased_q)
    run_simulation(ENV.mdp, minimal_exploration_policy, live=True)
