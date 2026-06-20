from classes.grid_mdp import GridMDP
from classes.q_table import QTable
from classes.state import State
from classes.action import Action
from init.init_simulation import init_random
# from util.gridworld import plot_grid



def value_iteration(mdp: GridMDP, noise=0.0, n_iterations=100) -> QTable:
    """Runs the value iteration algorithm for the Grid World MDP"""
    # Initialize the QTable with the states and actions coming from the MDP.
    q_table = QTable(mdp.all_states, mdp.all_actions)

    for _ in range(0, n_iterations):
        # TODO: Create a new QTable to store the updated values.
        new_q_table = QTable(mdp.all_states, mdp.all_actions)
        # Loop over all states, over all actions, get the transition probabilities, get the
        #       reward via the MDP reward function, and update the new_qtable according to the
        #       update-rule defined above (i.e., prob * (reward + gamma * state_value))
        for state in mdp.all_states:
            for action in mdp.all_actions:
                transition_probabilities = mdp.transition(state, action, noise)
                for next_state, probability in transition_probabilities.items():
                    # P => probabililty of going to the nex state
                    # R => the immediate reward for that next state (bomb or target or 0.0)
                    reward = mdp.reward(state, action, next_state)
                    # V => the next state's expected further/future value based on the q table (policy)
                    value = q_table.value(next_state)
                    # gamma => the discount factor for that further/future value
                    # Q = P * (R + yV)
                    # New value of state, action => probability of taking it * immediate reward + discounted reward
                    new_q_value = probability * (reward + (mdp.gamma * value))
                    # That action occurs with some probability and contributes its immediate and future value to the state
                    new_q_table[state, action] += new_q_value

        # Swap the qtable with the updated new_qtable.
        q_table = new_q_table

    return q_table


def test_value_iteration(mdp: GridMDP):
    # Test our implementation!
    init_random()
    qtable = value_iteration(mdp, noise=0.2, n_iterations=100)
    print(qtable.value(State(0, 0)))
    assert f"{qtable.value(State(0, 0)):.2f}" == "0.55"
    assert f"{qtable.value(State(1, 0)):.2f}" == "0.48"
    assert f"{qtable.value(State(2, 0)):.2f}" == "0.53"
    assert f"{qtable.value(State(3, 0)):.2f}" == "0.31"
    assert f"{qtable.value(State(0, 1)):.2f}" == "0.63"
    assert f"{qtable.value(State(1, 1)):.2f}" == "0.00"
    assert f"{qtable.value(State(2, 1)):.2f}" == "0.64"
    assert f"{qtable.value(State(3, 1)):.2f}" == "0.00"
    assert f"{qtable.value(State(0, 2)):.2f}" == "0.72"
    assert f"{qtable.value(State(1, 2)):.2f}" == "0.83"
    assert f"{qtable.value(State(2, 2)):.2f}" == "0.94"
    assert f"{qtable.value(State(3, 2)):.2f}" == "0.00"

    # Using provided utilities, we can print the state values of the Grid World!
    # _, _ = plot_grid(mdp.grid, qtable)