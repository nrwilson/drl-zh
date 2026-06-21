import random
from classes.q_table import QTable
from methods.policies import Policy
from classes.state import State


def greedy_policy(qtable: QTable) -> Policy:
    """Returns the greedy policy for the specified QTable."""
    # TODO: Return the policy lambda for action selection using the best action from QTable.
    greedy_policy = lambda state: qtable.best_action(state)
    return greedy_policy


def epsilon_greedy_policy(qtable: QTable, epsilon: float) -> Policy:
    """Returns the epsilon-greedy policy for the specified QTable."""

    def choose_action(qtable: QTable, state: State):
        # The probability of the best action is (1 - epsilon + epsilon / nA), while the
        #       other actions have probability (epsilon / nA). Compute those and choose the action.
        #       Use np.random.choice for sampling :)
        actions = qtable.list_actions(state)
        best_action = qtable.best_action(state)
        choices = {}
        for action in actions:
            choices[action] = epsilon / len(actions)
        choices[best_action] = 1 - epsilon + epsilon / len(actions)

        actions = list(choices.keys())
        weights = list(choices.values())
        next_state = random.choices(actions, weights=weights, k=1)[0]
        return next_state

    return lambda state: choose_action(qtable, state)


def epsilon_generator(eps_start=1.0, eps_decay=0.99999, eps_min=0.05):
    """Generator function for Ɛ and its decay (e.g., exploration via Ɛ-greedy policy)."""
    eps = eps_start
    while True:
        # TODO: Yield and update eps, selecting the max(eps * decay, eps_min).
        yield(eps)
        eps = max(eps * eps_decay, eps_min)
