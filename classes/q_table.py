import random
from classes.state import State
from classes.action import Action


class QTable:
    """
    A Q table stores the current beliefs of the value of each action from each state.
    
    Table storing Q, i.e., the state-action values.

    This is not an optimal implementation, but hopefully useful for learning purposes.
    """

    def __init__(self, states: list[State], actions: list[Action]):
        """Initializes the QTable given all states and actions of the MDP."""
        self.states = states
        self.actions = actions
        self.num_actions = len(actions)
        # Why a dict of dict instead of a single dict indexed by tuple (State, Action)? Just for
        # convenience when looking up all action values for a state.
        self.table: dict[State, dict[Action, float]] = {
            s: {a: 0.0 for a in actions} for s in states
        }

    def __getitem__(self, key: tuple[State, Action]) -> float:
        """Returns the Q value for a tuple of (state, action)."""
        # Return the corresponding value in the internal table.
        state = key[0]
        action = key[1]
        return self.table[state][action]

    def __setitem__(self, key: tuple[State, Action], value: float):
        """Set the Q value for a tuple of (state, action)."""
        # Set the value in the internal table.
        state = key[0]
        action = key[1]
        self.table[state][action] = value

    def value(self, state: State) -> float:
        """Returns the value of a state."""
        actions_and_values = self.table[state]
        values = list(actions_and_values.values())
        max_value = max(values)
        return max_value

    def best_action(self, state: State) -> Action:
        """Returns the best action for a certain state."""
        best_action = None
        best_value = float("-inf")
        actions = list(self.table[state].keys())
        # Use random.shuffle to shuffle the actions (in case multiple have the same value).
        random.shuffle(actions)
        for action in actions:
            # Search for the best value.
            value = self.table[state][action]
            if value > best_value:
                best_action = action
                best_value = value
        return best_action

    def random_action(self, state: State) -> Action:
        actions = list(self.table[state].keys())
        # Use random.shuffle to shuffle the actions (in case multiple have the same value).
        random.shuffle(actions)
        return actions[0]
