from classes.grid import Grid
from classes.state import State
from classes.action import Action
from collections import defaultdict
from numpy.testing import assert_almost_equal
from classes.cell import Cell
import random


class GridMDP:
    """
    A GridMDP defines how to interact with a grid, including determining action probabilities.
    Class modeling the Grid World Markov Decision Process (MDP).

    It defines the transition and reward functions, as well as states and gamma factor.
    """

    def __init__(self, grid: Grid, start=State(), gamma=1.0):
        """Initializes the GridMDP."""
        # Accept a grid that has been defined.
        self.grid = grid
        # Accept what state to start in.
        self.start = start
        # Accept the gamma factor.
        self.gamma = gamma
        # Enumerate all available actions.
        self.all_actions = list(Action)
        # Enumerate all possible states. We could filter out the non-reachable states here,
        #       but let's keep things simple and enumerate all of them in the grid.
        states = []
        for j in list(range(grid.height)):
            for i in list(range(grid.width)):
                states.append(State(i, j))
        self.all_states = states

    def observe(self, state: State) -> Cell:
        """Returns what is in the cell at position defined by state."""
        x, y = state.pos()
        return self.grid[x, y]

    def is_terminal(self, state: State) -> bool:
        """Determines if the `state` is a terminal state."""
        cell = self.observe(state)
        is_terminal = cell == Cell.TARGET or cell == Cell.BOMB or cell == Cell.NUKE or cell == Cell.GLORY
        return is_terminal

    def is_reachable(self, state: State) -> bool:
        """Determines if the `state` is reachable."""
        # Return whether the state is reachable. Hint: a state is reachable if it is inside
        #       the grid, and it is not a WALL.
        
        # Make sure it's inside the grid.
        x, y = state.pos()
        x_in_grid = x >= 0 and x < self.grid.width
        y_in_grid = y >= 0 and y < self.grid.height
        inside_grid = x_in_grid and y_in_grid
        if not inside_grid:
            return False

        # Make sure it's not a wall.
        cell = self.observe(state)
        is_wall = cell == Cell.WALL
        if is_wall:
            return False
        
        return True

    def reward(self, state: State, action: Action, next_state: State) -> float:
        """The reward function of the Grid World MDP.

        Note: the reward is intrinsic to the MDP — gamma does NOT appear here. Discounting
        is applied separately inside the Bellman backup (see `value_iteration` below).
        """
        # Compute and return the reward for the (state, action, next_state) tuple. We'll only
        #       use next_state. If the next_state is TARGET, return 1.0. If it is a BOMB, return
        #       -1.0. Else return zero.
        if self.observe(next_state) == Cell.TARGET:
            return 1.0
        elif self.observe(next_state) == Cell.BOMB:
            return -1.0
        elif self.observe(next_state) == Cell.NUKE:
            return -10.0
        elif self.observe(next_state) == Cell.GLORY:
            return 10.0
        else:
            return 0

    def transition(self, state: State, action: Action) -> dict[State, float]:
        return self.get_transition_probabilities(state, action, noise=0.0)

    def get_transition_probabilities(self, state: State, action: Action, noise: float) -> dict[State, float]:
        """Get the next states available and the probabilities of each one being reached."""
        if not self.is_reachable(state) or self.is_terminal(state):
            return {}

        def landing(candidate: State) -> State:
            """Returns the state that we will end up, when attempting to reach another state.

            Basically, if `candidate` is reachable it returns `candidate`, otherwise we stay put.
            """
            return candidate if self.is_reachable(candidate) else state

        def next_state(state: State, action: Action):
            (x, y) = state.pos()
            if action == Action.LEFT:
                next_state = State(x-1, y)
            elif action == Action.UP:
                next_state = State(x, y+1)
            elif action == Action.RIGHT:
                next_state = State(x+1, y)
            elif action == Action.DOWN:
                next_state = State(x, y-1)
            return landing(next_state)

        # Compute the action expected "landings" for all the actions.
        action_landings = {}
        for possible_action in self.all_actions:
            action_landings[possible_action] = next_state(state, possible_action)

        # Compute the direction the agent might go by mistake (instead of the direction of the
        #       chosen action) given the probabilistic Grid World MDP dynamics.
        #       Hint: an action can go wrong only in the orthogonal directions (e.g., the agent
        #       tries to go left, but it ends up going up or down; it cannot go right if it attemps
        #       to go left).
        if action == Action.LEFT:
            mistake_actions = [Action.DOWN, action.UP]
        elif action == Action.UP:
            mistake_actions = [Action.LEFT, Action.RIGHT]
        elif action == Action.RIGHT:
            mistake_actions = [Action.UP, Action.DOWN]
        elif action == Action.DOWN:
            mistake_actions = [Action.LEFT, Action.RIGHT]

        # Compute the next state using the action_landings.
        next_state = action_landings[action]

        # Compute the list of possible "mistake" states given the mistaken_directions.
        mistake_states = []
        for mistake_action in mistake_actions:
            if action_landings[mistake_action]:
                mistake_state = action_landings[mistake_action]
                mistake_states.append(mistake_state)

        # Compute the transition probabilities. Probabilities are zero in every state, besides the
        # next expected state from the chosen action - which has probability (1.0 -noise) - and the
        # potentially "mistake" states which all have probability (noise / n_possible_mistakes).
        probs = defaultdict(lambda: 0.0)

        # Compute the probability of the next expected state.
        probs[next_state] = 1.0 - noise

        # Compute the probability of the mistaken states.
        for m in mistake_states:
            probs[m] += noise / len(mistake_states)
    
        # A couple of assertions to verify the correctness of the probability computation.
        assert sum(probs.values()) <= 1.0
        assert_almost_equal(sum(probs.values()), 1.0)

        return probs

    def resolve_next_state(self, state: State, action: Action, noise: float) -> State:
        transition_probabilities = self.get_transition_probabilities(state, action, noise)
        items = list(transition_probabilities.keys())
        weights = list(transition_probabilities.values())
        next_state = random.choices(items, weights=weights, k=1)[0]
        return next_state
