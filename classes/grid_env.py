import numpy as np
from classes.grid_mdp import GridMDP
from classes.state import State
from classes.action import Action
import random


class GridEnv:
    """A reinforcement learning environment to interact with the Grid World and sample from it."""

    def __init__(self, mdp: GridMDP):
        """Initializes the GridEnv."""
        self.mdp = mdp
        self.state = mdp.start
        self.terminated = False

    def reset(self) -> State:
        """Resets the state to the start state, not terminated, and returns the new start state."""
        self.state = self.mdp.start
        self.terminated = False
        return self.state

    def step(self, action: Action) -> tuple[State, float, bool]:
        """Performs a step in the environment given the selected `action` by the agent.

        Returns a tuple of (next_state, reward, is_done), with the next state after the action has
        been taked, the associated reward, and whether the episode terminated.
        """
        if self.terminated:
            raise Exception("Environment episode completed, please call reset.")

        #: Compute the state probabilities for the action, sample the probabilities to select
        #       the next state, calculate the reward, determine if the next_state is terminal, and
        #       return the tuple (next_state, reward, done).
        transition_probabilities = self.mdp.transition(self.state, action)
        items = list(transition_probabilities.keys())
        weights = list(transition_probabilities.values())
        next_state = random.choices(items, weights=weights, k=1)[0]
        reward = self.mdp.reward(self.state, action, next_state)
        self.state = next_state
        self.terminated = self.mdp.is_terminal(self.state)
        return (self.state, reward, self.terminated)
