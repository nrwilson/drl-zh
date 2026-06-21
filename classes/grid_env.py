import numpy as np
from classes.grid_mdp import GridMDP
from classes.state import State
from classes.action import Action
from classes.q_table import QTable
import random


class GridEnv:
    """
    A GridEnv lets you interact with a GridMDP and take actions on it and sample rewards.
    A reinforcement learning environment to interact with the Grid World and sample from it."""

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

    def step(self, action: Action, noise: float=0.2) -> tuple[State, float, bool]:
        """Performs a step in the environment given the selected `action` by the agent.

        Returns a tuple of (next_state, reward, is_done), with the next state after the action has
        been taked, the associated reward, and whether the episode terminated.
        """
        if self.terminated:
            raise Exception("Environment episode completed, please call reset.")

        #: Compute the state probabilities for the action, sample the probabilities to select
        #       the next state, calculate the reward, determine if the next_state is terminal, and
        #       return the tuple (next_state, reward, done).
        next_state = self.mdp.resolve_next_state(self.state, action, noise)
        reward = self.mdp.reward(self.state, action, next_state)
        self.state = next_state
        self.terminated = self.mdp.is_terminal(self.state)
        return (self.state, reward, self.terminated)

    def random_action_step(self, q_table: QTable, noise=0.0):
        state = self.state
        target_action = q_table.random_action(state)
        result = self.step(target_action, noise)
        return result

    def best_action_step(self, q_table: QTable, noise):
        state = self.state
        target_action = q_table.best_action(state)
        result = self.step(target_action, noise)
        return result
