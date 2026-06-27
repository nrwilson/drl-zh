import random
import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, RecordVideo

from util.gymnastics import DEVICE, init_random, plot_scores, show_gym_video_recording


# Let's learn Gymnasium! https://ale.farama.org/environments/pong/
def gym_simulate(agent=None):
    """Runs an Atari pong game with our agent passed as input."""
    # We use pong deterministic b/c it is simpler to train.
    # Actions: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
    # Use gym.make to create the "ALE/Pong-v5" environment. Also, pass the render_mode
    #       'rgb_array_list', frameskip=1, and repeat_action_probability=0.0 to make sure we can
    #       record the video and the play is deterministic.
    sim_env = gym.make("ALE/Pong-v5", render_mode="rgb_array_list", frameskip=1, repeat_action_probability=0.0)
    # Initializes the random generators for determinism.
    sim_env = init_random(sim_env)
    # Add the preprocessor for Atari. This does black/white conversion, and other convenient
    #       operations for learning (see documentation).
    sim_env = AtariPreprocessing(sim_env)
    # Adding the RecordVideo wrapper to be able to record videos.
    video_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos")
    sim_env = RecordVideo(sim_env, video_folder, lambda i: i == 0)

    # Reset the environment
    init_position, _ = sim_env.reset()
    # Take the first step with action=1 that starts the game.
    first_observation, _, _, _, _ = sim_env.step(action=1)
    # To create the state, we stack two observations, that is to have a sense of time/velocity
    #       for the pong game. Hint: use np.stack.
    state = np.stack([init_position, first_observation])

    for _ in range(2_500):
        # Call agent.act if the agent is specified, otherwise use action_space.sample() from
        #       the gym environment to select a random action.
        action = sim_env.action_space.sample()
        # Perform an environment step.
        observation, _, terminated, truncated, _ = sim_env.step(action)
        # Check for completion, if completed reset the environment.
        if terminated or truncated:
            observation, _ = sim_env.reset()
        # Update the state with the new stacked observations (the last and the new one)
        state = np.stack([state[1], observation])

    # TODO: Remember to close the gym environment!
    sim_env.close()

    # Call a convenient utility function to show the video in the notebook as a GIF.
    return show_gym_video_recording(folder=video_folder)