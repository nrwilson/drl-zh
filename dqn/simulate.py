import random
import tempfile

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, RecordVideo

from util.gymnastics import DEVICE, init_random, plot_scores, show_gym_video_recording