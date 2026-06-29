from dqn.replay_buffer import ReplayBuffer
from dqn.q_network import QNetwork
import torch
from torch.nn import functional as F
from torch import optim
from util.gymnastics import DEVICE
import numpy as np
import random


class Agent:
    """Agent that interacts with and learns from the environment."""

    def __init__(
        self,
        action_size=6,
        gamma: float = 0.99,
        tau: float = 1e-3,
        lr: float = 1e-4,
        batch_size: int = 32,
        learn_every: int = 4,
        update_target_every: int = 2,
        preload_file: str = None,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.batch_size = batch_size
        self.learn_every = learn_every
        self.update_target_every = update_target_every
        self.t_learn_step = 0
        self.t_update_target_step = 0

        # Create the replay buffer.
        self.memory = ReplayBuffer()

        # Create both the local and target networks.
        self.qnetwork_local = QNetwork(action_size=self.action_size).to(DEVICE)
        self.qnetwork_target = QNetwork(action_size=self.action_size).to(DEVICE)
        # Copy the weights of the local network in the target one. Hint use state_dict() and
        #       load_state_dict methods.
        parameters = self.qnetwork_local.state_dict()
        self.qnetwork_target.load_state_dict(parameters)

        # Remember to set the target network only in eval mode.
        self.qnetwork_target.eval()

        # Creating the optimizer.
        self.optimizer = optim.RMSprop(self.qnetwork_local.parameters(), lr=self.lr)

        # If we want to preload a saved network, we do it here.
        if preload_file is not None:
            print(f"Loading pre-trained model: {preload_file}")
            self.qnetwork_local.load_state_dict(torch.load(preload_file, map_location=DEVICE))
            # Now that we loaded the local weights, re-sync the target network to them so
            #       that subsequent learning starts from a coherent state.
            parameters = self.qnetwork_local.state_dict()
            self.qnetwork_target.load_state_dict(parameters)

    def step(self, state, action, reward, next_state, done):
        """Tells the agent to make a step: record experience and possibly learn."""
        # Save experience in replay memory
        self.memory.add(state, [action], reward, next_state, done)
        # Update t_learn_step to determine when to learn (every "learn_every" time steps).
        self.t_learn_step = (self.t_learn_step + 1) % self.learn_every
        if self.t_learn_step == 0:
            # If enough samples are available in memory, get random subset and learn.
            if len(self.memory) > self.batch_size:
                self.learn()
        # Update target network every "update_target_every" step.
        self.t_update_target_step = (self.t_update_target_step + 1) % self.update_target_every
        if self.t_update_target_step == 0:
            self.soft_update_model_params(self.qnetwork_local, self.qnetwork_target, self.tau)

    def act(self, state: np.ndarray, eps=0.0):
        """Makes the agent take an action for the state passed as input."""
        # Convert the state to a torch.Tensor
        state = torch.from_numpy(state).float().to(DEVICE)
        # Set the local network to eval, probe it to get the action values, and set it back
        #       to training mode. Remember to use torch.no_grad().
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        # Perform epsilon-greedy action selection based on eps.
        #       Hint: either np.argmax or random.choice across all actions.
        # If we do not cross decaying eps and are still in exploration mode, pick random:
        if random.random() < eps:
            action = random.choice(range(self.action_size))
        # If we cross the decaying eps and can take an action, pick the best one.
        else:
            action = np.argmax(action_values)
        return action

    def learn(self):
        """Executes one learning step for the agent."""
        # Select a batch of experiences from the replay buffer
        experiences = self.memory.sample(self.batch_size)
        # Unpack them
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            # Get the predicted action values of the *NEXT* states from the target model.
            target_action_values = self.qnetwork_target(next_states).detach()
            # Select the max action value for each state:
            #       Hint: https://pytorch.org/docs/stable/generated/torch.amax.html
            max_action_values = torch.amax(target_action_values, 1)[0]  # (batch_size, 1)
            max_action_values = torch.amax(target_action_values, 1, keepdim=True)
            # Then, compute the Q _targets_ for the current states.
            Q_targets = rewards + (self.gamma * max_action_values * (1 - dones))  # (batch_size, 1)

        # Get the predicted Q values from local model...
        predicted_action_values = self.qnetwork_local(states)
        # ...but choose only the action value that was selected in the experience replay.
        model_predictions_for_actions = predicted_action_values.gather(1, actions)
        # Compute loss. Hint: use the Huber Loss.
        loss = F.huber_loss(model_predictions_for_actions, Q_targets)

        # Minimize the loss. Hint: zero_grad the optim, backward on loss, step the optim.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @staticmethod
    def soft_update_model_params(src: torch.nn.Module, dest: torch.nn.Module, tau=1e-3):
        """Soft updates model parameters (θ_dest = τ * θ_src + (1 - τ) * θ_dest)."""
        # For each dest parameter (get them via the parameters() function), update it with
        #       the update-rule in the method description. Hint: use data.copy_ of the parameter.
        for src_parameter, dest_parameter in zip(src.parameters(), dest.parameters()):
            new_value = tau * src_parameter + (1-tau) * dest_parameter
            dest_parameter.data.copy_(new_value)

    def checkpoint(self):
        """Save the QNetwork weights in a file."""
        torch.save(self.qnetwork_local.state_dict(), "dqn_weights.pth")