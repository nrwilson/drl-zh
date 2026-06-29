import torch.functional as F
import torch.nn as nn
import torch


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        # Recall that the state is 2 frames of 84x84 pixels, which translates into a 2x84x84
        #       tensor. Let's run 2D convolutions to transform to 4x40x40 and then 16x9x9. To do
        #       that, add two Conv2d layers with (out_channels, kernel_size, stride) to match those
        #       dimensions. Hint: (size - kernel) / stride + 1
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=6, stride=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        
        # Flatten the output of the conv layers.  Flatten length = # out channels * width * height
        self.flattened_length = 16 * 9 * 9

        # Create two fully connected layers with 256 and action_size units.
        self.fc1 = nn.Linear(in_features=self.flattened_length, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=action_size)

    def forward(self, x):
        """Build a network that maps state -> action values."""
        # Forward through the convolutions (use ReLU non-linearity)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        # Remember to flatten the tensor for the linear layers!
        x = x.view(-1, self.flattened_length)
        # Forward through linear layers (using ReLU again)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        # The output is directly the output of the last linear layer, representing the
        #       action value function.
        return x
