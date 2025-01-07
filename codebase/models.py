import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# define a feed forward neural network
class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(FeedForward, self).__init__()
        layers = []
        layer_size = input_size
        for i in range(num_layers):
            layers.append(nn.Linear(layer_size, hidden_size))
            layers.append(nn.ReLU())
            layer_size = hidden_size
        layers.append(nn.Linear(layer_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)