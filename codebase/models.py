import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# define a feed forward neural network
class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size = 3):
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


# define a feed forward neural network removing the last layer for the feature detector

class FeatureDetector(nn.Module):
    def __init__(self, input_size=200, hidden_size=32, num_layers=2):  #optimal parameters found earlier
        super(FeatureDetector, self).__init__()
        layers = []
        layer_size = input_size
        for i in range(num_layers):
            layers.append(nn.Linear(layer_size, hidden_size))
            layers.append(nn.ReLU())
            layer_size = hidden_size
        self.network = nn.Sequential(*layers)  #bundle the layers together

    def forward(self, x):
        return self.network(x)


#defining conditional RealNVP
class CondRealNVP(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, blocks, condition_size, a=1):
        super(CondRealNVP, self).__init__()
        self.input_size = input_size
        self.condition_size = condition_size
        self.blocks = nn.ModuleList([
            CouplingBlock(input_size, hidden_size, num_hidden_layers, condition_size, a) for _ in range(blocks)
        ])
        self.rotations = [torch.qr(torch.randn(input_size, input_size))[0] for _ in range(blocks - 1)]
        self.a = a  # Scaling factor for exponent in affine transformation

    def forward(self, x, condition):
        log_det = 0
        for i, block in enumerate(self.blocks):
            x, ld = block.forward(x, condition)
            log_det += ld
            if i < len(self.rotations):
                x = x @ self.rotations[i]
        return x, log_det

    def inverse(self, z, condition):
        for i, block in reversed(list(enumerate(self.blocks))):
            if i < len(self.rotations):
                z = z @ self.rotations[i].T
            z = block.inverse(z, condition)
        return z

    def sample(self, num_samples, conditions):
        # 'conditions' is a tensor of size (num_samples, condition_size)
        z = torch.randn(num_samples, self.input_size)
        return self.inverse(z, conditions)


class CouplingBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, condition_size, a=1):
        super(CouplingBlock, self).__init__()

        # Calculate split sizes
        self.split_size1 = input_size // 2  # Smaller part
        self.split_size2 = input_size - self.split_size1  # Larger part

        self.t_net = FeedForward(self.split_size1 + condition_size, hidden_size, num_hidden_layers, self.split_size2)  
        self.s_net = FeedForward(self.split_size1 + condition_size, hidden_size, num_hidden_layers, self.split_size2)
        self.a = a
    
    def forward(self, x, condition):
        x1 = x[:, :self.split_size1]  # Take the first part
        x2 = x[:, self.split_size1:]  # Take the rest

        input_conditioned = torch.cat([x1, condition], dim=1)        
        t = self.t_net(input_conditioned)
        s = self.s_net(input_conditioned)
        
        z1 = x1  # Identity transformation
        z2 = x2 * torch.exp(self.a * torch.tanh(s)) + t  # Affine transformation
        log_det = (self.a * torch.tanh(s)).sum(dim=1)
        
        return torch.cat([z1, z2], dim=1), log_det

    def inverse(self, z, condition):
        z1 = z[:, :self.split_size1]  # Take the first part
        z2 = z[:, self.split_size1:]  # Take the rest
    
        # Concatenate the condition to z1
        input_conditioned = torch.cat([z1, condition], dim=1)

        t = self.t_net(input_conditioned)
        s = self.s_net(input_conditioned)
        
        x1 = z1
        x2 = (z2 - t) * torch.exp(-self.a * torch.tanh(s))
        
        return torch.cat([x1, x2], dim=1)