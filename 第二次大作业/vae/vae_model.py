import jittor as jt
from jittor import nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim):
        super(Encoder, self).__init__()
        
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(1, len(hidden_dims)):
            layers.extend([nn.Linear(hidden_dims[i-1], hidden_dims[i]), nn.ReLU()])
        self.layers = nn.Sequential(*layers)

        self.mu = nn.Linear(hidden_dims[-1], z_dim)
        self.log_var = nn.Linear(hidden_dims[-1], z_dim)

        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                n_in = layer.in_features
                n_out = layer.out_features
                stdv = np.sqrt(2.0 / (n_in + n_out))
                nn.init.gauss_(layer.weight, mean=0, std=stdv)
                nn.init.constant_(layer.bias, 0)

    def execute(self, x):
        hidden = self.layers(x)
        return self.mu(hidden), self.log_var(hidden)


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()

        layers = [nn.Linear(z_dim, hidden_dims[-1]), nn.ReLU()]
        for i in range(len(hidden_dims)-1, 0, -1):
            layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i-1]), nn.ReLU()])
        self.layers = nn.Sequential(*layers)

        self.out = nn.Linear(hidden_dims[0], output_dim)

        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                n_in = layer.in_features
                n_out = layer.out_features
                stdv = np.sqrt(2.0 / (n_in + n_out))
                nn.init.gauss_(layer.weight, mean=0, std=stdv)
                nn.init.constant_(layer.bias, 0)

    def execute(self, z):
        hidden = self.layers(z)
        return jt.sigmoid(self.out(hidden))

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims, z_dim)
        self.decoder = Decoder(z_dim, hidden_dims, input_dim)

    def execute(self, x):
        mu, log_var = self.encoder(x)
        std = jt.exp(0.5 * log_var)
        eps = jt.randn(std.shape)
        z = mu + eps * std
        return self.decoder(z), mu, log_var

def vae_loss(recon_x, x, mu, log_var,beta=1):
    recon_loss = nn.mse_loss(recon_x, x,reduction='sum')
    kl_loss = -0.5 * jt.sum(1 + log_var - mu*mu - jt.exp(log_var))
    return recon_loss + beta * kl_loss