# Class to implement the Averaged Autoencoder
# File: AAE.py
# Author: Atharva Kulkarni

import torch
import torch.nn as nn
import torch.nn.functional as functional


class AAE(nn.Module):
    """ Class to implement the Averaged Autoencoder """
    
    def __init__(self, input_dim, latent_dim, activation, lambda1, lambda2, lambda3):
        """ Constructor
        @param input_dim (int): Input dimension for the autoencoders .Default: 300.
        @param latent_dim (int): latent_dimension for each autoencoders. Default: 300.
        @ activation (string): type of activation: leaky_relu, paramaterized_leaky_relu, relu, tanh, and sigmoid. Default: leaky_relu.
        @param lambda1 (int): Multiplicaiton factor for computing loss for part1. Default: 1.
        @param lambda2 (int): Multiplicaiton factor for computing loss for part2. Default: 1.
        @param lambda3 (int): Multiplicaiton factor for computing loss for part3. Default: 1.
        """
        super(AAE, self).__init__()
        self.activation = activation
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.encoder1 = nn.Linear(in_features=input_dim, out_features=latent_dim)
        nn.init.normal_(self.encoder1.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.encoder1.bias)
        
        self.encoder2 = nn.Linear(in_features=input_dim, out_features=latent_dim)
        nn.init.normal_(self.encoder2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.encoder2.bias)
        
        self.encoder3 = nn.Linear(in_features=input_dim, out_features=latent_dim)
        nn.init.normal_(self.encoder3.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.encoder3.bias)
        
        self.decoder1 = nn.Linear(in_features=latent_dim, out_features=input_dim)
        nn.init.normal_(self.decoder1.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.decoder1.bias)
        
        self.decoder2 = nn.Linear(in_features=latent_dim, out_features=input_dim)
        nn.init.normal_(self.decoder2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.decoder2.bias)
        
        self.decoder3 = nn.Linear(in_features=latent_dim, out_features=input_dim)
        nn.init.normal_(self.decoder3.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.decoder3.bias)
        
        
        
        
    def forward(self, x1, x2, x3):
        """Function to build the Concatenated autoencoder.
        @param input_dim (shape): shape of the input dimensions.
        """
        x1 = self.encoder1(x1)
        x1 = self.activation(x1)
        
        x2 = self.encoder2(x2)
        x2 = self.activation(x2)
        
        x3 = self.encoder3(x3)
        x3 = self.activation(x3)
        
        bottleneck = functional.normalize(torch.div(x1+x2+x3, torch.norm(x1+x2+x3)), dim=1)
        
        x1 = self.decoder1(bottleneck)
        x1 = self.activation(x1)
        
        x2 = self.decoder2(bottleneck)  
        x2 = self.activation(x2)
        
        x3 = self.decoder3(bottleneck)
        x3 = self.activation(x3)
        
        return [x1, x2, x3], bottleneck
        
        
        
         
    def mse(self, output, target, factor):   
        """ Function to compute weighted Mean Squared Error (MSE)
        @param target (array): input vector.
        @param output (array): output vector.
        @param factor (float): multiplicative factor.
        @return mse_loss (float): the mean squared error loss.        
        """
        return factor*torch.mean((output - target)**2)
        
        
        
        
    def loss(self, output, target):
        """ Function to compute loss for Averaged Autoencoder.
        @param target (np.array): input vector.
        @param output (np.array): output vector.
        @return loss (float): the computed loss
        """        
        return (self.mse(target[0], output[0], self.lambda1) + 
                self.mse(target[1], output[1], self.lambda2) + 
                self.mse(target[2], output[2], self.lambda3))
        
      

