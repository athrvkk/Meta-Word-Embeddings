# Class to implement the Concatenated Autoencoder
# File: DAEME.py
# Author: Atharva Kulkarni

#from tensorflow.keras.layers import Input, Dense, concatenate
#from tensorflow.keras import backend as K
#from tensorflow.keras import Model

import torch
import torch.nn as nn

class CAE(nn.Module):
    """ Class to implement the Concatenated Autoencoder """
    
    def __init__(self, input_dim, latent_dim, activation, lambda1, lambda2, lambda3):
        """
        @param latent_dim (int): latent_dimension for each autoencoder. Default: 300.
        @ activation (string): type of activation: leaky_relu, paramaterized_leaky_relu, relu, tanh, and sigmoid. Default: leaky_relu.
        @param lambda1 (int): Multiplicaiton factor for computing loss for part1. Default: 1.
        @param lambda2 (int): Multiplicaiton factor for computing loss for part2. Default: 1.
        @param lambda3 (int): Multiplicaiton factor for computing loss for part3. Default: 1.
        """
        super().__init__()
        self.activation = activation
        self.encoder1 = nn.Linear(in_features=input_dim, in_features=latent_dim)
        self.encoder2 = nn.Linear(in_features=input_dim, in_features=latent_dim)
        self.encoder3 = nn.Linear(in_features=input_dim, in_features=latent_dim)
        self.decoder1 = nn.Linear(in_features=latent_dim, in_features=input_dim)
        self.decoder2 = nn.Linear(in_features=latent_dim, in_features=input_dim)
        self.decoder3 = nn.Linear(in_features=latent_dim, in_features=input_dim)
        
        
        
        
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
        
        bottleneck = torch.cat([x1, x2, x3], dim=0)
        
        x1 = self.decoder1(bottleneck)
        x1 = self.activation(x1)
        
        x2 = self.decoder2(bottleneck)
        x2 = self.activation(x2)
        
        x3 = self.decoder3(bottleneck)
        x3 = self.activation(x3)
        
       return x1, x2, x3
        
        
        
        
        
     def mse(self, y_true, y_pred, factor):   
        """ Function to compute weighted Mean Squared Error (MSE)
        @param y_true (array): input vector.
        @param y_pred (array): output vector.
        @param factor (float): multiplicative factor.
        @return mse_loss (float): the mean squared error loss.        
        """
        return factor*K.mean(K.square(y_true - y_pred))
        
        
    def loss(self, y_true, y_pred):
        """ Function to compute loss for Concatenated Autoencoder.
        @param y_true (np.array): input vector.
        @param y_pred (np.array): output vector.
        @return loss (float): the computed loss
        """        
        return (self.mse(y_true[0], y_pred[0], self.lambda1) + 
                self.mse(y_true[1], y_pred[1], self.lambda2) + 
                self.mse(y_true[2], y_pred[2], self.lambda3))
        
      
