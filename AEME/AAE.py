# Class to implement the Averaged Autoencoder
# File: DAEME.py
# Author: Atharva Kulkarni

from tensorflow.keras.layers import Input, Dense, Average
import numpy as np
from tensorflow.keras import Model


class AAE():
    """ Class to implement the Averaged Autoencoder """
    
    def __init__(self, latent_dim, activation, lambda1, lambda2, lambda3):
        """
        @param latent_dim (int): latent_dimension for each autoencoder. Default: 300.
        @ activation (string): type of activation: leaky_relu, paramaterized_leaky_relu, relu, tanh, and sigmoid. Default: leaky_relu.
        @param lambda1 (int): Multiplicaiton factor for computing loss for part1. Default: 1.
        @param lambda2 (int): Multiplicaiton factor for computing loss for part2. Default: 1.
        @param lambda3 (int): Multiplicaiton factor for computing loss for part3. Default: 1.
        """
        self.latent_dim = latent_dim
        self.activation = activation
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        
        
        
    def build(self, input_dim):
        """Function to build the Averaged autoencoder.
        @param input_dim (shape): shape of the input dimensions.
        """
        input1 = Input(shape=(input_dim,))
        Dense1 = Dense(self.latent_dim, activation=self.activation)(input1)
        
        input2 = Input(shape=(input_dim,))
        Dense2 = Dense(self.latent_dim, activation=self.activation)(input2)
        
        input3 = Input(shape=(input_dim,))
        Dense3 = Dense(self.latent_dim, activation=self.activation)(input3)
        
        bottleneck = Average([Dense1, Dense2, Dense3])
        
        output1 = Dense(input_dim, activation=self.activation)(bottleneck)
        output2 = Dense(input_dim, activation=self.activation)(bottleneck)
        output3 = Dense(input_dim, activation=self.activation)(bottleneck)
        
        model = Model(inputs=[input1, input2, input3], outputs=[output1, output2, output3])
        encoder = Model(inputs=[input1, input2, input3], outputs=bottleneck)
        model.compile(optimizer="adam", loss=self.AAE_loss)
        model.summary()
        
        return model, encoder
        
        
        
        
     def mse(self, y_true, y_pred, factor):   
        """ Function to compute weighted Mean Squared Error (MSE)
        @param y_true (array): input vector.
        @param y_pred (array): output vector.
        @param factor (float): multiplicative factor.
        @return mse_loss (float): the mean squared error loss.        
        """
        return factor*K.mean(K.square(y_true - y_pred))
        
        
    def AAE_loss(self, y_true, y_pred):
        """ Function to compute loss for Averaged Autoencoder.
        @param y_true (np.array): input vector.
        @param y_pred (np.array): output vector.
        @return loss (float): the computed loss
        """        
        return (self.mse(y_true[0], y_pred[0], self.lambda1) + 
                self.mse(y_true[1], y_pred[1], self.lambda2) + 
                self.mse(y_true[2], y_pred[2], self.lambda3))
        
        
        
        
