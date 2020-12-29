# Class to generate Autoencoder Meta Embeddings (AEME)
# File: AEME.py
# Author: Atharva Kulkarni

from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU, PReLU, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from DAE import DAE
from CAE import CAE
from AAE import AAE


class AEME():
    """ Class to implement Autoencoder for generating Meta-Embeddings"""
    
    def __init__(self, model_checkpoint_path, mode="DAEME", latent_dim=100, activation="leaky_relu", lambda1=1, lambda2=1, lambda3=1, lambda4=1, lambda5=1, lambda6=1, lr_reduce_factor=0.2, patience=5):
        """ Constructor to initialize autoencoder parameters
        @param model_checkpoint_path (string): path to store ModelCheckpoints.
        @param mode (string): type of Autoencoder to build: Decoupled Autoencoded Meta-Embedding (DAEME), Concatenated Autoencoded Meta-Embedding (CAEME), Averaged Autoencoded Meta-Embedding (AAEME).
        @param latent_dim (int): latent_dimension for each autoencoder. Default: 300.
        @ activation (string): type of activation: leaky_relu, paramaterized_leaky_relu, relu, tanh, and sigmoid. Default: leaky_relu.
        @param lambda1 (int): Multiplicaiton factor for computing loss for part1. Default: 1.
        @param lambda2 (int): Multiplicaiton factor for computing loss for part2. Default: 1.
        @param lambda3 (int): Multiplicaiton factor for computing loss for part3. Default: 1.
        @param lambda4 (int): Multiplicaiton factor for computing loss for part4 (Only for DAE). Default: 1.
        @param lambda5 (int): Multiplicaiton factor for computing loss for part5 ((Only for DAE). Default: 1.
        @param lambda6 (int): Multiplicaiton factor for computing loss for part6 ((Only for DAE). Default: 1.
        @param lr_reduce_factor (float): factor by which the learning rate will be reduced. new_lr = lr * factor. Default value: 0.2.
        @param patience (int): number of epochs with no improvement after which learning rate will be reduced. Default: 5.
        """
        self.mode = mode
        self.model = None
        self.encoder = None
        
        if activation == "leaky_relu":
            activation = LeakyReLU()
        elif activation == "paramaterized_leaky_relu":
            activation = PReLU()           
        else:
            activation = Activation(activation)
              
        if mode == "DAEME":
            self.ae = DAE(latent_dim, activation, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6)
        elif mode == "CAEME":
            self.ae = CAE(latent_dim, activation, lambda1, lambda2, lambda3)
        elif mode == "AAEME":
            self.ae = AAE(latent_dim, activation, lambda1, lambda2, lambda3)
              
        # Save best model callback
        self.model_checkpoint_callback = ModelCheckpoint(filepath=model_checkpoint_path,
                                                         save_weights_only=True,
                                                         monitor='loss',
                                                         mode='auto',
                                                         save_freq = 'epoch',
                                                         save_best_only=True)
        # Reduce learning rate callback
        self.reduce_lr_callback = ReduceLROnPlateau(monitor='loss', 
                                                    mode='auto',
                                                    factor=lr_reduce_factor, 
                                                    patience=patience, 
                                                    min_lr=0.0005, 
                                                    verbose=1)
                                                    
            


            
    def build(self, input_dim):
        """ Function to build the autoencoder.
        @param input_dim (shape): shape of the input dimensions.
        """
        self.model, self.encoder = self.ae.build(input_dim)
        
            


    
    def add_noise(self, data, masking_noise_factor):   
        """Function to add mask noise to data.
        @param data (np.array): data to add noise to.
        @param masking_noise_factor (float): Percentage of noise to add to the data.
        @return data (np.array): noise added data.
        """
        data_size, feature_size = data.shape
        for i in range(data_size):
            mask_noise = np.random.randint(0, feature_size, int(feature_size * masking_noise_factor))
            for m in mask_noise:
                data[i][m] = 0
        return data
        


        
    def train(self, x_train1, x_train2, x_train3, epochs=200, batch_size=32, masking_noise=True, masking_noise_factor=0.05):
        """ Function to train the Autoencoder Model.
        @param x_train (np.array): The input data.
        @@param epochs (int): Number of epochs for which the model is to be trained. Default: 10.
        @param batch_size (int): Number of batches to divide the training data during training.
        @param masking_noise (float): Percentage noise to be induced in the input data. Default: 0.05 or 5%.
        @ return histroy (History object): history of the model.
        """
        if masking_noise:
            noisy_x_train1 = self.add_noise(x_train1, masking_noise_factor)
            noisy_x_train2 = self.add_noise(x_train2, masking_noise_factor)
            noisy_x_train3 = self.add_noise(x_train3, masking_noise_factor)
            
            history = self.ae.fit([noisy_x_train1, noisy_x_train2, noisy_x_train3],
                                  [x_train1, x_train2, x_train3], 
                                  epochs=epochs, 
                                  batch_size=batch_size,
                                  shuffle=True,
                                  verbose=1,
                                  callbacks=[self.model_checkpoint_callback, self.reduce_lr_callback])
        
        else:
            history = self.ae.fit([x_train1, x_train2, x_train3],
                                  [x_train1, x_train2, x_train3], 
                                  epochs=epochs, 
                                  batch_size=batch_size,
                                  shuffle=True,
                                  verbose=1,
                                  callbacks=[self.model_checkpoint_callback, self.reduce_lr_callback])
        return history
        
        
        
   
    def predict(self, x_test1, x_test2, x_test3, model_checkpoint):
        """ Function to generate predictions of the autoencoder's encoder.
        @param x_test1 (np.array): test input 1.
        @param x_test2 (np.array): test input 2.
        @param x_test3 (np.array): test input 3.
        @param model_checkpoint (string): model weights.
        @return predictions (np.array): Autoencoder's encoder's predictions.
        """
        self.model.load_weights(model_checkpoint)     
        return self.encoder.predict([x_test1, x_test2, x_test3])
    
            
            
            
            
            
