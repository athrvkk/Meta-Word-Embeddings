# Class to generate Autoencoder Meta Embeddings (AEME)
# File: AEME.py
# Author: Atharva Kulkarni

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
import numpy as np
import gc
from DAE import DAE
from CAE import CAE
from AAE import AAE


class AEME():
    """ Class to implement Autoencoder for generating Meta-Embeddings """
    
    def __init__(self, mode="CAE", input_dim=300, latent_dim=100, activation="leaky_relu", lambda1=1, lambda2=1, lambda3=1, lambda4=1, lambda5=1, lambda6=1):
        """ Constructor to initialize autoencoder parameters
        @param mode (string): type of Autoencoder to build: Decoupled Autoencoder (DAE), Concatenated Autoencoder (CAE), Averaged Autoencoder (AAE).
        @param latent_dim (int): latent_dimension for each autoencoder. Default: 300.
        @ activation (string): type of activation: leaky_relu, paramaterized_leaky_relu, relu, tanh, and sigmoid. Default: leaky_relu.
        @param lambda1 (int): Multiplicaiton factor for computing loss for part1. Default: 1.
        @param lambda2 (int): Multiplicaiton factor for computing loss for part2. Default: 1.
        @param lambda3 (int): Multiplicaiton factor for computing loss for part3. Default: 1.
        @param lambda4 (int): Multiplicaiton factor for computing loss for part4 (Only for DAE). Default: 1.
        @param lambda5 (int): Multiplicaiton factor for computing loss for part5 ((Only for DAE). Default: 1.
        @param lambda6 (int): Multiplicaiton factor for computing loss for part6 ((Only for DAE). Default: 1.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("GPU : ", torch.cuda.get_device_name(0))
        else:
          self.device = torch.device("cpu")
          print("CPU on")
          
        self.mode = mode
        self.encoder = None
        
        if activation == "leaky_relu":
            activation = nn.LeakyReLU()
        elif activation == "paramaterized_leaky_relu":
            activation = nn.PReLU()           
        else:
            activation = nn.ReLU()
              
        if mode == "DAE":
            self.ae = DAE(input_dim, latent_dim, activation, lambda1, lambda2, lambda3, lambda4, lambda5, lambda6)
        elif mode == "CAE":
            self.ae = CAE(input_dim, latent_dim, activation, lambda1, lambda2, lambda3)
        elif mode == "AAE":
            self.ae = AAE(input_dim, latent_dim, activation, lambda1, lambda2, lambda3)
                                                            
            
            

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
        


    
    def prepare_input(self, x_train1, x_train2, x_train3, batch_size=128, masking_noise=True, masking_noise_factor=0.05): 
        """ Funciton to compile data in Dataloaders
        @param x_train1 (np.array): The input data1.
        @param x_train2 (np.array): The input data2.
        @param x_train3 (np.array): The input data3.
        @param batch_size (int): Number of batches to divide the training data into.
        @param masking_noise (bool): To add Masking Noise or not.
        @param masking_noise_factor (float): Percentage noise to be induced in the input data. Default: 0.05 or 5%.
        """
        if masking_noise:
            x_train1 = self.add_noise(x_train1, masking_noise_factor)
            x_train2 = self.add_noise(x_train2, masking_noise_factor)
            x_train3 = self.add_noise(x_train3, masking_noise_factor)
            
        tensor_dataset = torch.utils.data.TensorDataset(x_train1, 
                                                        x_train2, 
                                                        x_train3)
        del x_train1
        del x_train2
        del x_train3
        gc.collect()
        torch.cuda.empty_cache()
        return torch.utils.data.DataLoader(dataset=tensor_dataset, 
                                           sampler=torch.utils.data.RandomSampler(tensor_dataset),
                                           batch_size=batch_size)
                   
        
        
        
    def train(self, data_loader, epochs=200, checkpoint_path=""):
        """ Function to train the Autoencoder Model.    
        @param tensor_data_loader (torch.tensor): Batch-wise dataset.
        @@param epochs (int): Number of epochs for which the model is to be trained. Default: 10.
        """
        self.ae.train()
        self.ae.to(self.device)
     
        optimizer = torch.optim.Adam(self.ae.parameters())    
        
        training_loss = []
        
        if self.mode == "DAE": 
            for step in range(1, epochs+1):
                epoch_loss = 0.0
                for batch_data in data_loader:
                    x_train1, x_train2, x_train3 = tuple(t.to(self.device) for t in batch_data)
                    optimizer.zero_grad()
                    output, bottleneck = self.ae(x_train1, x_train2, x_train3)
                    loss = self.ae.loss([output, bottleneck], [x_train1, x_train2, x_train3])
                    loss.backward()
                    epoch_loss = epoch_loss + loss.item() 
                    optimizer.step()
                epoch_loss = epoch_loss/len(data_loader)
                training_loss.append(epoch_loss)
                print("\nEpoch: {} of {} ----> loss: {:.3f}".format(step, epochs, epoch_loss))
                
                if len(training_loss) > 2:
                  if epoch_loss < training_loss[-2]:
                      model_checkpoint = checkpoint_path + "_epoch_" + str(step) + "_loss_" + str(epoch_loss) + ".pt"
                      torch.save({"epoch": step,
                                  "model_state_dict": self.ae.state_dict(),
                                  "optimizer_state_dict": optimizer.state_dict(),
                                  "loss": loss}, 
                                  model_checkpoint)
                                
        else:
            for step in range(1, epochs+1):
                epoch_loss = 0.0
                for batch_data in data_loader:
                    x_train1, x_train2, x_train3 = tuple(t.to(self.device) for t in batch_data)
                    optimizer.zero_grad()
                    output, bottleneck = self.ae(x_train1, x_train2, x_train3)
                    loss = self.ae.loss(output, [x_train1, x_train2, x_train3])
                    loss.backward()
                    epoch_loss = epoch_loss + loss.item() 
                    optimizer.step()
                epoch_loss = epoch_loss/len(data_loader)
                training_loss.append(epoch_loss)
                print("\nEpoch: {} of {} ----> loss: {:.3f}".format(step, epochs, epoch_loss))
                
                if len(training_loss) > 2:
                  if epoch_loss < training_loss[-2]:
                      model_checkpoint = checkpoint_path + "_epoch_" + str(step) + "_loss_" + str(epoch_loss) + ".pt"
                      torch.save({"epoch": step,
                                  "model_state_dict": self.ae.state_dict(),
                                  "optimizer_state_dict": optimizer.state_dict(),
                                  "loss": loss}, 
                                  model_checkpoint)
  
        
   
   
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

