# File containing all the necessary functions
# File: utils.py
# Author: Atharva Kulkarni

import numpy as np
import pandas as pd
import re
import torch
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import KeyedVectors



class Utils():
    """" Class for I/O functionalities """
    
    
    def read_data(self, path, columns):
            """Function to read input data
            @param path(string): path to the data
            @return data(pd.DataFrame): A dataframe containing the necessary data
            """
            data = pd.read_csv(path, usecols=columns, encoding="utf-8")
            return data





    def clean_data(self, corpus):
            """Function to clean data (remove punctuations. extra spaces, special characters, etc.)
            @param corpus(list): Dataset to be cleaned.
            @return corpus(list): cleaned dataset.
            """
            corpus = [str(text).split() for text in corpus]
            corpus = [" ".join([re.sub(r'\W', '', word.lower()) for word in text]) for text in corpus]
            return corpus




    
    def get_vocab(self, path):
        """ Function to get vocabulary from embedding file.
        @param path (string): path to the embedding file.
        @return vocab_list (list): vocabulary list.
        """
        if path.split(".")[-1] == "bin":
            return [word for word in KeyedVectors.load_word2vec_format(path, binary=True).vocab.keys()]
        else:
            return [word for word in KeyedVectors.load_word2vec_format(path, binary=False).vocab.keys()]
        
        
        

         
    def get_embedding_dict(self, path, vocab):
            """ Function to retreive embeddings from a file into a dictionary.
            @param path (str): path to the file where embeddings are to be written to.
            @param vocab (list): vocabulary list.
            @return embedding_dict (dict): Dictionary of words/documents as keys and their respective embeddings as values.
            """
            if path.split(".")[-1] == "bin":
                model = KeyedVectors.load_word2vec_format(path, binary=True)
            else:
                model = KeyedVectors.load_word2vec_format(path, binary=False)

            embedding_dict = dict()
            for word in vocab:
                embedding_dict[word] = model.wv[word.lower()]
            return embedding_dict

      
            

                             
    def get_embedding_matrix(self, path, vocab=None):
            """ Function to retreive embeddings from a file into a matrix.
            @param path (str): path to the file where embeddings are to be written to.
            @param vocab (list): vocabulary list.
            @return embedding_matrix (np.array): an array containing the embeddings of all the documents/vocabulary.
            """
            if path.split(".")[-1] == "bin":
                model = KeyedVectors.load_word2vec_format(path, binary=True)
            else:
                model = KeyedVectors.load_word2vec_format(path, binary=False)
            
            embedding_matrix = np.zeros((len(vocab)+1, 300))
            for index in range(len(vocab)):
                vec = model.wv[vocab[index].lower()]
                if vec is not None:
                    embedding_matrix[index] = vec
            return embedding_matrix




    
    def map_embeddings(self, vocab, embedding_dict1, embedding_dict2, embedding_dict3):
        """ Funciton to map embeddings.
        @param vocab (list): list of the vocabulary to map embeddings for.
        @param embedding_dict1 (dict): Dictionary for embedding 1.
        @param embedding_dict2 (dict): Dictionary for embedding 2.
        @param embedding_dict3 (dict): Dictionary for embedding 3.
        @ return embedding1 (tensor), embedding2 (tensor), embedding3 (tensor): arrays of mapped embeddings.
        """
        embedding1 = []
        embedding2 = []
        embedding3 = []
        for word in vocab:
            embedding1.append(embedding_dict1.get(word))
            embedding2.append(embedding_dict2.get(word))
            embedding3.append(embedding_dict3.get(word))
        return torch.tensor(embedding1), torch.tensor(embedding2), torch.tensor(embedding3)
        
        
        

    
    def read_vocab(self, file_path):
        """ Function to read vocab
        @param file_path (str): file path to read the vocabulary.
        @return vocab (list): return a vocabulary list.
        """
        vocab = []
        with open(file_path, "r") as f:
            data = f.readlines()
        f.close()
        for row in data:
            row = row.split("\n")
            vocab.append(row[0])
        return vocab
        
        
        

        
    def write_vocab(self, file_path, vocab):
        """ Function to write vocabulary list to a file.
        @param file_path (str): file path to write the vocabulary.
        @param vocab (list): vocabulary list.
        """
        with open(file_path, 'w') as f:
            for word in vocab:
                f.write("%s\n" % word)
        f.close()





    def write_embeddings(self, path, vocab, embeddings):
            """ Fucntion to write embeddings to a file
            @param path (str): path to the file where embeddings are to be written to.
            @param vocab (list): vocabulary or id of the total words/ documents of whom we have the embeddings.
            @param embeddings (np.arrau): embedding array of shape (document/vocab size, embedding dimensions).
            """
            file = open(path, "w")
            i = 0
            for word in vocab:
                wv_string = ""
                for vi in embeddings[i].tolist():
                    wv_string = wv_string + " " + str(vi)
                wv_string = str(word) + " " + str(wv_string) + "\n"
                i = i + 1
                file.write(wv_string)
            file.close()


