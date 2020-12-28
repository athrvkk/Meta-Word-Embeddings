# File containing all the necessary functions
# File: utils.py
# Author: Atharva Kulkarni

import numpy as np
import pandas as pd
import re
from gensim.models.coherencemodel import CoherenceModel
from tensorflow.nn import l2_normalize

def read_data(path, columns=["id", "author", "title", "selftext", "is_url"]):
        """Function to read input data
        @param path(string): path to the data
        @return data(pd.DataFrame): A dataframe containing the necessary data
        """
        data = pd.read_csv(path, usecols=columns, encoding="utf-8")
        return data



def clean_data(corpus):
        """Function to clean data (remove punctuations. extra spaces, special characters, etc.)
        @param corpus(list): Dataset to be cleaned.
        @return corpus(list): cleaned dataset.
        """
        corpus = [str(text).split() for text in corpus]
        corpus = [" ".join([re.sub(r'\W', '', word.lower()) for word in text]) for text in corpus]
        return corpus


def write_vocab(file_path, vocab):
    with open(file_path, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)
    f.close()



def write_to_file(path, vocab, embeddings):
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



def get_embedding_dict(path):
        """ Function to retreive embeddings from a file into a dictionary.
        @param path (str): path to the file where embeddings are to be written to.
        @return embedding_dict (dict): Dictionary of words/documents as keys and their respective embeddings as values.
        """
        embedding_dict = dict()
        with open(path, 'r') as f:
            data = f.readlines()
        f.close()
        for row in data:
            row = row.split()
            embedding_dict[row[0]] = np.array(row[1:], dtype='float32')
        if "nan" in embedding_dict:
            del(embedding_dict['nan'])
        return embedding_dict
        
        
        
def get_embedding_matrix(path):
        """ Function to retreive embeddings from a file into a matrix.
        @param path (str): path to the file where embeddings are to be written to.
        @return embedding_matrix (np.array): an array containing the embeddings of all the documents/vocabulary.
        """
        embeddings = []
        with open(path, 'r') as f:
            data = f.readlines()
        f.close()
        for row in data:
            row = row.split()
            if str(row[0]) == "nan":
                continue
            vec = [float(x) for x in row[1:]]
            embeddings.append(vec)
        embedding_matrix = np.array(embeddings, dtype="float32")
        return embedding_matrix




def map_embeddings(vocab, lda_embedding_dict, bert_selftext_embedding_dict, bert_title_embedding_dict, title=True, concat=True, normalize=False, gamma=1):
        """ Function to map LDA and BERT embeddings to one another.
        @param bert_embedding_dict (dictionary): Dictionary containing post ids as keys and BERT embeddings as values.
        @param LDA_embedding_dict (dictionary): Dictionary containing post ids as keys and LDA embeddings as values.
        @param concat (boolean): True if you want to concat LDA and BERT Embeddings.
        @param gamma (int): value to be multiply the LDA embeddings. Indicated the importance given to LDA embeddings.
        @return x_train (numpy.array): array of both the embeddings combined if concat=True.
        @return LDA_inputs (numpy.array), BERT_inputs (numpy.array): Array of LDA and BERT embeddings in correct sequence.
        """
        LDA_inputs = []
        BERT_selftext_inputs = []
        
        if title:
            BERT_title_inputs = []
            for key in vocab:
              LDA_inputs.append(lda_embedding_dict.get(key))
              BERT_selftext_inputs.append(bert_selftext_embedding_dict.get(key))
              BERT_title_inputs.append(bert_title_embedding_dict.get(key))
              
            if normalize:
                LDA_inputs = l2_normalize(np.array(LDA_inputs, dtype="float32"), 1)
                BERT_selftext_inputs = l2_normalize(np.array(BERT_selftext_inputs, dtype="float32"), 1)
                BERT_title_inputs = l2_normalize(np.array(BERT_title_inputs, dtype="float32"), 1)
                
            else:
                LDA_inputs = np.array(LDA_inputs, dtype="float32")
                BERT_selftext_inputs = np.array(BERT_selftext_inputs, dtype="float32")
                BERT_title_inputs = np.array(BERT_title_inputs, dtype="float32")               
                
            print("LDA Embeddings Shape: ", LDA_inputs.shape)    
            print("BERT Selftext Embeddings Shape: ", BERT_selftext_inputs.shape)
            print("BERT Title Embeddings Shape: ", BERT_title_inputs.shape)            
            
            
        else:
            for key in vocab:
                LDA_inputs.append(lda_embedding_dict.get(key))
                BERT_selftext_inputs.append(bert_selftext_embedding_dict.get(key))
            
            if normalize:
                LDA_inputs = l2_normalize(np.array(LDA_inputs, dtype="float32"), 1)
                BERT_selftext_inputs = l2_normalize(np.array(BERT_selftext_inputs, dtype="float32"), 1)
                                
            else:
                LDA_inputs = np.array(LDA_inputs, dtype="float32")
                BERT_selftext_inputs = np.array(BERT_selftext_inputs, dtype="float32")                
                
            print("LDA Embeddings Shape: ", LDA_inputs.shape)    
            print("BERT Selftext Embeddings Shape: ", BERT_selftext_inputs.shape)
        
      
        if concat:
            if title:
                BERT_selftext_inputs = np.add(BERT_selftext_inputs, BERT_title_inputs)/2
                return np.c_[LDA_inputs*gamma, BERT_selftext_inputs]           
            else:
                return np.c_[LDA_inputs*gamma, BERT_selftext_inputs]
        
                  
        else:
            if title:
                return LDA_inputs, BERT_selftext_inputs, BERT_title_inputs
            else:
                return LDA_inputs, BERT_selftext_inputs





