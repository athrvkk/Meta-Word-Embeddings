# Main code to run
# File: main.py
# Author: Atharva Kulkarni

from utils import Utils
#from AEME import AEME

import argparse
from time import process_time
from google.colab import drive
drive.mount('/content/gdrive')


if __name__ == "__main__":
    """ Main method to generate Meta-Word-Embeddings """
    
    utils = Utils()
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_path", default="/content/gdrive/My Drive/Meta-Word-Embeddings/vocabulary/word2vec-glove-fasttext-intersection.txt")
    parser.add_argument("--word2vec_path", default="/content/gdrive/My Drive/Meta-Word-Embeddings/word-embeddings/GoogleNews-vectors-negative300.bin")
    parser.add_argument("--glove_path", default="/content/gdrive/My Drive/Meta-Word-Embeddings/word-embeddings/glove.word2vec.6B.300d.bin")
    parser.add_argument("--fasttext_path", default="/content/gdrive/My Drive/Meta-Word-Embeddings/word-embeddings/wiki-news-300d-1M.bin")
    args = parser.parse_args()
    
    # Get Intersection Vocabulary
    start = process_time()
    vocab = utils.read_vocab(str(args.vocab_path))
    print("\nTotal Intersection Vocabulary: ", len(vocab))
    end = process_time()
    print("\nTotal time taken: ", end-start)
    
    
    # Map embeddings
    start = process_time()
    print("\nLoading word2vec embeddings...")
    word2vec_dict = utils.get_embedding_dict(str(args.word2vec_path), vocab)
    end = process_time()
    print("\nTotal time taken: ", end-start)
    
    start = process_time()
    print("\nLoading glove embeddings...")
    glove_dict = utils.get_embedding_dict(str(args.glove_path), vocab)
    end = process_time()
    print("\nTotal time taken: ", end-start)
    
    start = process_time()
    print("\nLoading fasttext embeddings...")
    fasttext_dict = utils.get_embedding_dict(str(args.fasttext_path), vocab)
    end = process_time()
    print("\nTotal time taken: ", end-start)
    
    start = process_time()  
    print("\nMapping Embeddings...")  
    x_train1, x_train2, x_train3 = utils.map_embeddings(vocab, word2vec_dict, glove_dict, fasttext_dict)
    end = process_time()
    print("\nTotal time taken: ", end-start)
    print(x_train1.shape)
    print(x_train2.shape)
    print(x_train3.shape)
    



