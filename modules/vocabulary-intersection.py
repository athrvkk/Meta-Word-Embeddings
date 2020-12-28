# Code to generate vocabulary intersection of word2vec, GloVe, and fasttext
# File: vocabulary-intersection.py
# Autor: Atharva kulkarni

from utils import write_vocab
import argparse
from time import process_time
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

# python vocabulary-intersection.py

if __name__ == '__main__':
    """ The main method to receive user inputs"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--word2vec_filepath", default="/home/eastwind/word-embeddings/word2vec/GoogleNews-vectors-negative300.bin")
    parser.add_argument("--glove_filepath", default="/home/eastwind/word-embeddings/glove/glove.word2vec.6B.300d.bin")
    parser.add_argument("--fasttext_filepath", default="/home/eastwind/word-embeddings/fasttext/wiki-news-300d-1M.bin")
    parser.add_argument("--output_filepath", default="../vocabulary/word2vec-glove-fasttext-interseciton.txt")
    parser.add_argument("--operation", default="intersection")
    args = parser.parse_args()
    
    # Loading Word2vec embeddings
    print("\n\nLoading word2vec embeddings...")
    start = process_time()
    word2vec_vocab = set([word for word in KeyedVectors.load_word2vec_format(str(args.word2vec_filepath), binary=True).vocab.keys()])
    end = process_time()
    print("\nTotal word2vec vocab: ", len(word2vec_vocab))
    print("\nTotal time taken: ", end-start)
   

    # Loading Glove embeddings
    print("\n\nLoading Glove Embeddings...")
    start = process_time()
    glove_vocab = set([word for word in KeyedVectors.load_word2vec_format(str(args.glove_filepath), binary=True).vocab.keys()])
    end = process_time()
    print("\nTotal Glove vocab: ", len(glove_vocab))
    print("\nTotal time taken: ", end-start)
    

    # Loading fasttext embeddings
    print("\n\nLoading fasttext embeddings...")
    start = process_time()
    fasttext_vocab = set([word for word in KeyedVectors.load_word2vec_format(str(args.fasttext_filepath), binary=True).vocab.keys()])
    end = process_time()
    print("\nTotal fasttext vocab: ", len(fasttext_vocab))
    print("\nTotal time taken: ", end-start)

    if str(args.operation) == "intersection":
        vocab_intersection = word2vec_vocab & glove_vocab & fasttext_vocab
        print("\n\nIntersection vocab length: ", len(vocab_intersection))
        
    elif str(args.operation) == "union":
        vocab_union = word2vec_vocab | glove_vocab | fasttext_vocab
        print("\nUnion vocab length: ", len(vocab_union))
    
    write_vocab(str(args.output_filepath), vocab_intersection)

