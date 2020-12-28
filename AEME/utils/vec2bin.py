# Code to convert .vec/ /txt files to .bin for fast access
# File: vec2bin.py
# Autor: Atharva kulkarni

import argparse
from gensim.models.KeyedVectors import load_word2vec_format
from gensim.scripts.glove2word2vec import glove2word2vec
from time import process_time

#python vec2bin.py --file_path /home/eastwind/word-embeddings/fasttext/wiki-news-300d-1M.vec

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", default="/home/eastwind/word-embeddings/glove/glove.6B.300d.txt")
    args = parser.parse_args()

    
    if "glove" in str(args.file_path):
        start = process_time()
        print("\nConverting Glove to word2vec format...")
        output_path = str(args.file_path).split(".")[0] + ".word2vec" + str(args.file_path).split("/")[-1]
        glove2word2vec(glove_input_file=str(args.file_path), word2vec_output_file=output_path)
        
        print("\nLoading .vec or .txt embeddings...")
        model = load_word2vec_format(output_path, binary=False)
        
        print("\nSaving .bin embeddings...")
        model.save_word2vec_format(output_path+".bin", binary=True)
        
        print("\nDone!")
        end = process_time()
        print("\nTotal time taken: ", end-start)

     
        
    else:
        start = process_time()
        print("\nLoading .vec or .txt embeddings...")
        model = load_word2vec_format(str(args.file_path), binary=False) 
        
        print("\nSaving .bin embeddings...")
        model.save_word2vec_format(str(args.file_path)+".bin", binary=True)

        print("\nDone!")
        end = process_time()
        print("\nTotal time taken: ", end-start)
    
