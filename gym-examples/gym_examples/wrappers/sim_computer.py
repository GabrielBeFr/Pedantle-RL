# This file contains the code to compute similarity between two words.

import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin'

def load_embedding_model(test_model):
    '''
    Load the word2vec model from the file.

    params:
    - test_model: a bool: 
        *if True, load a small model for testing;
        *if False, load the full model.

    output:
    - the embedding model that will be used by the environment to compute similarity 
        between proposed and true words.
    '''
    if test_model:
        print("Loading small embedding model for testing")
        return KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True, limit=10000)
    else:
        print("Loading full embedding model for testing")
        return KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

def compute_similarity(word1, word2, model):
    '''
    Compute the cosine similarity between two words.
    
    params:
    - word1: a string representing the first word.
    - word2: a string representing the second word.
    
    output:
    - similarity: a float representing the cosine similarity between the two words.
    '''

    if word1 == word2:
        similarity = 1.0
    else:
        try:
            vec1, vec2 = model[word1], model[word2]
            similarity = cosine_similarity([vec1], [vec2]).item()
        except:
            similarity = 0.0

    return similarity