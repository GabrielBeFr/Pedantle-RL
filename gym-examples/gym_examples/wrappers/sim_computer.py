# This file contains the code to compute similarity between two words.

import numpy as np

def compute_similarity(word1, word2):
    if word1 == word2:
        return 1
    return np.random.rand()