import numpy as np
from math import ceil

def compute_state(observation, offset):
    total_words = len(observation["fitted_words"])
    found_words = len(np.where(observation["words_prox"]==1)[0]) # [0] because np.where returns a tuple
    state = ceil(100 * (found_words-offset) / (total_words-offset)) # we create 100 possible states
    if state == 100:
        state = int(100 * (found_words-offset) / (total_words-offset))
    return state