import numpy as np

def compute_state(observation):
    total_words = len(observation["fitted_words"])
    found_words = len(np.where(observation["words_prox"]==1)[0]) # [0] because np.where returns a tuple
    state = int(100 * found_words / total_words) # we create 100 possible states
    return state