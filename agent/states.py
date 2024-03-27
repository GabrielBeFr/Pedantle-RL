import numpy as np
from math import ceil, log

N_STATES = 100

def compute_state(observation, offset):
    total_words = len(observation["fitted_words"])
    found_words = len(np.where(observation["words_prox"]==1)[0]) # [0] because np.where returns a tuple
    x = (found_words-offset) / (total_words-offset)
    x = log(1 + x/0.1) + x*(1 + log(0.1/1.1))
    state = N_STATES * ceil(x) # we create 100 possible states
    if state == N_STATES:
        state = int(N_STATES * (found_words-offset) / (total_words-offset))
    return state