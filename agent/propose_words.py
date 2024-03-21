import numpy as np
from agent.actions import ACTIONS
from pdb import set_trace
import re

class ProposeWords():
    def __init__(self, model, index, observation, memory_size = 15) -> None:
        self.last_target_id = None
        self.pos_neg_words = {}
        self.model = model
        self.index = index
        self.voc = model.key_to_index.keys()
        self.memory_size = memory_size

        self.punctuation = np.count_nonzero(observation["words_prox"]==1)

    def propose_words(self, action, observation, logging):
        words, target = ACTIONS[action](
        observation, 
        self,
        logging,
        )
        if target is not None:
            self.last_target_id = target
            logging.info(f"Target id: {target}")
            logging.info(f"Target word: {observation['fitted_words'][target]}")

        logging.info(f"Proposed word: {words}")

        return words