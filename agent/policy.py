import numpy as np
from agent.actions import ACTIONS
from agent.states import compute_state
from pdb import set_trace
import re

class Agent():
    def __init__(self, model, index, observation, memory_size = 15) -> None:
        self.last_target_id = None
        self.pos_neg_words = {}
        self.model = model
        self.index = index
        self.voc = model.key_to_index.keys()
        self.memory_size = memory_size

        for i in range(len(observation["fitted_words"])):
            self.pos_neg_words[i] = {"negative": [], "positive": []}

        self.punctuation = np.count_nonzero(observation["words_prox"]==1)

    def _update_pos_neg(self, observation):
        if len(observation["proposed_words"]) != 0:
            last_proposed_word = observation["proposed_words"][-1]
            if last_proposed_word in self.voc:
                for i,word in enumerate(observation["fitted_words"]):
                    if word == last_proposed_word:
                        self.pos_neg_words[i]["positive"].append(last_proposed_word)
                        self.pos_neg_words[i]["positive"] = self.pos_neg_words[i]["positive"][-self.memory_size:]
                    else:
                        self.pos_neg_words[i]["negative"].append(last_proposed_word)
                        self.pos_neg_words[i]["negative"] = self.pos_neg_words[i]["negative"][-self.memory_size:]


    def policy(self, observation, logging,current_agent,_reward):
        state = compute_state(observation, self.punctuation)
        self._update_pos_neg(observation)
        L=list(ACTIONS.keys())
        if not any(re.match(r'[a-zA-Z0-9]', item) for item in observation["fitted_words"] if item is not None): #pas de mots grisés
            random_action = "list_classic_word"
        else:
            # random_action = np.random.choice(list(ACTIONS.keys()))
            random_action = current_agent.agent_step(_reward,state)
            random_action = L[random_action]

        # logging.info(f"Random Policy")
        logging.info(f"State: {state}")
        logging.info(f"Action: {random_action}")

        words, target = ACTIONS[random_action](
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