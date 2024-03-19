import numpy as np
from agent.actions import ACTIONS
from agent.states import compute_state
from pdb import set_trace
import re

class Agent():
    def __init__(self, model, observation) -> None:
        self.last_target_id = None
        self.pos_neg_words = {}
        self.model = model

        for i in range(len(observation["fitted_words"])):
            self.pos_neg_words[i] = {"negative": [], "positive": []}

    def _update_pos_neg(self, observation):
        if len(observation["proposed_words"]) != 0:
            last_proposed_word = observation["proposed_words"][-1]
            for i,word in enumerate(observation["fitted_words"]):
                if word == last_proposed_word:
                    self.pos_neg_words[i]["positive"].append(last_proposed_word)
                else:
                    self.pos_neg_words[i]["negative"].append(last_proposed_word)


    def policy(self, observation, logging):
        state = compute_state(observation)
        self._update_pos_neg(observation)

        if not any(re.match(r'[a-zA-Z0-9]', item) for item in observation["fitted_words"] if item is not None):
            random_action = "random"
        else:
            random_action = np.random.choice(list(ACTIONS.keys()))

        logging.info(f"Random Policy")
        logging.info(f"State: {state}")
        logging.info(f"Action: {random_action}")

        try:
            words, target = ACTIONS[random_action](
            observation, 
            self,
            logging,
            )
        except Exception as e:
            print(e)
            set_trace()
        if target is not None:
            self.last_target_id = target
            logging.info(f"Target id: {target}")
            logging.info(f"Target word: {observation['fitted_words'][target]}")

        logging.info(f"Proposed word: {words}")

        return words