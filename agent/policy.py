import numpy as np
from agent.actions import ACTIONS
from agent.states import compute_state

class Agent():
    def __init__(self, model) -> None:
        self.targetted_words = []
        self.model = model

    def policy(self, observation, logging):
        state = compute_state(observation)
        if observation["proposed_words"] == []:
            random_action = "random"
        else:
            random_action = np.random.choice(list(ACTIONS.keys()))

        logging.info(f"Random Policy")
        logging.info(f"State: {state}")
        logging.info(f"Action: {random_action}")

        words, target = ACTIONS[random_action](
            observation, 
            self.model, 
            self.targetted_words, 
            logging,
            )
        self.targetted_words.append(target)

        logging.info(f"Target: {target}")
        logging.info(f"Proposed word: {words}")

        return words