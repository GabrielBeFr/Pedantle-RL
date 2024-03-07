import numpy as np
from agent.actions import ACTIONS
from agent.states import compute_state

def policy(observation, logging):
    state = compute_state(observation)
    if observation["proposed_words"] == []:
        random_action = "random"
    else:
        random_action = np.random.choice(list(ACTIONS.keys()))
    logging.info(f"For the state: {state} \n the random policy chose the action: {random_action}.")
    return ACTIONS[random_action]