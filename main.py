import matplotlib.pyplot as plt
import agent.Q_learning_agent as Q_learning_agent
import gym_examples
import gym
import time
from agent.states import compute_state, N_STATES
import numpy as np
import logging
from agent.propose_words import ProposeWords
import datetime
from tqdm import tqdm
from agent.actions import ACTIONS
from collections import defaultdict

if __name__ == "__main__":
    
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d %H:%M")
    logging.basicConfig(filename='logs/output_'+date_string+'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    env = gym.make(
        "gym_examples/Pedantle-v0", 
        render_mode=None, # else "human" 
        test_model=True, 
        wiki_file="data/wikipedia_april.csv",
        logging = logging,
        )
    
    model, index = env.get_model()
    agent_info = {"num_actions": len(ACTIONS), "num_states": N_STATES, "epsilon": 0.1, "step_size": 0.1, "discount": 1.0, "seed": 0}
    current_agent = Q_learning_agent.QLearningAgent()
    current_agent.agent_init(agent_info)

    state_visits = np.zeros(100)
    
    observation, _ = env.reset()
    punctuation = np.count_nonzero(observation["words_prox"]==1)
    words_class = ProposeWords(model, index, observation)
    logging.info(f"First observation is:{observation} \n")
    reward = 0
    to_start = True
    terminated = False

    for i in tqdm(range(3000)):
        state = compute_state(observation,  punctuation)
        state_visits[state] += 1
        logging.info(f"State: {state}")

        if to_start:
            action_id = current_agent.agent_start(state)
            action = list(ACTIONS.keys())[action_id]
            logging.info(f"Action: {action}")
            proposed_words = words_class.propose_words(action, observation, logging)

            to_start = False

        elif terminated:
            current_agent.agent_end(last_reward)

            observation, info = env.reset()
            punctuation = np.count_nonzero(observation["words_prox"]==1)
            words_class = ProposeWords(model, index, observation)
            terminated = False
            to_start = True
            reward = 0

        else:
            action_id = current_agent.agent_step(last_reward, state)
            action = list(ACTIONS.keys())[action_id]
            logging.info(f"Action: {action}")
            proposed_words = words_class.propose_words(action, observation, logging)

        for word in proposed_words:
            observation, last_reward, terminated, _, _ = env.step(word)
            reward += last_reward
            logging.info(f'The reward is: {reward}')

            if terminated:
                logging.info(f"Terminated! \n")
                break

    print(state_visits)
    col_labels = list(ACTIONS.keys())
    plt.figure(figsize=(5,10))
    plt.imshow(current_agent.q)
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(col_labels)), labels=col_labels)
    plt.show()
    env.close()