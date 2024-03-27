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
import datetime
import json

def run_episode(current_agent, env, model, index, state_visits, max_words_per_episode, logging):
    
    nb_proposed_words = 0
    observation, _ = env.reset()
    punctuation = np.count_nonzero(observation["words_prox"]==1)
    words_class = ProposeWords(model, index, observation)
    logging.info(f"First observation is:{observation} \n")
    reward = 0
    to_start = True
    terminated = False

    t_init = datetime.datetime.now()
    while nb_proposed_words < max_words_per_episode:
        state = compute_state(observation,  punctuation)
        state_visits[state] += 1

        logging.info(f"Time for the iteration: {datetime.datetime.now()-t_init}")
        t_init = datetime.datetime.now()

        logging.info(f"State: {state}")

        if to_start:
            action_id = current_agent.agent_start(state)
            action = list(ACTIONS.keys())[action_id]
            logging.info(f"Action: {action}")
            proposed_words = words_class.propose_words(action, observation, logging)

            to_start = False

        elif terminated:
            current_agent.agent_end(last_reward)
            return nb_proposed_words

        else:
            action_id = current_agent.agent_step(last_reward, state)
            action = list(ACTIONS.keys())[action_id]
            logging.info(f"Action: {action}")
            t_action = datetime.datetime.now()
            proposed_words = words_class.propose_words(action, observation, logging)
            logging.info(f"Time for this action: {datetime.datetime.now()-t_action}")

        for word in proposed_words:
            observation, last_reward, terminated, _, _ = env.step(word)
            reward += last_reward
            nb_proposed_words += 1
            logging.info(f'The reward is: {reward}')

            if terminated:
                logging.info(f"Terminated! \n")
                break

    return nb_proposed_words

if __name__ == "__main__":
    num_episodes = 1
    max_words_per_episode = 500
    
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d %H:%M")
    logging.basicConfig(filename='logs/output_'+date_string+'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    env = gym.make(
        "gym_examples/Pedantle-v0", 
        render_mode="human", # else "human" 
        test_model=True, 
        #wiki_file="data/wikipedia_dataset.csv",
        logging = logging,
        )
    
    model, index = env.get_model()
    agent_info = {"num_actions": len(ACTIONS), "num_states": N_STATES, "epsilon": 0.1, "step_size": 0.1, "discount": 1.0, "seed": 0}
    current_agent = Q_learning_agent.QLearningAgent()
    current_agent.agent_init(agent_info)

    state_visits = np.zeros(100)
    nb_words = []

    for i in range(num_episodes):
        logging.info(f"Episode: {i}")
        print(f"Episode: {i}")
        nb_words.append(run_episode(current_agent, env, model, index, state_visits, max_words_per_episode, logging))
        logging.info(f"State visits: {state_visits}")
        print(f"Amount of words to complete this episode: {nb_words[-1]}")

    print("******** Results *********")
    print(f"state_visits: {state_visits}")
    print(f"nb_words: {nb_words}")
    print(f"current_agent.q: {current_agent.q}")

    # After the loop ends
    results = {
        "q": current_agent.q.tolist(),
        "nb_words": nb_words,
        "state_visits": state_visits.tolist()
    }

    # Save the results in a JSON file
    with open("results.json", "w") as file:
        json.dump(results, file)


    col_labels = list(ACTIONS.keys())
    plt.figure(figsize=(5,10))
    plt.imshow(current_agent.q)
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(col_labels)), labels=col_labels)
    plt.savefig("logs/q_values.png")
    env.close()
