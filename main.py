import matplotlib.pyplot as plt
import algo
import gym_examples
import gym
import time
from agent.states import compute_state
import numpy as np
import logging
from agent.policy import Agent
import datetime
from tqdm import tqdm

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
    observation, _ = env.reset()
    agent = Agent(model, index, observation)
    logging.info(f"First observation is:{observation} \n")
    reward = 0

    agent_info = {"num_actions": 5, "num_states": 100, "epsilon": 0.1, "step_size": 0.1, "discount": 1.0, "seed": 0}
    current_agent = algo.QLearningAgent()
    current_agent.agent_init(agent_info)
    current_agent.agent_start(0)
    # print(current_agent)
    state_visits = np.zeros(100)
    _reward = 0
    punctuation = np.count_nonzero(observation["words_prox"]==1)
    state = compute_state(observation, punctuation)

    for i in tqdm(range(3000)):
        words = agent.policy(observation, logging,current_agent,_reward)
        state = compute_state(observation,  punctuation)
        state_visits[state] += 1
        for word in words:
            observation, _reward, terminated, _, _ = env.step(word)
            reward += _reward
            logging.info(f'The reward is: {reward}')

            if terminated:
                logging.info(f"Terminated! \n")
                observation, info = env.reset()
                agent = Agent(model, index, observation)

    print(state_visits)
    col_labels = ["list_classic_word","first_word","closest_word_of_random_word","closest_word_of_last_targetted_word","closest_of_closest_words"]
    plt.figure(figsize=(5,10))
    plt.imshow(current_agent.q)
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(col_labels)), labels=col_labels)
    plt.show()
    env.close()