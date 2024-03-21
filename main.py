import gym_examples
import gym
import time
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
        test_model=False, 
        wiki_file="data/wikipedia_april.csv",
        logging = logging,
        )
    
    model, index = env.get_model()
    observation, _ = env.reset()
    agent = Agent(model, index, observation)
    logging.info(f"First observation is:{observation} \n")
    for i in tqdm(range(3000)):
        words = agent.policy(observation, logging)
        reward = 0
        for word in words:
            observation, _reward, terminated, _, _ = env.step(word)
            reward += _reward

            if terminated:
                logging.info(f"Terminated! \n")
                observation, info = env.reset()
                agent = Agent(model, index, observation)

    env.close()