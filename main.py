import gym_examples
import gym
import time
import logging
from agent.policy import policy
import datetime

if __name__ == "__main__":

    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d %H:%M")
    logging.basicConfig(filename='logs/output_'+date_string+'.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    env = gym.make(
        "gym_examples/Pedantle-v0", 
        render_mode="human", 
        test_model=True, 
        wiki_file="/home/gabriel/cours/RL/projet/wikipedia_april.csv",
        logging = logging,
        )
    
    model = env.get_model()
    observation, _ = env.reset()

    for i in range(20):
        action = policy(observation, logging)
        words = action(observation, model, logging)
        time.sleep(0.5)

        reward = 0
        for word in words:
            observation, _reward, terminated, _, _ = env.step(word)
            reward += _reward

            if terminated:
                observation, info = env.reset()

    env.close()