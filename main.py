import gym_examples
import gym
import time
from agent import policy, ACTIONS

if __name__ == "__main__":
    env = gym.make(
        "gym_examples/Pedantle-v0", 
        render_mode="human", 
        test_model=True, 
        wiki_file="/home/gabriel/cours/RL/projet/wikipedia_april.csv",
        )
    
    model = env.get_model()
    observation, _ = env.reset()
    actions = ACTIONS

    for i in range(20):
        action = policy(observation)
        words = action(observation, model)
        time.sleep(0.5)

        reward = 0
        for word in words:
            observation, _reward, terminated, _, _ = env.step(word)
            reward += _reward

            if terminated:
                observation, info = env.reset()

    env.close()