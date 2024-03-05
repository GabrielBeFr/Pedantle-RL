import gym_examples
import gym
import time
from agent import policy

if __name__ == "__main__":
    env = gym.make(
        "gym_examples/Pedantle-v0", 
        render_mode="human", 
        test_model=True, 
        wiki_file="/home/gabriel/cours/RL/projet/wikipedia_april.csv",
        )

    env.action_space.seed(42)

    observation = env.reset(seed=42)

    actions = ["and","fourth","be","the","of","year","is","be","for","day","always","often","between","come","can","do","in","common","start","first","second","April"]
    for i in range(20):
        action = actions[i]
        time.sleep(2)
        observation, reward, terminated, _, _ = env.step(action)

        if terminated:
            observation, info = env.reset()

    env.close()