from gym.envs.registration import register

register(
    id='gym_examples/Pedantle-v0',
    entry_point='gym_examples.envs:PedantleEnv',
    max_episode_steps=300,
)