from gym.envs.registration import register

register(
    id='ManimalAI-v0',
    entry_point='manimalai.environment:AAIEnvironment',
)
