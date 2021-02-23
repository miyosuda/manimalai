import gym
import manimalai
from PIL import Image

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, state):
        return self.action_space.sample()

task_id = "1-1-1"
env = gym.make('ManimalAI-v0', width=256, height=256, task_id=task_id)
agent = RandomAgent(env.action_space)

state = env.reset()
for i in range(10):
    action = agent.choose_action(state)
    state, reward, terminal, _ = env.step(action=action)
    pimage = Image.fromarray(state)
    pimage.save("frame{}.png".format(i))
