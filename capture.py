import os
import gym
import numpy as np
from PIL import Image
from gym.envs.registration import register

register(id='MiniAnimalAI-v0',
         entry_point='aai_environment:AAIEnvironment')


def capture_task(task_id):
    env = gym.make('MiniAnimalAI-v0', width=256, height=256, task_id=task_id)
    
    action = [1,1]
    state, reward, terminal, _ = env.step(action=action)
    
    top_image = env.get_top_view()
    
    pimage = Image.fromarray(top_image)
    path = "captures/{}.png".format(task_id)
    pimage.save(path)
    env.close()
    

def main():
    config_files = os.listdir("./configurations")
    config_files.sort()

    # [ok]
    # 1-
    # 2-
    # 3-
    # 10-
    
    for file in config_files:
        if ".yml" in file:
            task_id = file[:-4]
            print(task_id)
            capture_task(task_id)
    

if __name__ == '__main__':
    main()
