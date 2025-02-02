import os
import gym
import numpy as np
from PIL import Image

import manimalai

def capture_task(task_id):
    env = gym.make('ManimalAI-v0', width=256, height=256, task_id=task_id)
    
    action = [1,1]
    state, reward, terminal, _ = env.step(action=action)
    
    top_image = env.get_top_view()
    
    pimage = Image.fromarray(top_image)
    path = "captures/{}.png".format(task_id)
    pimage.save(path)
    env.close()
    

def main():
    config_files = os.listdir("../manimalai/configurations")
    config_files.sort()

    if not os.path.exists("./captures"):
        os.makedirs("./captures")

    for file in config_files:
        if ".yml" in file:
            task_id = file[:-4]
            print(task_id)
            capture_task(task_id)


if __name__ == '__main__':
    main()
