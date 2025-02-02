import gym
import numpy as np
import argparse
import pygame, sys
from pygame.locals import *

import manimalai

BLACK = (0, 0, 0)


class Display(object):
    def __init__(self, display_size, task_id):
        self.width = display_size[0]
        self.height = display_size[1]
        
        self.env = gym.make('ManimalAI-v0', width=256, height=256, task_id=task_id)
        
        pygame.init()
        
        self.surface = pygame.display.set_mode(display_size, 0, 32)
        pygame.display.set_caption('manimalai')
        
        self.last_state = self.env.reset()

    def update(self):
        self.surface.fill(BLACK)
        self.process()
        pygame.display.update()

    def get_action(self):
        lookAction = 1
        moveAction = 1

        pressed = pygame.key.get_pressed()

        if pressed[K_a]:
            lookAction += 1
        if pressed[K_d]:
            lookAction -= 1
        if pressed[K_w]:
            moveAction += 1
        if pressed[K_s]:
            moveAction -= 1

        return [lookAction, moveAction]

    def process(self):
        action = self.get_action()

        state, reward, terminal, info = self.env.step(action=action)

        top_image = self.env.get_top_view()

        if reward != 0:
            print("reward={}".format(reward))

        image = pygame.image.frombuffer(state, (256, 256), 'RGB')
        top_image = pygame.image.frombuffer(top_image, (256, 256), 'RGB')
        
        self.surface.blit(image, (0, 0))
        self.surface.blit(top_image, (256, 0))

        self.last_state = state

        if terminal:
            self.last_state = self.env.reset()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str,
                        default="1-1-1")
    args = parser.parse_args()
    task_id = args.task
    
    display_size = (512, 256)
    display = Display(display_size, task_id)
    clock = pygame.time.Clock()

    running = True
    FPS = 30

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

        display.update()
        clock.tick(FPS)


if __name__ == '__main__':
    main()
