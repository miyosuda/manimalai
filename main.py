import gym
import numpy as np
import pygame, sys
from pygame.locals import *
from gym.envs.registration import register

register(id='MiniAnimalAI-v0',
         entry_point='aai_environment:AAIEnvironment')

BLACK = (0, 0, 0)


class Display(object):
    def __init__(self, display_size, task_id):
        self.width = display_size[0]
        self.height = display_size[1]
        
        self.env = gym.make('MiniAnimalAI-v0', width=256, height=256, task_id=task_id)
        
        pygame.init()
        
        #self.surface = pygame.display.set_mode(display_size, 0, 24)
        self.surface = pygame.display.set_mode(display_size, 0, 32)
        pygame.display.set_caption('mini-animalai')
        
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

        state, reward, terminal, _ = self.env.step(action=action)

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

    def get_frame(self):
        data = self.surface.get_buffer().raw
        return data


def main():
    #task_id = "1-1-1"
    #task_id = "1-1-2"
    #task_id = "1-1-3"
    #task_id = "1-3-1"
    #task_id = "2-1-1" # wall
    #task_id = "3-16-1" # ramp
    #task_id = "3-28-1" # maze
    #task_id = "3-15-1" # Cylinder
    #task_id = "10-1-1" # death zone, Cardbox2
    #task_id = "10-25-1" # Cardbox1
    #task_id = "1-18-1" # GoodGoalBounce
    #task_id = "10-11-1" # LObject, UObject
    #task_id = "10-12-1" # L2Object
    #task_id = "10-13-1" # LObject, LObject2, UObject
    #task_id = "10-13-3"
    #task_id = "debug0"
    #task_id = "debug1"
    task_id = "2-27-3"
    
    display_size = (512, 256)
    display = Display(display_size, task_id)
    clock = pygame.time.Clock()

    running = True
    FPS = 30

    recording = False
    
    if recording:
        from movie_writer import MovieWriter
        writer = MovieWriter("out.mov", display_size, FPS)
    else:
        writer = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

        display.update()
        clock.tick(FPS)

        if writer is not None:
            frame_str = display.get_frame()
            d = np.fromstring(frame_str, dtype=np.uint8)
            d = d.reshape((display_size[1], display_size[0], 4))
            d = d[:,:,:3]
            writer.add_frame(d)

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
