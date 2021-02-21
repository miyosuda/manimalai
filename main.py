import numpy as np
import pygame, sys
from pygame.locals import *

from aai_environment import AAIEnvironment

BLACK = (0, 0, 0)


class Display(object):
    def __init__(self, display_size, config_path):
        self.width = display_size[0]
        self.height = display_size[1]
        
        self.env = AAIEnvironment(256, 256, config_path)
        
        pygame.init()
        
        #self.surface = pygame.display.set_mode(display_size, 0, 24)
        self.surface = pygame.display.set_mode(display_size, 0, 32)
        pygame.display.set_caption('mini-animalai')
        
        self.last_state = self.env.reset()

    def update(self):
        self.surface.fill(BLACK)
        self.process()
        pygame.display.update()

    def get_real_action(self):
        lookAction = 0
        moveAction = 0

        pressed = pygame.key.get_pressed()

        if pressed[K_a]:
            lookAction += 6
        if pressed[K_d]:
            lookAction -= 6
        if pressed[K_w]:
            moveAction += 1
        if pressed[K_s]:
            moveAction -= 1

        # TODO: actionの整理
        return [lookAction, 0, moveAction]

    def process(self):
        real_action = self.get_real_action()

        state, reward, terminal = self.env.step(real_action=real_action)

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
    #config_path = "./configurations/1-1-1.yml"
    #config_path = "./configurations/1-1-2.yml"
    #config_path = "./configurations/1-1-3.yml"
    #config_path = "./configurations/1-3-1.yml"
    #config_path = "./configurations/2-1-1.yml" # wall
    #config_path = "./configurations/3-16-1.yml" # ramp
    #config_path = "./configurations/3-28-1.yml" # maze
    #config_path = "./configurations/3-15-1.yml" # Cylinder
    #config_path = "./configurations/10-1-1.yml" # death zone, Cardbox2
    #config_path = "./configurations/10-25-1.yml" # Cardbox1
    #config_path = "./configurations/1-18-1.yml" # GoodGoalBounce
    #config_path = "./configurations/10-11-1.yml" # LObject, UObject
    #config_path = "./configurations/10-12-1.yml" # L2Object
    #config_path = "./configurations/10-13-1.yml" # LObject, LObject2, UObject
    #config_path = "./configurations/10-13-3.yml"
    config_path = "./debug_configurations/debug0.yml"
    
    display_size = (512, 256)
    display = Display(display_size, config_path)
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
