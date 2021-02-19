import rodentia
import os
import math
import random
import numpy as np

from envs.arena_config import ArenaConfig


class AAIEnvironment(object):
    def __init__(self, width, height, config_path, arena_index=0):
        # Load arena config yaml
        config = ArenaConfig(config_path)
        self.arena = config.arenas[arena_index]
        
        # Where model and texture data are located
        self.data_path = os.path.dirname(
            os.path.abspath(__file__)) + "/data/"
        
        # Create environment
        self.env = rodentia.Environment(
            width=width, height=height, bg_color=[0.19, 0.3, 0.47])
        # Set light direction
        self.env.set_light(dir=[0.5, -1.0, -0.5],
                           color=[1.0, 1.0, 1.0],
                           ambient_color=[0.4, 0.4, 0.4],
                           shadow_rate=0.2)
        
        # Prepare default stage objects (wall, floor)
        self._prepare_stage()
        
        # Object id for collision checking
        # TODO:
        self.plus_obj_ids_set = set()
        self.minus_obj_ids_set = set()
        
        # Add additional camera for top view rendering
        self.additional_camera_id = self.env.add_camera_view(256, 256,
                                                             bg_color=[0.19, 0.3, 0.47],
                                                             far=50.0,
                                                             focal_length=30.0,
                                                             shadow_buffer_width=1024)
        
        # Reset stage
        self.reset()
        
    def _prepare_stage(self):
        # Floor
        floor_texture_path = self.data_path + "floor0.png"
        
        self.env.add_box(
            texture_path=floor_texture_path,
            half_extent=[20.0, 1.0, 20.0],
            pos=[0.0, -1.0, 0.0],
            rot=0.0,
            detect_collision=False)

        # Wall
        wall_distance = 20.0
        
        wall_texture_path = self.data_path + "wall0.png"

        # -Z
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[wall_distance+1, 1.0, 1.0],
            pos=[0.0, 1.0, -wall_distance],
            rot=0.0,
            detect_collision=False)
        # +Z
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[wall_distance+1, 1.0, 1.0],
            pos=[0.0, 1.0, wall_distance],
            rot=0.0,
            detect_collision=False)
        # -X
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[1.0, 1.0, wall_distance+1],
            pos=[-wall_distance, 1.0, 0.0],
            rot=0.0,
            detect_collision=False)
        # +X
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[1.0, 1.0, wall_distance+1],
            pos=[wall_distance, 1.0, 0.0],
            rot=0.0,
            detect_collision=False)

        invisible_wall_half_height = 10.0
        # -Z (invisible)
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[wall_distance+1, invisible_wall_half_height, 1.0],
            pos=[0.0, invisible_wall_half_height, -wall_distance],
            rot=0.0,
            detect_collision=False,
            visible=False)
        # +Z (invisible)
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[wall_distance+1, invisible_wall_half_height, 1.0],
            pos=[0.0, invisible_wall_half_height, wall_distance],
            rot=0.0,
            detect_collision=False,
            visible=False)
        # -X (invisible)
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[1.0, invisible_wall_half_height, wall_distance+1],
            pos=[-wall_distance, invisible_wall_half_height, 0.0],
            rot=0.0,
            detect_collision=False,
            visible=False)
        # +X (invisible)
        self.env.add_box(
            texture_path=wall_texture_path,
            half_extent=[1.0, invisible_wall_half_height, wall_distance+1],
            pos=[wall_distance, invisible_wall_half_height, 0.0],
            rot=0.0,
            detect_collision=False,
            visible=False)

    def _locate_good_goal_obj(self, pos, size):
        texture_path = self.data_path + "green.png"
        radius = size.x * 0.5
        pos = [pos.x-20, radius, -pos.z+20]
        
        # If the object's mass is not 0, the it is simulated as a rigid body object.
        ball_mass = 0.5
        obj_id = self.env.add_sphere(
            texture_path=texture_path,
            radius=radius,
            pos=pos,
            rot=0.0,
            mass=ball_mass,
            detect_collision=True)
        self.plus_obj_ids_set.add(obj_id)

    def _locate_wall_obj(self, pos, rot, color, size):
        pos = [pos.x-20, pos.y + size.y*0.5, -pos.z+20]
        texture_path = self.data_path + "white.png"
        half_extent = [size.x*0.5, size.y*0.5, size.z*0.5]
        rot = 2.0 * math.pi * -rot / 360.0
        
        # TODO: obj_idは保存していない
        # TODO: resetまわりをどうするか決める必要あり
        obj_id = self.env.add_box(
            texture_path=texture_path,
            half_extent=half_extent,
            pos=pos,
            rot=rot,
            mass=0.0,
            detect_collision=False)

    def _locate_ramp_obj(self, pos, rot, color, size):
        # TODO: 本当はmodelの中心をfbxに合わせるべき
        pos = [pos.x-20, pos.y, -pos.z+20]
        rot = 2.0 * math.pi * -rot / 360.0
        scale = [size.x*0.5, size.y*0.5, size.z*0.5]
        
        model_path = self.data_path + "ramp0.obj"
        obj_id = self.env.add_model(path=model_path,
                                    scale=scale,
                                    pos=pos,
                                    rot=0.0,
                                    mass=0.0,
                                    detect_collision=False,
                                    use_mesh_collision=True)

    def _locate_cylinder_obj(self, pos, rot, size):
        pos = [pos.x-20, pos.y, -pos.z+20]
        rot = 2.0 * math.pi * -rot / 360.0
        scale = [size.x*0.5, size.y*0.5, size.z*0.5]
        
        model_path = self.data_path + "cylinder0.obj"
        obj_id = self.env.add_model(path=model_path,
                                    scale=scale,
                                    pos=pos,
                                    rot=0.0,
                                    mass=0.0,
                                    detect_collision=False,
                                    use_mesh_collision=True)        

    def _reset_sub(self):
        # First clear remaining reward objects
        self._clear_objects()
        
        for item in self.arena.items:
            if item.name == "Agent":
                pos = item.positions[0]
                rot = item.rotations[0]
                rot = 2.0 * math.pi * -rot / 360.0
                # TODO: 本当はagenty=0.5の位置になる
                self.env.locate_agent(pos=[pos.x-20, pos.y+1, -pos.z+20], rot_y=rot)
            elif item.name == "GoodGoal":
                for i in range(len(item.positions)):
                    pos = item.positions[i]
                    size = item.sizes[i]
                    self._locate_good_goal_obj(pos, size)
            elif item.name == "Wall":
                for i in range(len(item.positions)):
                    pos = item.positions[i]
                    rot = item.rotations[i]
                    color = item.colors[i]
                    size = item.sizes[i]
                    self._locate_wall_obj(pos, rot, color, size)
            elif item.name == "Ramp":
                for i in range(len(item.positions)):
                    pos = item.positions[i]
                    rot = item.rotations[i]
                    color = item.colors[i]
                    size = item.sizes[i]
                    self._locate_ramp_obj(pos, rot, color, size)
            elif item.name == "CylinderTunnelTransparent" or item.name == "CylinderTunnel":
                for i in range(len(item.positions)):
                    pos = item.positions[i]
                    rot = item.rotations[i]
                    size = item.sizes[i]
                    self._locate_cylinder_obj(pos, rot, size)
                
        # Reset environment and get screen
        obs = self.env.step(action=[0, 0, 0])
        screen = obs["screen"]
        return screen

    def reset(self):
        self.step_num = 0
        return self._reset_sub()

    def _clear_objects(self):
        # Create reward objects
        for id in self.plus_obj_ids_set:
            self.env.remove_obj(id)
        for id in self.minus_obj_ids_set:
            self.env.remove_obj(id)

        self.plus_obj_ids_set = set()
        self.minus_obj_ids_set = set()
        
    def step(self, real_action):
        # TODO: actionの整理
        obs = self.env.step(action=real_action)
        self.step_num += 1
        
        screen = obs["screen"]
        collided = obs["collided"]

        # Check collision
        reward = 0
        if len(collided) != 0:
            for id in collided:
                if id in self.plus_obj_ids_set:
                    reward += 1
                    self.plus_obj_ids_set.remove(id)
                elif id in self.minus_obj_ids_set:
                    reward -= 1
                    self.minus_obj_ids_set.remove(id)
                # Remove reward object from environment
                self.env.remove_obj(id)

        # Check if all positive rewards are taken
        is_empty = len(self.plus_obj_ids_set) == 0

        # Episode ends when step size exceeds MAX_STEP_NUM
        # TODO: now specifying longer time.
        terminal = self.step_num >= self.arena.t
        
        """
        # TODO:
        if (not terminal) and is_empty:
            screen = self._reset_sub()
        """
        
        return screen, reward, terminal

    def get_top_view(self):
        # Capture stage image from the top view
        pos = [0, 40, 0]
        rot_x = -np.pi * 0.5
        rot = [np.sin(rot_x * 0.5), 0, 0, np.cos(rot_x * 0.5)]

        ret = self.env.render(self.additional_camera_id, pos, rot)
        return ret["screen"]
