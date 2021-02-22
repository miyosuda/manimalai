import rodentia
import os
import math
import random
import numpy as np

from envs.arena_config import ArenaConfig

LU_TYPE_L  = 1
LU_TYPE_L2 = 2
LU_TYPE_U  = 3


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
            width=width,
            height=height,
            bg_color=[0.19, 0.3, 0.47],
            shadow_buffer_width=1024,
            agent_radius=0.5)
        
        # Set light direction
        self.env.set_light(dir=[0.5, -1.0, -0.5],
                           color=[1.0, 1.0, 1.0],
                           ambient_color=[0.4, 0.4, 0.4],
                           shadow_rate=0.2)
        
        # Prepare default stage objects (wall, floor)
        self._prepare_fixed_stage()

        # Non fixed stage obj ids
        self.stage_obj_ids = []
        
        # Add additional camera for top view rendering
        self.additional_camera_id = self.env.add_camera_view(256, 256,
                                                             bg_color=[0.19, 0.3, 0.47],
                                                             far=50.0,
                                                             focal_length=30.0,
                                                             shadow_buffer_width=1024)
        
        # Reset stage
        self.reset()
        
    def _prepare_fixed_stage(self):
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

    def _locate_goal_obj(self, pos, size, rot, good, bounce, multi):
        if good:
            if multi:
                texture_path = self.data_path + "yellow.png"
            else:
                texture_path = self.data_path + "green.png"
        else:
            texture_path = self.data_path + "red.png"
        
        radius = size.x * 0.5
        pos = [pos.x-20, radius, -pos.z+20]

        impulse = None
        if rot is not None and bounce:
            IMPULSE_MAGNITUDE = 10.0
            impulse = [IMPULSE_MAGNITUDE * np.sin(rot),
                       0,
                       IMPULSE_MAGNITUDE * np.cos(rot)]
        
        ball_mass = 0.5
        obj_id = self.env.add_sphere(
            texture_path=texture_path,
            radius=radius,
            pos=pos,
            rot=0.0,
            mass=ball_mass,
            detect_collision=True)

        if impulse is not None:
            self.env.apply_impulse(obj_id, impulse)
        
        # TODO: Goaldのサイズ毎に報酬の値を変えるか?
        if good:
            self.reward_table[obj_id] = 1
        if not multi:
            self.terminate_obj_ids.add(obj_id)
        self.stage_obj_ids.append(obj_id)

    def _locate_wall_obj(self, pos, rot, color, size):
        pos = [pos.x-20, pos.y + size.y*0.5, -pos.z+20]
        color = [color.r/255.0, color.g/255.0, color.b/255.0]
        half_extent = [size.x*0.5, size.y*0.5, size.z*0.5]
        rot = 2.0 * math.pi * -rot / 360.0
        
        obj_id = self.env.add_box(
            color=color,
            half_extent=half_extent,
            pos=pos,
            rot=rot,
            mass=0.0,
            detect_collision=False)
        self.stage_obj_ids.append(obj_id)

    def _locate_ramp_obj(self, pos, rot, color, size):
        # TODO: 本当はmodelの中心をfbxに合わせるべき
        pos = [pos.x-20, pos.y, -pos.z+20]
        rot = 2.0 * math.pi * -rot / 360.0
        color = [color.r/255.0, color.g/255.0, color.b/255.0]
        scale = [size.x*0.5, size.y*0.5, size.z*0.5]
        
        model_path = self.data_path + "ramp.obj"
        obj_id = self.env.add_model(path=model_path,
                                    scale=scale,
                                    pos=pos,
                                    rot=rot,
                                    mass=0.0,
                                    color=color,
                                    detect_collision=False,
                                    use_mesh_collision=True)
        self.stage_obj_ids.append(obj_id)

    def _locate_cylinder_obj(self, pos, rot, size):
        pos = [pos.x-20, pos.y, -pos.z+20]
        rot = 2.0 * math.pi * -rot / 360.0
        scale = [size.x*0.5, size.y*0.5, size.z*0.5]
        
        model_path = self.data_path + "cylinder.obj"
        obj_id = self.env.add_model(path=model_path,
                                    scale=scale,
                                    pos=pos,
                                    rot=rot,
                                    mass=0.0,
                                    detect_collision=False,
                                    use_mesh_collision=True)
        self.stage_obj_ids.append(obj_id)

    def _locate_zone_obj(self, pos, rot, size, death):
        pos = [pos.x-20, pos.y+0.01, -pos.z+20]
        rot = 2.0 * math.pi * -rot / 360.0
        half_extent = [size.x*0.5, 0.01, size.z*0.5]

        if death:
            texture_path = self.data_path + "red.png"
        else:
            # TODO:
            texture_path = self.data_path + "white.png"
        
        obj_id = self.env.add_box(
            texture_path=texture_path,
            half_extent=half_extent,
            pos=pos,
            rot=rot,
            mass=0.0,
            detect_collision=True)
        self.stage_obj_ids.append(obj_id)
        if not death:
            self.hot_zone_obj_ids.add(obj_id)
        else:
            self.terminate_obj_ids.add(obj_id)

    def _locate_cardbox_obj(self, pos, rot, size, light):
        pos = [pos.x-20, pos.y + size.y*0.5, -pos.z+20]
        scale = [size.x*0.5,
                 size.y*0.5,
                 size.z*0.5]
        rot = 2.0 * math.pi * -rot / 360.0
        
        if light:
            mass = 1.0
            model_path = self.data_path + "cardbox1.obj"
        else:
            model_path = self.data_path + "cardbox2.obj"
            mass = 2.0
            
        obj_id = self.env.add_model(
            path=model_path,
            scale=scale,
            pos=pos,
            rot=rot,
            mass=mass,
            detect_collision=False,
            use_mesh_collision=False)
        self.stage_obj_ids.append(obj_id)

    def _locate_luobject_obj(self, pos, rot, size, lu_type):
        # TODO:
        pos = [pos.x-20, pos.y + size.y*0.5, -pos.z+20]
        rot = 2.0 * math.pi * -rot / 360.0
        # TODO: Wow swapping scale adjustment of x and z.
        scale = [size.x*0.1,
                 size.y*1.0,
                 size.z*0.3]
        
        if lu_type == LU_TYPE_L:
            model_path = self.data_path + "lobject.obj"
        elif lu_type == LU_TYPE_L2:
            model_path = self.data_path + "lobject2.obj"
        elif lu_type == LU_TYPE_U:
            model_path = self.data_path + "uobject.obj"
            
        mass = 1.0
        obj_id = self.env.add_model(path=model_path,
                                    scale=scale,
                                    pos=pos,
                                    rot=rot,
                                    mass=mass,
                                    detect_collision=False,
                                    use_mesh_collision=False,
                                    use_collision_file=True)
        self.stage_obj_ids.append(obj_id)

    def _reset_sub(self):
        # First clear remaining reward objects
        self._clear_objects()
        
        for item in self.arena.items:
            if item.name == "Agent":
                pos = item.positions[0]
                rot = item.rotations[0]
                rot = 2.0 * math.pi * -rot / 360.0
                self.env.locate_agent(pos=[pos.x-20, pos.y+0.5, -pos.z+20], rot_y=rot)

            elif item.name == "GoodGoal" or \
                 item.name == "GoodGoalBounce" or \
                 item.name == "BadGoal" or \
                 item.name == "BadGoalBounce" or \
                 item.name == "GoodGoalMulti" or \
                 item.name == "GoodGoalMultiBounce":
                name = item.name

                if "Good" in name:
                    good = True
                else:
                    good = False
                    
                if "Bounce" in name:
                    bounce = True
                else:
                    bounce = False

                if "Mutli" in name:
                    multi = True
                else:
                    multi = False
            
                for i in range(len(item.positions)):
                    pos = item.positions[i]
                    size = item.sizes[i]
                    if bounce:
                        rot = item.rotations[i]
                    else:
                        rot = None
                    self._locate_goal_obj(pos, size, rot, good, bounce, multi)
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
            elif item.name == "CylinderTunnelTransparent" or \
                 item.name == "CylinderTunnel":
                for i in range(len(item.positions)):
                    pos = item.positions[i]
                    rot = item.rotations[i]
                    size = item.sizes[i]
                    self._locate_cylinder_obj(pos, rot, size)
            elif item.name == "DeathZone" or item.name == "HotZone":
                for i in range(len(item.positions)):
                    pos = item.positions[i]
                    rot = item.rotations[i]
                    size = item.sizes[i]
                    self._locate_zone_obj(pos, rot, size,
                                          death=item.name=="DeathZone")
            elif item.name == "Cardbox1" or item.name == "Cardbox2":
                for i in range(len(item.positions)):
                    pos = item.positions[i]
                    rot = item.rotations[i]
                    size = item.sizes[i]
                    self._locate_cardbox_obj(pos, rot, size,
                                             light=item.name=="Cardbox1")
            elif item.name == "LObject" or \
                 item.name == "LObject2" or \
                 item.name == "UObject":
                 
                for i in range(len(item.positions)):
                    pos = item.positions[i]
                    rot = item.rotations[i]
                    size = item.sizes[i]
                    if item.name == "LObject":
                        lu_type = LU_TYPE_L
                    elif item.name == "LObject2":
                        lu_type = LU_TYPE_L2
                    elif item.name == "UObject":
                        lu_type = LU_TYPE_U
                    self._locate_luobject_obj(pos, rot, size, lu_type)
                
        # Reset environment and get screen
        obs = self.env.step(action=[0, 0, 0])
        screen = obs["screen"]
        return screen

    def reset(self):
        self.step_num = 0
        return self._reset_sub()

    def _clear_objects(self):
        # Remove object from the environment
        for id in self.stage_obj_ids:
            self.env.remove_obj(id)

        self.stage_obj_ids = []
        self.reward_table = {}
        self.terminate_obj_ids = set()
        self.hot_zone_obj_ids = set()   

    def _calc_hotzone_damage(self):
        # TODO: 時間に応じたダメージの計算
        return 1e-5
        
    def step(self, real_action):
        # TODO: actionの整理
        obs = self.env.step(action=real_action)
        self.step_num += 1
        
        screen = obs["screen"]
        collided = obs["collided"]

        # Check collision
        reward = 0
        terminal = False
        
        if len(collided) != 0:
            for id in collided:
                if id in self.reward_table:
                    reward += self.reward_table[id]
                    # Remove from reward table
                    del self.reward_table[id]
                    # Remove from stage object ids
                    self.stage_obj_ids.remove(id)
                    # Remove object from the environment
                    self.env.remove_obj(id)
                    
                if id in self.hot_zone_obj_ids:
                    reward -= self._calc_hotzone_damage()
                if id in self.terminate_obj_ids:
                    terminal = True

        if not terminal and self.step_num >= self.arena.t:
            # Time out
            terminal = True
            
        if terminal:
            screen = self._reset_sub()
        
        return screen, reward, terminal
    

    def get_top_view(self):
        # Capture stage image from the top view
        pos = [0, 40, 0]
        rot_x = -np.pi * 0.5
        rot = [np.sin(rot_x * 0.5), 0, 0, np.cos(rot_x * 0.5)]

        ret = self.env.render(self.additional_camera_id, pos, rot)
        return ret["screen"]
