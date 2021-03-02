import gym
import gym.spaces
import rodentia
import os
import numpy as np

from .arena_config import ArenaConfig, RGB, Vector3

LU_TYPE_L  = 1
LU_TYPE_L2 = 2
LU_TYPE_U  = 3


class Blackout:
    def __init__(self, pattern):
        self.pattern = pattern
        
    def is_blacked_out(self, step_num):
        if len(self.pattern) == 0:
            return False
        elif len(self.pattern) == 1 and self.pattern[0] < 0:
            pattern_index = step_num // (-self.pattern[0])
            return pattern_index % 2 == 1
        else:
            pattern_size = len(self.pattern)
            pattern_index = 0
            
            for i in range(pattern_size-1):
                start = self.pattern[i]
                end   = self.pattern[i+1]
                if start <= step_num and step_num < end:
                    pattern_index = i + 1
                    break
            if step_num >= self.pattern[-1]:
                pattern_index = len(self.pattern)

            return pattern_index % 2 == 1


class AAIEnvironment(gym.Env):
    def __init__(self, width=256, height=256, task_id="1-1-1", arena_index=0, debug=False):
        super().__init__()

        self.debug = debug

        # Gym setting
        self.action_space = gym.spaces.MultiDiscrete((3,3))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(height,width,3)
        )
        self.reward_range = [-np.inf, np.inf]
        
        # Load arena config yaml
        config_path = os.path.dirname(os.path.abspath(__file__)) + \
                      "/configurations/{}.yml".format(task_id)
        config = ArenaConfig(config_path)
        self.arena = config.arenas[arena_index]
        
        self.blackout = Blackout(self.arena.blackouts)
        
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

    def _convert_pos(self, pos, offset_y):
        if pos is not None:
            x = pos.x
            y = pos.y
            z = pos.z
            if x < 0:
                x = np.random.rand() * 40
            if z < 0:
                z = np.random.rand() * 40
            return [x-20, offset_y + y, -z+20]
        else:
            return [np.random.rand() * 40 - 20, offset_y, np.random.rand() * 40 - 20]

    def _convert_pos_inv(self, pos):
        return [pos[0]+20, pos[1], -pos[2]+20]

    def _convert_color(self, color):
        if type(color) is RGB:
            return [color.r/255.0, color.g/255.0, color.b/255.0]
        elif type(color) is Vector3:
            return [color.x/255.0, color.y/255.0, color.z/255.0]
        else:
            return [np.random.rand(), np.random.rand(), np.random.rand()]

    def _convert_rot(self, rot):
        if rot is not None:
            rot = 2.0 * np.pi * -rot / 360.0
        else:
            rot = np.random.rand() * 2.0 * np.pi
        return rot

    def _convert_rot_inv(self, rot):
        rot = -rot * 360.0 / (2.0 * np.pi)
        if rot < 0.0:
            rot += 360
        return rot
        
    def _prepare_fixed_stage(self):
        # Floor
        floor_texture_path = self.data_path + "stage/floor.png"
        
        self.env.add_box(
            texture_path=floor_texture_path,
            half_extent=[20.0, 1.0, 20.0],
            pos=[0.0, -1.0, 0.0],
            rot=0.0,
            detect_collision=False)

        # Wall
        wall_distance = 20.0
        
        wall_texture_path = self.data_path + "stage/wall.png"

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
        
    def _locate_agent(self, pos, rot):
        pos = self._convert_pos(pos, 0.5)
        rot = self._convert_rot(rot)
        self.env.locate_agent(pos, rot_y=rot)

    def _locate_goal_obj(self, pos, size, rot, good, bounce, multi):
        if good:
            if multi:
                texture_path = self.data_path + "misc/good_goal_multi.png"
            else:
                texture_path = self.data_path + "misc/good_goal.png"
        else:
            texture_path = self.data_path + "misc/bad_goal.png"
        
        radius = size.x * 0.5
        pos = self._convert_pos(pos, offset_y=radius)

        impulse = None
        if bounce:
            rot = self._convert_rot(rot)
            IMPULSE_MAGNITUDE = 10.0
            impulse = [-IMPULSE_MAGNITUDE * np.sin(rot),
                       0,
                       -IMPULSE_MAGNITUDE * np.cos(rot)]
        
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
        
        # TODO: Change reward depending on the goal size?
        if good:
            self.reward_table[obj_id] = 1
        if not multi:
            self.terminate_obj_ids.add(obj_id)
        self.stage_obj_ids.append(obj_id)

    def _locate_wall_obj(self, pos, rot, color, size):
        pos = [pos.x-20, pos.y + size.y*0.5, -pos.z+20]
        color = self._convert_color(color)
        half_extent = [size.x*0.5, size.y*0.5, size.z*0.5]
        rot = self._convert_rot(rot)
            
        obj_id = self.env.add_box(
            color=color,
            half_extent=half_extent,
            pos=pos,
            rot=rot,
            mass=0.0,
            detect_collision=False)
        self.stage_obj_ids.append(obj_id)

    def _locate_ramp_obj(self, pos, rot, color, size):
        pos = [pos.x-20, pos.y, -pos.z+20]
        rot = self._convert_rot(rot)
        color = self._convert_color(color)
        scale = [size.x*0.5, size.y*0.5, size.z*0.5]
        
        model_path = self.data_path + "immovable/ramp.obj"
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
        rot = self._convert_rot(rot)
        scale = [size.x*0.5, size.y*0.5, size.z*0.5]
        
        model_path = self.data_path + "immovable/cylinder.obj"
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
        rot = self._convert_rot(rot)
        half_extent = [size.x*0.5, 0.01, size.z*0.5]

        if death:
            texture_path = self.data_path + "immovable/death_zone.png"
        else:
            texture_path = self.data_path + "immovable/hot_zone.png"
        
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
        rot = self._convert_rot(rot)
        
        if light:
            model_path = self.data_path + "movable/cardbox1.obj"
            mass = 1.0
        else:
            model_path = self.data_path + "movable/cardbox2.obj"
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
        rot = self._convert_rot(rot)
        # TODO: now swapping scale adjustment of x and z.
        scale = [size.x*0.1,
                 size.y*1.0,
                 size.z*0.3]
        
        if lu_type == LU_TYPE_L:
            model_path = self.data_path + "movable/lobject.obj"
        elif lu_type == LU_TYPE_L2:
            model_path = self.data_path + "movable/lobject2.obj"
        elif lu_type == LU_TYPE_U:
            model_path = self.data_path + "movable/uobject.obj"
            
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

    def get_item_element_size(self, item):
        return np.max([len(item.positions),
                       len(item.rotations),
                       len(item.sizes),
                       len(item.colors)])

    def get_item_position(self, item, index):
        if index < len(item.positions):
            return item.positions[index]
        else:
            return None

    def get_item_rotation(self, item, index):
        if index < len(item.rotations):
            return item.rotations[index]
        else:
            return None

    def get_item_size(self, item, index):
        if index < len(item.sizes):
            return item.sizes[index]
        else:
            return None

    def get_item_color(self, item, index):
        if index < len(item.colors):
            return item.colors[index]
        else:
            return None

    def reset(self):
        self.step_num = 0 
        
        # First clear remaining reward objects
        self._clear_objects()
        
        for item in self.arena.items:
            if item.name == "Agent":
                pos = self.get_item_position(item, 0)
                rot = self.get_item_rotation(item, 0)
                self._locate_agent(pos, rot)

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

                if "Multi" in name:
                    multi = True
                else:
                    multi = False
            
                for i in range(self.get_item_element_size(item)):
                    pos = self.get_item_position(item, i)
                    size = self.get_item_size(item, i)
                    rot = self.get_item_rotation(item, i)
                    self._locate_goal_obj(pos, size, rot, good, bounce, multi)
            elif item.name == "Wall":
                for i in range(self.get_item_element_size(item)):
                    pos = self.get_item_position(item, i)
                    rot = self.get_item_rotation(item, i)
                    color = self.get_item_color(item, i)
                    size = self.get_item_size(item, i)
                    self._locate_wall_obj(pos, rot, color, size)
            elif item.name == "Ramp":
                for i in range(self.get_item_element_size(item)):
                    pos = self.get_item_position(item, i)
                    rot = self.get_item_rotation(item, i)
                    color = self.get_item_color(item, i)
                    size = self.get_item_size(item, i)
                    self._locate_ramp_obj(pos, rot, color, size)
            elif item.name == "CylinderTunnelTransparent" or \
                 item.name == "CylinderTunnel":
                for i in range(self.get_item_element_size(item)):
                    pos = self.get_item_position(item, i)
                    rot = self.get_item_rotation(item, i)
                    size = self.get_item_size(item, i)
                    self._locate_cylinder_obj(pos, rot, size)
            elif item.name == "DeathZone" or item.name == "HotZone":
                for i in range(self.get_item_element_size(item)):
                    pos = self.get_item_position(item, i)
                    rot = self.get_item_rotation(item, i)
                    size = self.get_item_size(item, i)
                    self._locate_zone_obj(pos, rot, size,
                                          death=item.name=="DeathZone")
            elif item.name == "Cardbox1" or item.name == "Cardbox2":
                for i in range(self.get_item_element_size(item)):
                    pos = self.get_item_position(item, i)
                    rot = self.get_item_rotation(item, i)
                    size = self.get_item_size(item, i)
                    self._locate_cardbox_obj(pos, rot, size,
                                             light=item.name=="Cardbox1")
            elif item.name == "LObject" or \
                 item.name == "LObject2" or \
                 item.name == "UObject":
                 
                for i in range(self.get_item_element_size(item)):
                    pos = self.get_item_position(item, i)
                    rot = self.get_item_rotation(item, i)
                    size = self.get_item_size(item, i)
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

    def _clear_objects(self):
        # Remove object from the environment
        for id in self.stage_obj_ids:
            self.env.remove_obj(id)

        self.stage_obj_ids = []
        self.reward_table = {}
        self.terminate_obj_ids = set()
        self.hot_zone_obj_ids = set()

    def _calc_hotzone_damage(self):
        # TODO: Calculate damage based on elapsed time.
        return 1e-5

    def _convert_to_real_action(self, action):
        real_action = [0, 0, 0]
        real_action[0] = (np.clip(action[0], 0, 2) - 1) * 6
        real_action[2] = np.clip(action[1], 0, 2) - 1
        return np.array(real_action, dtype=np.int32)

    def _get_agent_info(self):
        agent_info = self.env.get_agent_info()
        
        global_pos = agent_info["pos"]
        global_velocity = agent_info["velocity"]
        
        global_rot_y = agent_info["rot_y"]
        
        gx =  global_velocity[0]
        gy =  global_velocity[1]
        gz = -global_velocity[2]
        s = np.sin(-global_rot_y)
        c = np.cos(-global_rot_y)
        lx = c * gx - s * gz
        lz = s * gx + c * gz
        local_velociy = np.array([lx, gy, lz], dtype=np.float32)
        
        return local_velociy, \
            self._convert_pos_inv(global_pos), \
            self._convert_rot_inv(global_rot_y)
        
    def step(self, action):
        real_action = self._convert_to_real_action(action)
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
            screen = self.reset()

        agent_local_velociy, agent_global_pos, agent_global_rot_y = self._get_agent_info()
        
        info = {}
        info["local_velocity"] = agent_local_velociy
        if self.debug:
            # Set global position and rotation for debug purpose
            info["global_pos"] = agent_global_pos
            info["global_rot"] = agent_global_rot_y
        
        if self.blackout.is_blacked_out(self.step_num):
            # Black out screen
            screen = np.zeros_like(screen)
        
        return screen, reward, terminal, info
    

    def get_top_view(self):
        # Capture stage image from the top view
        pos = [0, 40, 0]
        rot_x = -np.pi * 0.5
        rot = [np.sin(rot_x * 0.5), 0, 0, np.cos(rot_x * 0.5)]

        ret = self.env.render(self.additional_camera_id, pos, rot)
        return ret["screen"]
