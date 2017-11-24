import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

foot_traj_filepath = '/home/caffe/Documents/baselines/baselines/ddpg/data/walker2d_foot_traj_xz.txt'
foot_traj = np.loadtxt(foot_traj_filepath)
leg_len = 0.95
body_height = [0.9, 1.2]
cycle_time = 1.0
speed = 2 # the average speed from the mocap data is 1.39

class Walker2dNewEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]    
        lfoot_reward = self.update_foot_info(True)
        rfoot_reward = self.update_foot_info(False)
        alive_bonus = 1.0
        #reward_vel = -((posafter - posbefore) / self.dt - speed)**2
        reward_vel = (posafter - posbefore)/self.dt
        reward_alive = alive_bonus
        reward_act = -1e-3 * np.square(a).sum()
        reward_foot = lfoot_reward + rfoot_reward
        reward = reward_vel + reward_alive + reward_act + reward_foot
        #print('rewards')
        #print(reward_vel)
        #print(reward_alive)
        #print(reward_act)
        #print(reward_foot)
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
            
        ob = self._get_obs()
        return ob, reward, done, {'reward_foot':reward_foot}
    
    def check_foot_strike(self, left, threshold=0.2):
        if left:
            zfoot_pos = self.model.data.xpos[4, 2]
        else:
            zfoot_pos = self.model.data.xpos[7, 2]        
        if zfoot_pos < threshold:
            return True
        else:
            return False
    
    def check_foot_liftoff(self, left, threshold=0.2):
        if left:
            zfoot_pos = self.model.data.xpos[4, 2]
        else:
            zfoot_pos = self.model.data.xpos[7, 2]        
        if zfoot_pos > threshold:
            return True
        else:
            return False        
    
    def init_foot_contact(self, left):
        if left:
            self.lfoot_info = {'start_t': self.model.data.time-cycle_time/4*3}
        else:
            self.rfoot_info = {'start_t': self.model.data.time-cycle_time/4}            
    
    def update_foot_info(self, left):
        reward = 0
        if left:
            if hasattr(self, 'lfoot_info'):
                reward = self.compute_foot_reward(left)
            else:
                self.init_foot_contact(left)
        else:
            if hasattr(self, 'rfoot_info'):
                reward = self.compute_foot_reward(left)                                  
            else:
                self.init_foot_contact(left)
        return reward
    

    
    def compute_foot_reward(self, left, stride=0.6):
        # compare the foot trajectory with respect to the desired foot trajectory
        hip_pos = np.array([self.model.data.xanchor[2, 0], self.model.data.xanchor[2, 2]])            
        if left:
            ankle_pos = np.array([self.model.data.xanchor[5, 0], self.model.data.xpos[5, 2]])
            ankle_to_hip = (hip_pos - ankle_pos)/leg_len
            timing = (self.model.data.time - self.lfoot_info['start_t'])%cycle_time/cycle_time
        else:
            ankle_pos = np.array([self.model.data.xanchor[8, 0], self.model.data.xanchor[8, 2]])
            ankle_to_hip = (hip_pos - ankle_pos)/leg_len
            timing = (self.model.data.time - self.rfoot_info['start_t'])%cycle_time/cycle_time
        index = int(timing*foot_traj.shape[0])
        if index >= foot_traj.shape[0]:
            index = foot_traj.shape[0] - 1
        foot_traj_pos = foot_traj[index]
        diff_vec = ankle_to_hip - foot_traj_pos
        height_reward = 0
        if hip_pos[1] < body_height[0]:
            height_reward = (hip_pos[1] - body_height[0])**2
        elif hip_pos[1] > body_height[1]:
            height_reward = (hip_pos[1] - body_height[1])**2
        #return -(diff_vec[0]**2 + diff_vec[1]**2*2 + height_reward*10)
        return -height_reward-np.linalg.norm(diff_vec)
        # return the difference in x and z axes as separate components
        # return abs(diff_vec[0]), abs(diff_vec[1])    
    
    def compute_stance_reward(self, left, duration=0.5):
        reward = 0   
        current_time = self.model.data.time
        if left:
            touch_time = self.lfoot_info['t_touch']
        else:
            touch_time = self.rfoot_info['t_touch']
        if (current_time - touch_time) > duration:
            reward = -((current_time - touch_time) - duration)**2
        return reward
                

    def _get_obs(self):
        qpos = self.model.data.qpos
        #print("qpos")
        #print(qpos)
        qvel = self.model.data.qvel
        #print("qvel")
        #print(qvel)
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        try:
            delattr(self, 'lfoot_info')
            delattr(self, 'rfoot_info')
        except:
            pass
        return self._get_obs()

    def viewer_setup(self):
        #self.viewer.cam.trackbodyid = 2
        #self.viewer.cam.distance = self.model.stat.extent * 0.5
        #self.viewer.cam.lookat[2] += .8
        self.viewer.cam.distance = 4.0
        self.viewer.cam.lookat[0] = self.model.data.qpos[0, 0]
        self.viewer.cam.lookat[1] = 0.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
