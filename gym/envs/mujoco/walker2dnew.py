import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

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
        reward_vel = -((posafter - posbefore) / self.dt - 2)**2
        reward_alive = alive_bonus
        reward_act = -1e-3 * np.square(a).sum()
        reward_foot = lfoot_reward + rfoot_reward
        reward = reward_vel + reward_alive + reward_act + reward_foot
        print('rewards')
        print(reward_vel)
        print(reward_alive)
        print(reward_act)
        print(reward_foot)
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}
    
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
            self.lfoot_info = {'stance': True,
                               't_touch': self.model.data.time,
                               't_lift': None,
                               'xpos': self.model.data.xpos[4, 0]}
        else:
            self.rfoot_info = {'stance': True,
                               't_touch': self.model.data.time,
                               't_lift': None,
                               'xpos': self.model.data.xpos[7, 0]}            
    
    def update_foot_info(self, left):
        if left:
            if hasattr(self, 'lfoot_info'):
                if self.lfoot_info['stance']:
                    if self.check_foot_liftoff(left):
                        self.lfoot_info['stance'] = False
                        self.lfoot_info['t_lift'] = self.model.data.time
                else: 
                    # it only computes the reward during the swing mode
                    reward = self.compute_foot_reward(left)
                    if self.check_foot_strike(left):
                        self.init_foot_contact(left) 
                    else:
                        self.lfoot_info['xpos'] = self.model.data.xpos[4, 0]
                    return reward
            else:
                self.init_foot_contact(left)
        else:
            if hasattr(self, 'rfoot_info'):
                if self.rfoot_info['stance']:
                    if self.check_foot_liftoff(left):
                        self.rfoot_info['stance'] = False
                        self.rfoot_info['t_lift'] = self.model.data.time
                else:
                    # it only computes the reward during the swing mode
                    reward = self.compute_foot_reward(left)
                    if self.check_foot_strike(left):                        
                        self.init_foot_contact(left)
                    else:
                        self.rfoot_info['xpos'] = self.model.data.xpos[7, 0]
                    return reward                 
            else:
                self.init_foot_contact(left)
        return 0
    
    def compute_foot_reward(self, left, stride=0.6):
        if left:
            current_pos = self.model.data.xpos[4, 0]            
            reward = -((current_pos - self.lfoot_info['xpos'])/self.dt - 4)**2
        else:
            current_pos = self.model.data.xpos[7, 0]
            reward = -((current_pos - self.rfoot_info['xpos'])/self.dt - 4)**2          
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
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
