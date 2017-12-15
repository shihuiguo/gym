import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

joint_list_dic = {'knee':  [[1, 2, 3], [6, 7], [3, 4]], 
                  'foot':  [[4, 5, 6], [7, 8], [4, 5]]} 
                 # first element is the indices in the trajectory file
                 # second element is the indices (parent, child) in the simulation model for the body parts on the right side
                 # third element is the indices (parent, child) in the simulation model for the body parts on the left side
traj_filepath = '/home/caffe/Documents/baselines/baselines/ddpg/data/walker2d_knee_foot_traj.txt'
traj_data = np.loadtxt(traj_filepath)
total_frames = traj_data.shape[0]
body_height = [0.9, 1.2]
cycle_time = 1.0
speed = 1.4 # the average speed from the mocap data is 1.39

class Walker2dNewEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]    
        alive_bonus = 1.0
        vel = (posafter - posbefore)/self.dt
        #if vel < 2.0 and vel > 1.0:
        #    reward_vel = 1.0
        #else:
        #    reward_vel = 0.0
        #reward_vel = 
        reward_vel =  2*np.exp(-(vel - speed)**2/2)
        reward_alive = alive_bonus        
        reward_pos = self.compute_posture_reward()        
        # reward_foot = self.compute_foot_reward()
        reward = reward_vel + reward_alive #+ reward_pos
        #print('rewards')
        #print(reward_vel)
        #print(reward_alive)
        #print(reward_act)
        #print(reward_foot)
        done = not (height > 0.8 and height < 1.4 and
                    ang > -1.0 and ang < 1.0)
            
        ob = self._get_obs()
        return ob, reward, done, {'reward_pos':reward_pos}
    
 
            
    def compute_posture_reward(self):
        timing = (self.model.data.time+cycle_time/4)%cycle_time/cycle_time
        index = self.get_index(timing)
        reward_height = self.compute_height_reward(index)
        #reward_head = self.compute_head_reward()
        reward_joints = 0
        for joint in ['knee', 'foot']:
            reward_L = self.compute_joint_reward(True, joint, index)
            reward_R = self.compute_joint_reward(False, joint, index)
            reward_joints = reward_joints + reward_L + reward_R
        posture_reward = reward_height + reward_joints
        return posture_reward

    def compute_foot_reward(self):
        joint_pos = self.model.data.xanchor
        geom_quat = self.model.data.geom_xmat
        foot_reward_L = 0
        foot_reward_R = 0
        if joint_pos[5, 2] < 0.15:
            foot_reward_L = min(geom_quat[4,0], 0)
        if joint_pos[8, 2] < 0.15:
            foot_reward_R = min(geom_quat[7, 0], 0)
        return foot_reward_L + foot_reward_R
    
    def compute_height_reward(self, index):
        target_height = traj_data[index, 0]
        simula_height = self.model.data.xanchor[2, 2] 
        reward_height =  2*np.exp(-(target_height - simula_height)**2/2)        
        return reward_height

    def compute_joint_reward(self, left, joint, index):
        '''
        joint: ['knee', 'foot']
        '''
        if left == False:
            index = int(index + total_frames/2)%total_frames
            side_index = 1 
        else:
            side_index = 2 
        target_joint_pos = traj_data[index, joint_list_dic[joint][0]]
        simula_parent_joint_pos = self.model.data.xanchor[joint_list_dic[joint][side_index][0], :]
        simula_child_joint_pos  = self.model.data.xanchor[joint_list_dic[joint][side_index][1], :]
        simula_joint_vec = simula_parent_joint_pos - simula_child_joint_pos
        simula_joint_pos = simula_joint_vec / np.linalg.norm(simula_joint_vec)
        dist = np.linalg.norm(target_joint_pos -simula_joint_pos)
        reward_joint_pos =  2*np.exp(-dist**2/2)
        return reward_joint_pos
        

    def get_index(self, timing):
        index = int(timing*total_frames)
        return index                

    def _get_obs(self):
        data = self.model.data
        #qpos = self.model.data.qpos
        #print("qpos")
        #print(qpos)
        #qvel = self.model.data.qvel
        #print("qvel")
        #print(qvel)
        return np.concatenate([data.qpos[1:].flat, 
                               np.clip(data.qvel.flat, -10, 10),
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])                              
                              

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
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
