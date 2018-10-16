from unityagents import UnityEnvironment
from PIL import Image
import numpy as np

class VisualEnvironment():

    def __init__(self, env_file, action_repeat=1, train=True):
        self.action_rpt = action_repeat
        self.env_file = env_file
        
        self.env = UnityEnvironment(file_name=env_file)
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size
        
    def step(self, action):
        next_state = np.zeros((1, self.action_rpt, 84, 84))            
        reward = 0
        for i in range(self.action_rpt):
            env_info = self.env.step(action)[self.brain_name]
            next_state[0,i] = self.preprocess(env_info.visual_observations[0].reshape(84, 84, 3))
            r = env_info.rewards[0]
            done = env_info.local_done[0]
            reward += r
            if done: 
                break
        return (next_state, reward, done)

    def sample(self):
        return np.random.randint(0, self.action_size)

    def reset(self, train=True):
        state = np.zeros((1, self.action_rpt, 84, 84))

        # Reset environment 
        env_info = self.env.reset(train_mode=train)[self.brain_name]
        state[0,0] = self.preprocess(env_info.visual_observations[0].reshape(84, 84, 3))

        # Random first action repeated action_rpt - 1 times
        action = np.random.randint(0, 4)
        for i in range(self.action_rpt-1):
            env_info = self.env.step(action)[self.brain_name]
            state[0,i+1] = self.preprocess(env_info.visual_observations[0].reshape(84, 84, 3))

        return state

    def preprocess(self, state_rgb):
        gray = 0.2989 * state_rgb[:,:,0] + 0.5870 * state_rgb[:,:,1] + 0.1140 * state_rgb[:,:,2]
        return gray
        #im = Image.fromarray(np.uint8(state_rgb*255), 'RGB')
        #im_gray = im.convert('L')
        #return np.array(im_gray) / 255.0

    def close(self):
        self.env.close()