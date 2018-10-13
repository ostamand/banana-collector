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
        reward = 0
        for _ in range(self.action_rpt):
            env_info = self.env.step(action)[self.brain_name]
            r = env_info.rewards[0]
            reward += r
        next_state = env_info.visual_observations[0].reshape(1,3,84,84)
        next_state = self.preprocess(next_state[0])
        done = env_info.local_done[0]
        return (next_state, reward, done)
    
    def reset(self, train=True):
        env_info = self.env.reset(train_mode=train)[self.brain_name]
        state = env_info.visual_observations[0].reshape(1,3,84,84)
        return self.preprocess(state[0])

    def preprocess(self, state_rgb):
        im = Image.fromarray(np.uint8(state_rgb*255).reshape(84,84,3), 'RGB')
        im_gray = im.convert('L')
        a = np.array(im_gray)
        return a.reshape(1, 1, a.shape[0],a.shape[1]) / 255.0