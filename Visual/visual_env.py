from unityagents import UnityEnvironment
#from PIL import Image
import numpy as np
from collections import deque

STATE_SIZE = (84,84)

class VisualEnvironment():

    def __init__(self, env_file, state_stack=4, train=True):
        self.state_stack = state_stack
        self.env_file = env_file
        
        self.env = UnityEnvironment(file_name=env_file)
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        self.action_size = brain.vector_action_space_size
        
    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        self.states_history.append(self.preprocess(env_info.visual_observations[0]))
        next_state = np.array(self.states_history).reshape((1, self.state_stack, *STATE_SIZE))
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return (next_state, reward, done)

    def sample(self):
        return np.random.randint(0, self.action_size)

    def reset(self, train=True):
        self.states_history = deque(maxlen=self.state_stack)
        for _ in range(self.state_stack):
            self.states_history.append(np.zeros(STATE_SIZE))
        # Reset environment 
        env_info = self.env.reset(train_mode=train)[self.brain_name]
        self.states_history.append(self.preprocess(env_info.visual_observations[0]))
        return np.array(self.states_history).reshape((1, self.state_stack, *STATE_SIZE))

    def preprocess(self, state_rgb):
        gray = 0.2989 * state_rgb[0,:,:,0] + 0.5870 * state_rgb[0,:,:,1] + 0.1140 * state_rgb[0,:,:,2]
        return gray

    def close(self):
        self.env.close()