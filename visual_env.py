from unityagents import UnityEnvironment
import numpy as np
from collections import deque

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
        self.states_history.append(env_info.visual_observations[0].transpose([0,3,1,2]))
        next_state = np.array(self.states_history).transpose([1,2,0,3,4])
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return (next_state, reward, done)

    def sample(self):
        return np.random.randint(0, self.action_size)

    # batch_size, channels (RGB), depth (state_stack), height, width
    def reset(self, train=True):
        self.states_history = deque(maxlen=self.state_stack)
        for _ in range(self.state_stack):
            self.states_history.append(np.zeros((1,3,84,84)))
        # Reset environment 
        env_info = self.env.reset(train_mode=train)[self.brain_name]
        self.states_history.append(env_info.visual_observations[0].transpose([0,3,1,2]))
        return np.array(self.states_history).transpose([1,2,0,3,4])

    def close(self):
        self.env.close()