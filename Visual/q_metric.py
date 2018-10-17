import torch
import numpy as np

def define_Q_metric(env, model, num_states):
    states = np.zeros((num_states, env.state_stack, 84, 84))
    metric = QMetric(states, model)
    _ = env.reset()
    for i in range(num_states):
        action = np.random.randint(env.action_size)
        state, _, done = env.step(action)
        states[i] = state
        if done:
           env.reset()
    return metric

class QMetric():
    def __init__(self, states, model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.states = torch.from_numpy(states).float().to(self.device).detach()
        self.model = model 
        
    def evaluate(self):
        with torch.no_grad():
            m, _ = torch.max(self.model(self.states), dim=1)
            return torch.mean(m).item()

