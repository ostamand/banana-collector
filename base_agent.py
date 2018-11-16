import torch
import torch.optim as optim
import numpy as np

class BaseAgent():

    def __init__(self, model, model_target, action_size, seed, replay_size, batch_size, 
                 update_frequency,gamma, lrate, tau, restore):
        if restore is not None:
            d = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.checkpoint = torch.load(restore, map_location=d)

        self.run_params = {} if not restore else self.checkpoint['run_params']
        self.update_frequency = update_frequency if not restore else self.checkpoint['update_frequency']
        self.seed = seed if not restore else self.checkpoint['seed']
        self.action_size = action_size if not restore else self.checkpoint['action_size']
        self.replay_size = replay_size if not restore else self.checkpoint['replay_size']
        self.batch_size = batch_size if not restore else self.checkpoint['batch_size']
        self.gamma = gamma if not restore else self.checkpoint['gamma']
        self.lrate = lrate if not restore else self.checkpoint['lrate']
        self.tau = tau if not restore else self.checkpoint['tau']
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.q_network = model.to(self.device)
        self.q_network_target = model_target.to(self.device)
        
        if restore is not None:
            self.q_network.load_state_dict(self.checkpoint['state_dict'])
            self.q_network_target.load_state_dict(self.checkpoint['state_dict'])

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lrate)

    def act(self, state, epsilon=0.):
        """ Epsilon-Greedy policy
        
        """
        probs = epsilon * np.ones(self.action_size) / self.action_size
        state = torch.from_numpy(state).float().to(self.device)
        
        self.q_network.eval()
        with torch.no_grad():
            probs[np.argmax(self.q_network(state).cpu().numpy())] += 1-epsilon
        self.q_network.train()
        return np.random.choice(np.arange(self.action_size), p=probs)

    def sample(self):
        return np.random.randint(self.action_size)

    def soft_update(self):
        """Soft update of target network
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, param in zip(self.q_network_target.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

    def save(self, filename, run_params=None):
        checkpoint = self.params
        if run_params:
            checkpoint['run_params'] = run_params
        checkpoint['state_dict'] = self.q_network.state_dict()
        torch.save(checkpoint, filename)

    @property
    def params(self):
        params = {
            'action_size': self.action_size,
            'seed': self.seed,
            'replay_size': self.replay_size,
            'batch_size': self.batch_size,
            'gamma': self.gamma, 
            'lrate': self.lrate,
            'tau': self.tau, 
            'update_frequency': self.update_frequency
        }
        return params