import torch
import torch.nn.functional as F
import numpy as np
from base_agent import BaseAgent
from p_replay_buffer import PrioritizedReplayBuffer

class Agent(BaseAgent):

    def __init__(self, *args, **kwargs):
        if args:
            model = args[0] if len(args) > 0 else None 
            model_target = args[1] if len(args) > 1 else None 
            action_size = args[2] if len(args) > 2 else None 
        if kwargs:
            seed = kwargs['seed'] if 'seed' in kwargs else 0 
            replay_size = kwargs['replay_size'] if 'replay_size' in kwargs else 100000 
            batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 64
            update_frequency = kwargs['update_frequency'] if 'update_frequency' in kwargs else 4 
            gamma = kwargs['gamma'] if 'gamma' in kwargs else 0.99
            lrate = kwargs['lrate'] if 'lrate' in kwargs else 5e-4
            tau = kwargs['tau'] if 'tau' in kwargs else 0.001
            restore = kwargs['restore'] if 'restore' in kwargs else None 

            self.training_starts = kwargs['training_starts'] if 'training_starts' in kwargs else 1000.0 
            self.alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.6 
            self.min_priority = kwargs['min_priority'] if 'min_priority' in kwargs else 0.1

        super(Agent, self).__init__(
            model, model_target, action_size, seed, 
            replay_size, batch_size, update_frequency, gamma, lrate, tau, restore
        )
        self.reset()

    def reset(self):
        self.it = 0
        self.memory = PrioritizedReplayBuffer(self.replay_size, self.alpha)

    def step(self, state, action, reward, next_state, done, **kwargs ):
        train = True 
        beta = 0.4
        if kwargs and 'beta' in kwargs:
            beta = kwargs['beta'] 
        if kwargs and 'train' in kwargs:
            train = kwargs['train']
        self.memory.add(state, action, reward, next_state, done)
        self.it += 1

        if train and self.it > self.training_starts and self.it % self.update_frequency == 0:
            experiences = self.memory.sample(self.batch_size, beta)
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones, weights, idxes = experiences

        # Convert to Tensor
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.device)
        weights = torch.from_numpy(weights[:, None]).float().to(self.device)

        with torch.no_grad():
            _ , best_actions = self.q_network(next_states).max(dim=1)
            # y^ = td_target
            # y^ = reward + gamma * Q^(next_state,argmax_a(next_state,a, w), w-), episode not terminal
            # y^ = reward, episode terminal
            td_targets = rewards.view(-1,1) + self.gamma * torch.gather(self.q_network_target(next_states),1,best_actions.view(-1,1))
            for i in range(self.batch_size):
                if dones[i].item() == 1:
                    td_targets[i] = rewards[i]  

        q_selected = torch.gather(self.q_network(states),1,actions.view(-1,1))

        # Update prioritized replay deltas
        with torch.no_grad():
            delta_raw = q_selected - td_targets
            p = torch.abs(delta_raw) + self.min_priority
            self.memory.update_priorities(idxes,p) 

        # delta_w  = lrate * sampling_weight * delta * dq^dw
        # where:
        #   sampling_weight = (1/(N*P(i))^beta
        loss = F.mse_loss(q_selected, td_targets, reduction='none') # Huber loss
        weighted_loss = torch.mean(weights * loss)
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # clip gradients if needed
        # torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2)
        self.optimizer.step()
        self.soft_update()
