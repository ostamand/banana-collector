import torch
from replay_buffer import ReplayBuffer
from base_agent import BaseAgent

class Agent(BaseAgent):
    """Double DQN Agent with epsilon-greedy policy 

    Parameters
    ----------
    seed: int 
        Number for random seeding
    replay_size: int
        Size of the experience replay buffer
    batch_size: int
        Size of the batch used when learning
    gamma: float
        Discount rate
    lrate: int or float
        Learning rate 
    tau: float
        Soft target update rate
    """
    def __init__(self, model=None, model_target=None, action_size=None, seed=None,
                 replay_size=100000, batch_size=64, update_frequency=4,
                 gamma=0.99, lrate=5e-4, tau=0.001, restore=None):
        super(Agent, self).__init__(
            model, model_target, action_size, seed, 
            replay_size, batch_size, update_frequency, gamma, lrate, tau, restore
            )
        self.reset()

    def reset(self):
        self.it = 0
        self.memory = ReplayBuffer(self.action_size, self.replay_size, self.batch_size, self.seed, self.device)

    def step(self, state, action, reward, next_state, done, **kwargs):
        train = True
        if kwargs and 'train' in kwargs:
            train = kwargs['train']
        self.memory.add(state, action, reward, next_state, done)
        self.it += 1
        if train and self.it % self.update_frequency == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            _, best_actions = self.q_network(next_states).max(dim=1)
            # y^ = td_target
            # y^ = reward + gamma * Q^(next_state,argmax_a(next_state,a, w), w-), episode not terminal
            # y^ = reward, episode terminal
            td_targets = rewards + self.gamma * torch.gather(self.q_network_target(next_states),1,best_actions.view(-1,1))
            for i in range(self.batch_size):
                if dones[i].item() == 1.0:
                    td_targets[i] = rewards[i]

        # delta = y^-Q
        # clamp btwn -1..1
        delta = torch.clamp(td_targets-torch.gather(self.q_network(states), 1, actions), -1., 1.)
        # loss = sum (y^-Q)^2
        loss = torch.sum(torch.pow(delta, 2))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update()
