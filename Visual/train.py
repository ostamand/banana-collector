import numpy as np
import argparse
from tqdm import trange
import sys
import os
import logging
#import matplotlib.pyplot as plt

from model import QNetwork
from visual_env import VisualEnvironment
from badaii.rl.agents.double_dqn import Agent
from badaii import helpers

import pdb 

# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ACTION_SIZE = 4 
SEED = 0

# Helpers 

def evaluate_policy(env, agent, episodes=100, steps=2000, eps=0.05):
    scores = []
    for i in range(episodes):
        score = 0
        state = env.reset()
        for _ in range(steps):
            action = agent.act(state, epsilon=eps)
            state, reward, done = env.step(action)
            score += reward
            if done:
                break
        scores.append(score)
    return np.mean(scores)

 # Taken from https://github.com/renatolfc/deep-rl-navigation/blob/visual/deeprl/train.py
 # Hack to get around env. memory leak
 # Process is restarted every reload_every episodes 
def reload_process(ckpt_file):
    if '--restore' not in sys.argv:
        sys.argv.append('--restore')
        sys.argv.append(ckpt_file)
    pdb.set_trace()
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    os.execv('/proc/self/exe', 'python -m deeprl.train'.split() + sys.argv[1:])
    
# Train 

def train(episodes=2000, steps=2000, env_file='data/Banana_x86_x64',
          out_file=None, restore=None, from_start=True, 
          reload_every=1000, log_every=10, action_repeat=4, update_frequency=1, 
          batch_size=32, gamma=0.99,lrate=2.5e-4, tau=0.05,
          replay_mem_size=100000, replay_start_size=5000, 
          ini_eps=1.0, final_eps=0.1, final_exp_it=200000):
    """Train Double DQN
    
    Args:
      episodes (int): Number of episodes to run 
      steps (int): Maximum number of steps per episode
      env_file (str): Path to environment file
    
    
    
    Returns:
        None
    """
    # Define agent 
    m = QNetwork(action_repeat, ACTION_SIZE, SEED)
    m_t = QNetwork(action_repeat, ACTION_SIZE, SEED)
    
    agent = Agent(
        m, m_t,
        action_size=ACTION_SIZE, 
        seed=SEED,
        batch_size=batch_size,
        gamma = gamma,
        update_frequency = update_frequency,
        lrate = lrate,
        replay_size = replay_mem_size,
        tau = tau,
        restore = restore
    )

    # Restore params from checkpoint if needed 
    it = 0 
    ep_start = 0
    if restore:
        logger.info('Restoring checkpoint...')
        if not from_start:
            it = agent.it-1
            ep_start = agent.run_params['episodes']-1
            
    # Create Unity Environment
    logger.info('Creating Unity virtual environment...')
    env = VisualEnvironment(env_file, action_repeat)

    # Train agent
    with trange(ep_start, episodes) as t:
        for ep_i in t:
            agent.reset_episode()
            state = env.reset()
            for _ in range(steps):
                # Decay exploration epsilon (linear decay)
                eps = max(final_eps,ini_eps-(ini_eps-final_eps)/final_exp_it*it)
                
                # Step agent 
                action = agent.act(state, epsilon=eps)
                next_state, reward, done = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
                it+=1 

            # Calculate score using policy epsilon=0.05 and 100 episodes
            if (ep_i+1) % log_every == 0:
                logger.info('Evaluating current policy...')
                score = evaluate_policy(env, agent)
                t.set_postfix(it=it,epsilon=eps, score=f'{score:.2f}')

            # Reload the environment to fix memory leak issues 
            if (ep_i+1) % reload_every == 0:
                logger.info('Reloading environment...')
                params = {
                    'episodes': ep_i+1,
                    'it': it
                    }
                agent.save(out_file, run_params=params)
                env.close()
                # reload_process()



                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Unity - Visual Banana Collector')  
    parser.add_argument("--env_file", help="Location of Unity env. file", default='data/Banana_x86_x64')
    parser.add_argument("--out_file", help="Checkpoint file", default='dbl_dqn_agent.ckpt')
    parser.add_argument("--restore", help="Restore checkpoint")
    args = parser.parse_args()

    train(
        env_file=args.env_file,
        out_file=args.out_file,
        restore=args.restore
    )


