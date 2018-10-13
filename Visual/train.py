import numpy as np
import argparse
from tqdm import trange
import sys
import os
import logging

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
    for _ in range(episodes):
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

# https://stackoverflow.com/questions/31447442/difference-between-os-execl-and-os-execv-in-python
def reload_process():
    if '--restore' not in sys.argv:
        sys.argv.append('--restore')
        sys.argv.append(None)
    idx = sum( [ i if arg=='--restore' else 0 for i, arg in enumerate(sys.argv)] )
    sys.argv[idx+1] = 'reload.ckpt'
    os.execv(sys.executable, ['python', __file__, *sys.argv[1:]])
    
# Train 

def train(episodes=2000, steps=2000, env_file='data/Banana_x86_x64',
          out_file=None, restore=None, from_start=True, 
          reload_every=1000, log_every=10, action_repeat=4, update_frequency=1, 
          batch_size=32, gamma=0.99,lrate=2.5e-4, tau=0.05,
          replay_mem_size=100000, replay_start_size=5000, 
          ini_eps=1.0, final_eps=0.1, final_exp_it=200000, save_thresh=5.0):
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
    if 'reloading' in agent.run_params:
        from_start = agent.run_params['from_start']

    avg_scores = []
    scores = []
    last_saved_score = 0
    it = 0 
    ep_start = 0
    if restore and not from_start:
        it = agent.run_params['it']
        ep_start = agent.run_params['episodes']
        scores= agent.run_params['scores']
        avg_scores = agent.run_params['avg_scores']
        last_saved_score = agent.run_params['last_saved_score']

    if 'reloading' in agent.run_params:
        restore = agent.run_params['restore']
            
    # Create Unity Environment
    logger.info('Creating Unity virtual environment...')
    env = VisualEnvironment(env_file, action_repeat)

    # Train agent
    with trange(ep_start, episodes) as t:

        score = 0
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
                score += reward
                state = next_state
                if done:
                    break
                it+=1 
            scores.append((ep_i+1, score))
            t.set_postfix(it=it,epsilon=eps, score=f'{score:.2f}')

            # Calculate score using policy epsilon=0.05 and 100 episodes
            if (ep_i+1) % log_every == 0:
                logger.info('Evaluation current policy...')
                avg_score = evaluate_policy(env, agent)
                avg_scores.append((ep_i+1, avg_score))
                logger.info(f'Average score: {avg_score:.2f}')

                # Save agent if score is greater than threshold & last saved score
                if avg_score > save_thresh and avg_score > last_saved_score:
                    logger.info('Saving agent...')
                    params = {
                        'episodes': ep_i+1,
                        'it': it,
                        'avg_scores': avg_scores, 
                        'scores': scores,
                        'last_saved_score': last_saved_score
                        }
                    agent.save(out_file, run_params=params)

            # Reload the environment to fix memory leak issues 
            if (ep_i+1) % reload_every == 0:
                logger.info('Reloading environment...')
                params = {
                    'episodes': ep_i+1,
                    'it': it,
                    'restore': restore,
                    'from_start': False, 
                    'reloading': True,
                    'avg_scores': avg_scores, 
                    'last_saved_score': last_saved_score,
                    'scores': scores
                    }
                agent.save('reload.ckpt', run_params=params)
                env.close()
                reload_process()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Unity - Visual Banana Collector')  
    parser.add_argument("--env_file", help="Location of Unity env. file", default='data/Banana_x86_x64')
    parser.add_argument("--out_file", help="Checkpoint file", default='dbl_dqn_agent.ckpt')
    parser.add_argument("--restore", help="Restore checkpoint")
    parser.add_argument('--reload_every', help="Reload env. every number of episodes", default=1000)
    parser.add_argument("--log_every", help="Log metric every number of episodes", default=10)
    parser.add_argument("--episodes", help="Number of episodes to run", default=1000)
    parser.add_argument("--save_thresh", help="Saving threshold", default=10.0)
    args = parser.parse_args()

    train(
        env_file=args.env_file,
        out_file=args.out_file,
        restore=args.restore,
        reload_every=int(args.reload_every),
        log_every=int(args.log_every),
        episodes=int(args.episodes),
        save_thresh=float(args.save_thresh)
    )


