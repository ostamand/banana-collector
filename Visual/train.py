import numpy as np
import argparse
import sys
import os
import logging
import datetime 

from model import QNetwork
from visual_env import VisualEnvironment
from badaii.agents.dbl_dqn import Agent
from badaii.agents.p_dbl_dqn import Agent as PrioAgent
from badaii.moving_result import MovingResult
from q_metric import define_Q_metric, QMetric

import pdb 

ACTION_SIZE = 4 
SEED = 0

# Logging
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

# Helpers

def save(agent, out_file, ep, it, avg_scores, scores, q_metrics, last_saved_score):
    params = { 
        'episodes': ep,
        'it': it,
        'avg_scores': avg_scores, 
        'scores': scores,
        'q_metrics': q_metrics,
        'last_saved_score': last_saved_score
        }
    agent.save(out_file, run_params=params)

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

def initialize_replay_buffer(agent, env, steps=1000):
    it = 0
    state = env.reset() 
    while it < steps:
        action = env.sample()
        next_state, reward, done = env.step(action)
        agent.step(state, action, reward, next_state, done, train=False)
        it += 1
        if done:
            state = env.reset()
        else:
            state = next_state

# https://stackoverflow.com/questions/31447442/difference-between-os-execl-and-os-execv-in-python
def reload_process():
    if '--restore' not in sys.argv:
        sys.argv.append('--restore')
        sys.argv.append(None)
    idx = sum( [ i if arg=='--restore' else 0 for i, arg in enumerate(sys.argv)] )
    sys.argv[idx+1] = 'reload.ckpt'
    os.execv(sys.executable, ['python', __file__, *sys.argv[1:]])

# Train 
def train(episodes=10000, 
          steps=2000, 
          final_exp_ep=5000, 
          env_file='data/Banana_x86_x64',
          out_file=None, 
          restore=None, 
          from_start=True, 
          reload_every=1000, 
          ckpt_every=1000,
          log_every=500, 
          state_stack=4, 
          update_frequency=1, 
          batch_size=32, 
          gamma=0.99,
          lrate=5.0e-4, 
          tau=0.001,
          replay_mem_size=500000, 
          replay_start_size=10000, 
          ini_eps=1.0, 
          final_eps=0.10, 
          save_thresh=5.0,
          prio=False, 
          min_priority=1e-6, 
          alpha=0.1, 
          final_beta=1.0, 
          ini_beta=0.4
           ):
    """Train Double DQN
    
    Args:
      episodes (int): Number of episodes to run 
      steps (int): Maximum number of steps per episode
      env_file (str): Path to environment file
      out_file (str): Output checkpoint name
      restore (str): Restore the specified checkpoint before starting the training 
      from_start (bool): Force the training to start from the start
      reload_evey (int): Reload environment every # of episodes
    
    Returns:
        None
    """
    # Define agent 
    logger.info('Creating agent...')
    m = QNetwork(state_stack, ACTION_SIZE, SEED)
    m_t = QNetwork(state_stack, ACTION_SIZE, SEED)
    
    if prio:
        agent = PrioAgent(m, m_t, ACTION_SIZE, 
            seed=SEED, batch_size=batch_size, gamma = gamma, update_frequency = update_frequency,
            lrate = lrate, replay_size = replay_mem_size, tau = tau, restore = restore, 
            min_priority = min_priority, alpha = alpha
        )
    else:
        agent = Agent(m, m_t, ACTION_SIZE,    
            seed=SEED, batch_size=batch_size, gamma = gamma, update_frequency = update_frequency,
            lrate = lrate, replay_size = replay_mem_size, tau = tau, restore = restore
        )

    # Create Unity Environment
    logger.info('Creating Unity virtual environment...')
    env = VisualEnvironment(env_file, state_stack)

    # Restore params from checkpoint if needed 
    if 'reloading' in agent.run_params:
        from_start = agent.run_params['from_start']

    if restore and not from_start:
        logger.info('Restoring params...')
        it = agent.run_params['it']
        ep_start = agent.run_params['episodes']
        scores= agent.run_params['scores']
        avg_scores = agent.run_params['avg_scores']
        last_saved_score = agent.run_params['last_saved_score']
        q_metric = QMetric(agent.run_params['q_metric_states'], m)
        q_metrics = agent.run_params['q_metrics']
    else:
        avg_scores = []
        scores = MovingResult()
        last_saved_score = 0
        it = 0 
        ep_start = 0
        q_metric = define_Q_metric(env, m, 100)
        q_metrics = []

    if 'reloading' in agent.run_params:
        restore = agent.run_params['restore']

    # Initialize replay buffer with random actions 
    logger.info("Initialize replay buffer with random actions...")
    initialize_replay_buffer(agent, env, steps=replay_start_size)
            
    # Train agent
    logger.info('Training')
    for ep_i in range(ep_start, episodes):
            score = 0
            agent.reset_episode()
            state = env.reset()

            # Decay exploration epsilon (linear decay)
            eps = max(final_eps,ini_eps-(ini_eps-final_eps)/final_exp_ep*ep_i)
            bta = min(final_beta,ini_beta-(ini_beta-final_beta)/final_exp_ep*ep_i)

            for _ in range(steps):
                # Step agent 
                action = agent.act(state, epsilon=eps)
                next_state, reward, done = env.step(action)
                agent.step(state, action, reward, next_state, done, beta=bta)
                score += reward
                state = next_state
                if done:
                    break
                it+=1 

            # Update metrics  
            q_metrics.append((ep_i+1, q_metric.evaluate()))
            scores.add(score)
            logger.info(f'ep={ep_i+1}/{episodes}, it={it}, epsilon={eps:.3f}, score={scores.last:.2f}, q_eval={q_metrics[-1][1]:.2f}')

            # Calculate score using policy epsilon=0.05 and 100 episodes
            if (ep_i+1) % log_every == 0:
                logger.info('Evaluating current policy...')
                avg_score = evaluate_policy(env, agent)
                avg_scores.append((ep_i+1, avg_score))
                logger.info(f'Average score: {avg_score:.2f}')

                # Save agent if score is greater than threshold & last saved score
                if avg_score > save_thresh and avg_score > last_saved_score:
                    logger.info("Saving checkpoint...")
                    save(agent, out_file, 
                         ep_i+1, it, avg_scores, scores, q_metrics, last_saved_score
                    )

            # Save checkpoint if needed
            if (ep_i+1) % ckpt_every == 0:  
                s = os.path.splitext(out_file)
                tm = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
                filename = f'{s[0]}_{tm}{s[1]}'
                logger.info(f'Saving checkpoint {filename}...')
                save(
                    agent, filename, ep_i+1, it, avg_scores, scores, q_metrics, last_saved_score
                )

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
                    'scores': scores,
                    'q_metric_states': q_metric.states.cpu().numpy(),
                    'q_metrics': q_metrics
                    }
                agent.save('reload.ckpt', run_params=params)
                env.close()
                reload_process()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Unity - Visual Banana Collector')  
    parser.add_argument("--env_file", help="Location of Unity env. file", default='data/Banana_x86_x64')
    parser.add_argument("--out_file", help="Checkpoint file", default='dbl_dqn_agent.ckpt')
    parser.add_argument("--restore", help="Restore checkpoint")
    parser.add_argument('--reload_every', help="Reload env. every x episodes", default=1000)
    parser.add_argument("--ckpt_every", help="Save checkpoint every x episodes", default=1000)
    parser.add_argument("--log_every", help="Log metric every number of episodes", default=10)
    parser.add_argument("--episodes", help="Number of episodes to run", default=1000)
    parser.add_argument("--save_thresh", help="Saving threshold", default=10.0)
    parser.add_argument("--final_exp_ep", help="final exploaration episode", default=2500)
    parser.add_argument("--ini_eps", help="initial epsilon", default=1.0)
    parser.add_argument("--final_eps", help="final epsilon", default=0.1)
    parser.add_argument("--ini_beta", help="initial beta", default=0.4)
    parser.add_argument("--final_beta", help="final beta", default=1.0)
    parser.add_argument("--prio", help="With or without prioritized experience replay", default=False)
    parser.add_argument("--lrate", help="Learning rate", default=5.0e-4)
    parser.add_argument("--replay_start_size", help="Replay start size", default=5000)
    args = parser.parse_args()

    train(
        env_file=args.env_file,
        out_file=args.out_file,
        restore=args.restore,
        reload_every=int(args.reload_every),
        log_every=int(args.log_every),
        ckpt_every=int(args.ckpt_every),
        episodes=int(args.episodes),
        save_thresh=float(args.save_thresh), 
        final_exp_ep=int(args.final_exp_ep),
        prio=bool(args.prio),
        ini_eps=float(args.ini_eps),
        final_eps=float(args.final_eps),
        lrate=float(5.0e-4),
        replay_start_size=int(args.replay_start_size)
    )


