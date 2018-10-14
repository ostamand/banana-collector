import numpy as np
import sys
import os

def save(agent, out_file, ep, it, avg_scores, scores, q_metrics, last_saved_score):
    log('Saving agent...')
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



