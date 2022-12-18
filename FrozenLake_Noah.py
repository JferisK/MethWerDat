# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 18:13:01 2022

@author: Noah
"""

#from IPython.display import HTML
#HTML('')



import numpy as np
import gym
import random
import time

env = gym.make("FrozenLake-v1",map_name="8x8")
action_size = env.action_space.n
state_size = env.observation_space.n


qtable = np.zeros((state_size, action_size))
#print(qtable)

total_episodes = 30000        # Total episodes
learning_rate = 0.65           # Learning rate
max_steps = 500                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.0001             # Exponential decay rate for exploration prob

# List of rewards
rewards = []

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()[0]
    step = 0
    done = False
    total_rewards = 0
    
    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)
        
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, truncated,info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state,:] : all the actions we can take from new state
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state,:]) - qtable[state, action])
        
        total_rewards += reward
        
        # Our new state is state
        state = new_state
        
        # If done (if we're dead) : finish episode
        if done == True: 
            break
        
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    rewards.append(total_rewards)

print ("Score over time: " +  str(sum(rewards)/total_episodes))
print(qtable)

#Play with model
#env = gym.make("FrozenLake-v1",render_mode="human",map_name="8x8")
env.reset()

result =0
for episode in range(1000):
    state = env.reset()[0]
    step = 0
    done = False
    #print("****************************************************")
    #print("EPISODE ", episode+1)

    for step in range(max_steps):
        
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state,:])
        
        new_state, reward, done, truncated, info = env.step(action)
        
        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            env.render()
            
            result += reward
            # We print the number of step it took.
            #print("Number of steps", step)
            #print("Reward",reward)
            break
        state = new_state

print("Completionrate",result/1000)
env.close()