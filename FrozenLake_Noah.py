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

def train(total_episodes,decay_rate,learning_rate,max_steps,gamma,min_epsilon=0.1,max_epsilon=1,epsilon=1):
    # List of rewards
    qtable = np.zeros((state_size, action_size))
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
    return(qtable)

#Play with model
#env = gym.make("FrozenLake-v1",render_mode="human",map_name="8x8")
def test(qtable,max_steps):
    env.reset()

    result = 0
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

    return(result/1000)


env = gym.make("FrozenLake-v1",map_name="8x8")
action_size = env.action_space.n
state_size = env.observation_space.n


total_episodes = [30000,25000,20000,15000,10000,5000]        # Total episodes
learning_rate = [0.65,0.8,0.9,0.1,0.5,0.7,0.3]           # Learning rate
max_steps = [100,200,300,400,500,600]                # Max steps per episode
gamma = [1,0.95,0.8,0.6,0.4,0.2,0.1,0.3]                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = [0.0001,0.0005,0.001,0.005,0.01]             # Exponential decay rate for exploration prob


test_results=[]
for i in range(200):
    params =[]
    episodes = random.choice(total_episodes)
    params.append(episodes)
    lr = random.choice(learning_rate)
    params.append(lr)
    steps = random.choice(max_steps)
    params.append(steps)
    g = random.choice(gamma)
    params.append(g)
    decay = random.choice(decay_rate)
    params.append(decay)

    qtable=train(episodes,decay,lr,steps,g)
    params.append(test(qtable,steps))
    test_results.append(params)

print(test_results)
env.close()