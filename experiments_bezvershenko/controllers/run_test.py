import os
import gymfc
import argparse
import gym
import numpy as np
from pid import Agent
import math
import matplotlib.pyplot as plt

def evaluate(agent, env, sum_reward=0, seed=17, verbose=False):
    agent.reset()
    env.seed(seed)
    ob = env.reset()
        
        
    ticks = 0
    while True:
        desired = env.omega_target
        actual = env.omega_actual
            
            
            
        action = agent.action(ob, env.sim_time, desired, actual)
            
        ob, reward, done, _ = env.step(action)
        sum_reward += reward
        ticks += 1
         
       
        if ticks % 1000 and verbose:
            print('\rTicks: {} Reward: {}'.format(ticks, sum_reward), end='', flush=True)
                
        if done:
            return sum_reward
       
 

def main():
    parser = argparse.ArgumentParser()
    env = 'AttFC_GyroErr-MotorVel_M4_Con-v0'
    seeds = [5, 8, 239]
    
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir,
                               "../configs/iris.config")
    print("Loading config from ", config_path)
    
    os.environ["GYMFC_CONFIG"] = config_path
    print(" Making env=", env)

    sum_reward = 0
    env = gym.make(env)
    agent = Agent()
    ob = env.reset()
    agent.fit(env)
   
 
    for seed in seeds:
        sum_reward = evaluate(agent, env, sum_reward, seed, verbose=True)
        
    print()
    print(sum_reward)


if __name__ == '__main__':
    main()
