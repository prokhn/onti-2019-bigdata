import os
import numpy as np
import argparse
import gym
import tqdm
from keras.models import load_model

from pid_lenya import Agent
from run_pid_optimized import PIDPolicy


model = load_model('model.hd5')
def main():
    parser = argparse.ArgumentParser()
    env = 'AttFC_GyroErr-MotorVel_M4_Ep-v0'
    seeds = [5,]
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir,
                               "../configs/iris.config")
    print("Loading config from ", config_path)
    os.environ["GYMFC_CONFIG"] = config_path
    print(" Making env=", env)
    sum_reward = 0
    env = gym.make(env)

    agent = PIDPolicy()  # AGENT
    for seed in seeds:
        agent.reset()

        env.seed(seed)
        ob = env.reset()

        pbar = tqdm.tqdm(total=60000)
        iter_reward = 0
        rewards_mean = []
        nn_cycles = 2
        cycles_gone = 0
        while True:
            desired = env.omega_target
            actual = env.omega_actual
            ac = agent.action(ob, env.sim_time, desired, actual)
            prediction = model.predict(np.array([desired - actual, ]))[0]
            print('delta:',desired-actual)

#             print(prediction, prediction.shape)
#             ob, reward, done, info = env.step(ac)
            ob, reward, done, info = env.step(prediction)
            print('reward:',reward)
            iter_reward += reward

            rewards_mean.append(abs(reward))
            if rewards_mean.__len__() == 10000:
                _mean = sum(rewards_mean) / 10000
                print(' --- Mean reward for 10k iters: {}'.format(_mean))
                rewards_mean.clear()

            pbar.update(1)

            if done:
                break

        print('ITERATION {} RESULT: {}'.format(seeds.index(seed), iter_reward))
        pbar.close()
    print(sum_reward)


if __name__ == '__main__':
    main()
