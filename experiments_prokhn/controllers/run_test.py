import os
import argparse
import gym
import tqdm
from pid_lenya import Agent
from run_pid_optimized import PIDPolicy


def evaluate_process(tick_: int,
                     mean_now: float,
                     rewards_mean: list,
                     iter_reward: float,
                     sum_reward: float = 0,
                     update_each=100,
                     is_training=False):
    if tick_ == 0 or rewards_mean.__len__() == update_each:
        mean_now = sum(rewards_mean) / update_each
        rewards_mean.clear()

    _its = str(float(abs(round(iter_reward - int(iter_reward), 2)))).split('.')[1]
    _its += '0' if _its.__len__() == 1 else ''
    str_iterr = '{}.{}'.format(str(int(iter_reward)).zfill(4), _its)

    _sms = str(float(abs(round(sum_reward - int(sum_reward), 2)))).split('.')[1]
    _sms += '0' if _sms.__len__() == 1 else ''
    str_total = '{}.{}'.format(str(int(sum_reward)).zfill(4), _sms)
    _mean = str(round(mean_now, 4))
    while _mean.__len__() != 6:
        _mean += '0'

    bar = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    for ind in range(bar.__len__()):
        bar_s = ind / bar.__len__()
        if tick_ / 60000 >= bar_s:
            bar[ind] = '#'

    if not is_training:
        print('\r\33[96mTick #{} {} | Mean reward now {} | Iteration reward {} | Summ reward {}\033[0m'.format(
            ''.join(bar),
            str(tick_).zfill(4),
            _mean,
            str_iterr,
            str_total), end='')
    else:
        print('\r\33[96mTick #{} {} | TRAINING IN PROCESS... \033[0m'.format(
            ''.join(bar),
            str(tick_).zfill(4)))

        return tick_ + 1, mean_now


def main():
    parser = argparse.ArgumentParser()
    env = 'AttFC_GyroErr-MotorVel_M4_Con-v0'
    seeds = [5, 8, 239]
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, "../configs/iris.config")
    # print("Loading config from ", config_path)
    os.environ["GYMFC_CONFIG"] = config_path
    sum_reward = 0
    env = gym.make(env)
    print(" Making env=", env)
    print('Something')

    print('\n\33[93m== EVALUATION STARTED ==\033[0m')
    agent = PIDPolicy()  # AGENT
    agent.fit(env, evaluate_process)

    for seed in seeds:
        agent.reset()

        env.seed(seed)
        ob = env.reset()

        iter_reward = 0
        tick_n = 0
        rewards_mean = []
        mean_now = 0.0

        while True:
            desired = env.omega_target
            actual = env.omega_actual

            ac = agent.action(ob, env.sim_time, desired, actual)

            ob, reward, done, info = env.step(ac)
            sum_reward += reward  # Reward total
            iter_reward += reward  # Reward for one iteration

            rewards_mean.append(abs(reward))
            tick_n, mean_now = evaluate_process(tick_n, mean_now, rewards_mean, iter_reward, sum_reward)

            if done:
                break

        print('\nITERATION {} | RESULT: {} | SUMM: {}'.format(seeds.index(seed), iter_reward, sum_reward))
    print(sum_reward)


if __name__ == '__main__':
    main()
