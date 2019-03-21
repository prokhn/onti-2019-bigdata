import json
import os
import sys
import time
import random
import unittest
import threading
import traceback
import subprocess
from pprint import pprint
from multiprocessing import Pool


class GymfcChecker:
    def run_sh(self, args: tuple):
        command = args[0]
        if args.__len__() == 2:
            thread_name = args[1]
        else:
            thread_name = 'Thread #0'

        print('[{}] >>> user@pc:\033[0m \033[34m{}\033[0m'.format(thread_name, command))

        sh_kernel = {'busy': True}

        # thread = threading.Thread(target=self.time_handler, args=[sh_kernel])
        # thread.start()

        try:
            output = subprocess.check_output(command, shell=True).decode().strip()
            sh_kernel['busy'] = False
            time.sleep(0.2)
            return output

        except (BaseException, Exception):
            sh_kernel['busy'] = False
            time.sleep(0.2)
            return traceback.format_exc()


if __name__ == '__main__':
    checker = GymfcChecker()

    PROCESSES = 3
    PRINT_ALL_STDOUT = False

    # seeds = [random.randint(1, 100000) for _ in range(PROCESSES)]
    seeds = [15, 17, 72132]
    ticks = 10000
    commands = ['python3 pid_lenya.py --ticks {} --seed {}'.format(ticks, seeds[i])
                for i in range(PROCESSES)]

    pool_args = []
    for ind in range(PROCESSES):
        pool_args.append((commands[ind], 'Thread #{}'.format(ind),))

    results_mean = []
    results_summ = []

    with Pool(PROCESSES) as pool:
        stdout = pool.map(checker.run_sh, pool_args)
        print('\n[EVL] ' + '\033[93m' + 'Evaluating finished' + '\033[0m')
        for thread_id in range(PROCESSES):
            if PRINT_ALL_STDOUT:
                print('\n\n')
                print('=' * 75)
                print('===== Thread #{} output start '.format(thread_id))
                print('=' * 75)
                print(stdout[thread_id], end='\033[0m\n')
                print('=' * 75)
                print('===== Thread #{} output end   '.format(thread_id))
                print('=' * 75)
            else:
                summary = stdout[thread_id][stdout[thread_id].index('Results summary'):]
                value_sum = float(summary.split('\n')[1].split()[-1])
                value_mean = float(summary.split('\n')[2].split()[-1])
                print('[Thread #{}] ======================='.format(thread_id))
                print(summary)
                results_summ.append(value_sum)
                results_mean.append(value_mean)
                print('=' * 35)

    print()
    print('=' * 35)

    sum_mean = round(sum(results_summ) / 3, 6)
    mean_mean = round(sum(results_mean) / 3, 6)

    print('Total of {} processes:\n\t'
          'Mean of three summ {}\n\t'
          'Mean of three mean {}'.format(PROCESSES, sum_mean, mean_mean))

    print('=' * 35)
