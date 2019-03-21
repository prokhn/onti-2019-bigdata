import os
import sys
import time
import json
import random
import logging
import unittest
import threading
import traceback
import subprocess
import numpy as np
from pprint import pprint
from multiprocessing import Pool


class GymFCComparator:
    def __init__(self):
        pass

    def run_sh(self, args: tuple):
        command = args[0]
        if args.__len__() == 2:
            thread_name = args[1]
        else:
            thread_name = 'Thread #0'

        print('[{}] >>> user@pc:\033[0m \033[34m{}\033[0m'.format(thread_name, command))

        try:
            output = subprocess.check_output(command, shell=True).decode().strip()
            return output

        except (BaseException, Exception):
            return traceback.format_exc()


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.abspath(__file__))

    log_filename = "logs/run_{}.log".format(time.strftime('%H_%M_%S'))
    log_filename = os.path.join(dir_path, log_filename)

    # format='[LINE:%(lineno)d][%(asctime)s]  %(message)s',
    logging.basicConfig(filename=log_filename, filemode='w',
                        format='%(message)s',
                        level=logging.INFO)
    logging.info('Logging initialized in file "{}"'.format(log_filename))

    comparator = GymFCComparator()

    PROCESSES = 3
    seeds = [random.randint(1, 100000) for _ in range(PROCESSES)]
    # seeds = [15, 17, 72132]
    ticks = -1

    py_group_1 = 'run_pid_optimized.py'
    py_group_2 = 'pid_lenya.py'

    commands_group_1 = ['python3 {} --ticks {} --seed {}'.format(py_group_1, ticks, seeds[i])
                        for i in range(PROCESSES)]

    commands_group_2 = ['python3 {} --ticks {} --seed {}'.format(py_group_2, ticks, seeds[i])
                        for i in range(PROCESSES)]

    commands_all = commands_group_1 + commands_group_2

    pool_args = []
    for ind in range(2 * PROCESSES):
        pool_args.append((commands_all[ind], 'Thread #{}'.format(ind),))

    gr1_results_mean = []
    gr1_results_summ = []

    gr2_results_mean = []
    gr2_results_summ = []

    with Pool(2 * PROCESSES) as pool:
        logging.info('Pool initialized')
        stdout = pool.map(comparator.run_sh, pool_args)
        logging.info('Pool closed')

    logging.info('=' * 60)
    for thread_id in range(2 * PROCESSES):
        summary = stdout[thread_id][stdout[thread_id].index('Results summary'):]
        summary = summary.replace('\t', '    ')
        value_sum = float(summary.split('\n')[1].split()[-1])
        value_mean = float(summary.split('\n')[2].split()[-1])

        if thread_id < PROCESSES:
            msg = 'Group #1 {} results at seed {}'.format(py_group_1, seeds[thread_id % PROCESSES]) \
                  + '\n' + '[Thread #{}] {}'.format(thread_id, '=' * 48) \
                  + '\n' + summary + '\n' + '=' * 60
            logging.info(msg)
            gr1_results_summ.append(value_sum)
            gr1_results_mean.append(value_mean)
        else:
            msg = 'Group #2 {} results at seed {}'.format(py_group_2, seeds[thread_id % PROCESSES]) \
                  + '\n' + '[Thread #{}] {}'.format(thread_id, '=' * 48) \
                  + '\n' + summary + '\n' + '=' * 60
            logging.info(msg)
            gr2_results_summ.append(value_sum)
            gr2_results_mean.append(value_mean)

    gr1_results_mean = np.sum(np.array(gr1_results_mean))
    gr2_results_mean = np.sum(np.array(gr2_results_mean))
    gr1_results_summ = np.sum(np.array(gr1_results_summ))
    gr2_results_summ = np.sum(np.array(gr2_results_summ))

    if abs(gr1_results_summ - gr2_results_summ) < 10:
        win_msg = 'Results approximately same'
    elif gr1_results_summ - gr2_results_summ > 0:
        win_msg = 'Group #1 {} won'.format(py_group_1)
    else:
        win_msg = 'Group #2 {} won'.format(py_group_2)

    logging.info('== Comparison results ({} - {}):'.format(py_group_1, py_group_2))
    # logging.info('     Results  mean delta: {}'.format(gr1_results_mean - gr2_results_mean))
    logging.info('        Group #1 {} summ: {}'.format(py_group_1, gr1_results_summ))
    logging.info('        Group #2 {} summ: {}'.format(py_group_2, gr2_results_summ))
    logging.info('        Results summ delta: {}'.format(gr1_results_summ - gr2_results_summ))
    logging.info('=' * 60)
    logging.info('== {}'.format(win_msg))
    logging.info('=' * 60)
