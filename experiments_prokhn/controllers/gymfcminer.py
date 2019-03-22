import os
import time
import random
import logging
import traceback
import subprocess
from multiprocessing import Pool


class GymFCMiner:
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

    log_filename = "logs/miner_{}.log".format(time.strftime('%H_%M_%S'))
    log_filename = os.path.join(dir_path, log_filename)

    # format='[LINE:%(lineno)d][%(asctime)s]  %(message)s',
    logging.basicConfig(filename=log_filename, filemode='w',
                        format='%(message)s',
                        level=logging.INFO)
    logging.info('Logging initialized in file "{}"'.format(log_filename))

    dataminer = GymFCMiner()

    PROCESSES = 10
    SEEDS_IN_PROCCESSES = 5
    w_filenames = ['mined/thread{}_{}.csv'.format(thind, time.strftime('%H_%M_%S')) for thind in range(PROCESSES)]

    seeds = []
    while seeds.__len__() < PROCESSES:
        seed_start = random.randint(0, 10000000)
        seed_end = seed_start + SEEDS_IN_PROCCESSES

        if all([seed_start < seeds[i][0] or seed_start > seeds[i][1] for i in range(seeds.__len__())]):
            if all([seed_end < seeds[i][0] or seed_end > seeds[i][1] for i in range(seeds.__len__())]):
                seeds.append((seed_start, seed_end))

    commands = ['python3 dataminer.py --seed-from {} --seed-to {} --w-file {}'.format(
        seeds[i][0], seeds[i][1], w_filenames[i]) for i in range(PROCESSES)]

    pool_args = []
    for ind in range(PROCESSES):
        pool_args.append((commands[ind], 'Thread #{}'.format(ind),))

    with Pool(PROCESSES) as pool:
        logging.info('Pool initialized')
        stdout = pool.map(dataminer.run_sh, pool_args)
        logging.info('Pool closed')

    logging.info('Mining ended')
