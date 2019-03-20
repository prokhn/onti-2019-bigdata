import os
import sys
import numpy as np
from run_pid_optimized import PIDEvaluator
from bayes_opt import BayesianOptimization
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events


def func(rx, ry, px, py, yx, yy):
    rz, pz, yz = 0.0001, 0.0001, 0.0001

    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, "../configs/iris.config")

    os.environ["GYMFC_CONFIG"] = config_path

    evaluator = PIDEvaluator()
    rewards = evaluator.main('AttFC_GyroErr-MotorVel_M4_Con-v0', 15, [rx, ry, rz], [px, py, pz], [yx, yy, yz])

    return rewards.sum()


# Bounded region of parameter space
# pbounds = {'rx': (0.0001, 50),
#            'ry': (0.0001, 50),
#            'rz': (0.0001, 50),
#            'px': (0.0001, 50),
#            'py': (0.0001, 50),
#            'pz': (0.0001, 50),
#            'yx': (0.0001, 50),
#            'yy': (0.0001, 50),
#            'yz': (0.0001, 50),
#            }
pbounds = {'rx': (1, 35),
           'ry': (1, 35),
           'px': (1, 35),
           'py': (1, 35),
           'yx': (1, 35),
           'yy': (1, 35),
           }

optimizer = BayesianOptimization(
    f=func,
    pbounds=pbounds,
    random_state=1,
)

logger = JSONLogger(path="./logs.json")
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

optimizer.maximize(
    init_points=25,
    n_iter=1000,
)

print(optimizer.max)
