import os
from baselines.common import tf_util as U
from baselines import logger
import gymfc
import argparse
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gym
import math


def train(num_timesteps, seed, model_path=None, env_id=None):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2)

    env = gym.make(env_id)

    # parameters below were the best found in a simple random search
    # these are good enough to make humanoid walk, but whether those are
    # an absolute best or not is not certain
    env = RewScale(env, 0.1)
    pi = pposgd_simple.learn(env, policy_fn,
                             max_timesteps=num_timesteps,
                             timesteps_per_actorbatch=2048,
                             clip_param=0.2, entcoeff=0.0,
                             optim_epochs=10,
                             optim_stepsize=3e-4,
                             optim_batchsize=64,
                             gamma=0.99,
                             lam=0.95,
                             schedule='linear',
                             )
    env.close()
    if model_path:
        U.save_state(model_path)

    return pi


class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale

    def reward(self, r):
        return r * self.scale


def plot_step_response(desired, actual,
                       end=1., title=None,
                       step_size=0.001, threshold_percent=0.1):
    """
        Args:
            threshold (float): Percent of the start error
    """

    # actual = actual[:,:end,:]
    end_time = len(desired) * step_size
    t = np.arange(0, end_time, step_size)

    # desired = desired[:end]
    threshold = threshold_percent * desired

    plot_min = -math.radians(350)
    plot_max = math.radians(350)

    subplot_index = 3
    num_subplots = 3

    f, ax = plt.subplots(num_subplots, sharex=True, sharey=False)
    f.set_size_inches(10, 5)
    if title:
        plt.suptitle(title)
    ax[0].set_xlim([0, end_time])
    res_linewidth = 2
    linestyles = ["c", "m", "b", "g"]
    reflinestyle = "k--"
    error_linestyle = "r--"

    # Always
    ax[0].set_ylabel("Roll (rad/s)")
    ax[1].set_ylabel("Pitch (rad/s)")
    ax[2].set_ylabel("Yaw (rad/s)")

    ax[-1].set_xlabel("Time (s)")

    """ ROLL """
    # Highlight the starting x axis
    ax[0].axhline(0, color="#AAAAAA")
    ax[0].plot(t, desired[:, 0], reflinestyle)
    ax[0].plot(t, desired[:, 0] - threshold[:, 0], error_linestyle, alpha=0.5)
    ax[0].plot(t, desired[:, 0] + threshold[:, 0], error_linestyle, alpha=0.5)

    r = actual[:, 0]
    ax[0].plot(t[:len(r)], r, linewidth=res_linewidth)

    ax[0].grid(True)

    """ PITCH """

    ax[1].axhline(0, color="#AAAAAA")
    ax[1].plot(t, desired[:, 1], reflinestyle)
    ax[1].plot(t, desired[:, 1] - threshold[:, 1], error_linestyle, alpha=0.5)
    ax[1].plot(t, desired[:, 1] + threshold[:, 1], error_linestyle, alpha=0.5)
    p = actual[:, 1]
    ax[1].plot(t[:len(p)], p, linewidth=res_linewidth)
    ax[1].grid(True)

    """ YAW """
    ax[2].axhline(0, color="#AAAAAA")
    ax[2].plot(t, desired[:, 2], reflinestyle)
    ax[2].plot(t, desired[:, 2] - threshold[:, 2], error_linestyle, alpha=0.5)
    ax[2].plot(t, desired[:, 2] + threshold[:, 2], error_linestyle, alpha=0.5)
    y = actual[:, 2]
    ax[2].plot(t[:len(y)], y, linewidth=res_linewidth)
    ax[2].grid(True)

    plt.savefig("gymfc-ppo-step-response.pdf")


def main():
    parser = argparse.ArgumentParser()
    logger.configure()
    parser.add_argument('--env', type=str, help="The Gym environement ID",
                        default="AttFC_GyroErr-MotorVel_M4_Con-v0")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--model-path', default=os.path.join('/root/code/nti/gymfc/humanoid_policy', 'hum'))
    parser.add_argument('--play', action="store_true", default=False)
    parser.add_argument('--num-timesteps', type=int, default=2*1e6)
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir,
                               "../configs/iris.config")
    print("Loading config from ", config_path)
    os.environ["GYMFC_CONFIG"] = config_path
    args = parser.parse_args()

    if not args.play:
        # train the model
        train(num_timesteps=args.num_timesteps, seed=args.seed, model_path=args.model_path, env_id=args.env)
    else:
        print(" Making env=", args.env)
        # construct the model object, load pre-trained model and render
        pi = train(num_timesteps=1, seed=args.seed, env_id=args.env)
        U.load_state(args.model_path)

        env = gym.make(args.env)
        #env.render()
        ob = env.reset()
        actuals = []
        desireds = []
        while True:
            desired = env.omega_target
            actual = env.omega_actual
            actuals.append(actual)
            desireds.append(desired)
            print("sp=", desired, " rate=", actual)
            action = pi.act(stochastic=False, ob=ob)[0]
            ob, _, done, _ = env.step(action)
            if done:
                break
        print(np.array(desireds))
        print(np.array(actuals))
        plot_step_response(np.array(desireds), np.array(actuals))


if __name__ == '__main__':
    main()
