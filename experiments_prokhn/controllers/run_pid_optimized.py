import os
import gym
import math
import time
import tqdm
import gymfc
import argparse
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt


class Policy(object):
    def action(self, state, sim_time=0, desired=np.zeros(3), actual=np.zeros(3)):
        pass

    def reset(self):
        pass


class PIDPolicy(Policy):
    def __init__(self, r, p, y):
        # self.r = [2, 10, 0.005]
        # self.p = [10, 10, 0.005]
        # self.y = [4, 50, 0.0]

        self.r = r
        self.p = p
        self.y = y

        # print('!!!!!!!', self.r, self.p, self.y)
        self.controller = PIDController(pid_roll=self.r, pid_pitch=self.p, pid_yaw=self.y)

    def action(self, state, sim_time=0, desired=np.zeros(3), actual=np.zeros(3)):
        # Convert to degrees
        desired = list(map(math.degrees, desired))
        actual = list(map(math.degrees, actual))
        motor_values = np.array(self.controller.calculate_motor_values(sim_time, desired, actual))
        # Need to scale from 1000-2000 to -1:1
        return np.array([(m - 1000) / 500 - 1 for m in motor_values])

    def reset(self):
        self.controller = PIDController(pid_roll=self.r, pid_pitch=self.p, pid_yaw=self.y)


class PIDController(object):
    FD_ROLL = 0
    FD_PITCH = 1
    FD_YAW = 2
    PTERM_SCALE = 0.032029
    ITERM_SCALE = 0.244381
    DTERM_SCALE = 0.000529
    minthrottle = 1070
    maxthrottle = 2000

    def __init__(self, pid_roll=[40, 40, 30], pid_pitch=[58, 50, 35], pid_yaw=[80, 45, 20], iterm_limit=150):

        # init gains and scale
        self.Kp = [pid_roll[0], pid_pitch[0], pid_yaw[0]]
        self.Kp = [self.PTERM_SCALE * p for p in self.Kp]

        self.Ki = [pid_roll[1], pid_pitch[1], pid_yaw[1]]
        self.Ki = [self.ITERM_SCALE * i for i in self.Ki]

        self.Kd = [pid_roll[2], pid_pitch[2], pid_yaw[2]]
        self.Kd = [self.DTERM_SCALE * d for d in self.Kd]

        self.itermLimit = iterm_limit

        self.previousRateError = [0] * 3
        self.previousTime = 0
        self.previous_motor_values = [self.minthrottle] * 4
        self.pid_rpy = [PID(*pid_roll), PID(*pid_pitch), PID(*pid_yaw)]

    def calculate_motor_values(self, current_time, sp_rates, gyro_rates):
        rpy_sums = []
        for i in range(3):
            self.pid_rpy[i].SetPoint = sp_rates[i]
            self.pid_rpy[i].update(current_time, gyro_rates[i])
            rpy_sums.append(self.pid_rpy[i].output)
        return self.mix(*rpy_sums)

    def constrainf(self, amt, low, high):
        # From BF src/main/common/maths.h
        if amt < low:
            return low
        elif amt > high:
            return high
        else:
            return amt

    def mix(self, r, p, y):
        pid_mixer_scaling = 1000.0
        pid_sum_limit = 10000.  # 500
        pid_sum_limit_yaw = 100000.  # 1000.0#400
        motor_output_mix_sign = 1
        motor_output_range = self.maxthrottle - self.minthrottle  # throttle max - throttle min
        motor_output_min = self.minthrottle

        current_mixer = [
            [1.0, -1.0, 0.598, -1.0],  # REAR_R
            [1.0, -0.927, -0.598, 1.0],  # RONT_R
            [1.0, 1.0, 0.598, 1.0],  # REAR_L
            [1.0, 0.927, -0.598, -1.0],  # RONT_L
        ]
        mixer_index_throttle = 0
        mixer_index_roll = 1
        mixer_index_pitch = 2
        mixer_index_yaw = 3

        scaled_axis_pid_roll = self.constrainf(r, -pid_sum_limit, pid_sum_limit) / pid_mixer_scaling
        scaled_axis_pid_pitch = self.constrainf(p, -pid_sum_limit, pid_sum_limit) / pid_mixer_scaling
        scaled_axis_pid_yaw = self.constrainf(y, -pid_sum_limit_yaw, pid_sum_limit_yaw) / pid_mixer_scaling
        scaled_axis_pid_yaw = -scaled_axis_pid_yaw

        # Find roll/pitch/yaw desired output
        motor_count = 4
        motor_mix = [0] * motor_count
        motor_mix_max = 0
        motor_mix_min = 0
        # No additional throttle, in air mode
        throttle = 0
        motor_range_min = 1000
        motor_range_max = 2000

        for i in range(motor_count):
            mix = (scaled_axis_pid_roll * current_mixer[i][1] +
                   scaled_axis_pid_pitch * current_mixer[i][2] +
                   scaled_axis_pid_yaw * current_mixer[i][3])

            if mix > motor_mix_max:
                motor_mix_max = mix
            elif mix < motor_mix_min:
                motor_mix_min = mix
            motor_mix[i] = mix

        motor_mix_range = motor_mix_max - motor_mix_min
        # print("range=", motor_mix_range)

        if motor_mix_range > 1.0:
            for i in range(motor_count):
                motor_mix[i] /= motor_mix_range
            # Get the maximum correction by setting offset to center when airmode enabled
            throttle = 0.5

        else:
            # Only automatically adjust throttle when airmode enabled. Airmode logic is always active on high throttle
            throttle_limit_offset = motor_mix_range / 2.0
            throttle = self.constrainf(throttle, 0.0 + throttle_limit_offset, 1.0 - throttle_limit_offset)

        motor = []
        for i in range(motor_count):
            motor_output = motor_output_min + (motor_output_range * (
                    motor_output_mix_sign * motor_mix[i] + throttle * current_mixer[i][mixer_index_throttle]))
            motor_output = self.constrainf(motor_output, motor_range_min, motor_range_max)
            motor.append(motor_output)

        motor = list(map(int, np.round(motor)))
        return motor

    def is_airmode_active(self):
        return True

    def reset(self):
        for pid in self.pid_rpy:
            pid.clear()


class PID:
    """PID Controller
    """

    def __init__(self, P=0.2, I=0.0, D=0.0):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = 0
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, current_time, feedback_value):
        """Calculates PID value for given reference feedback

        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}

        .. figure:: images/pid_1.png
           :align:   center

           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)

        """
        error = self.SetPoint - feedback_value

        delta_time = current_time - self.last_time
        delta_error = error - self.last_error

        if (delta_time >= self.sample_time):
            self.PTerm = self.Kp * error
            self.ITerm += error * delta_time

            if (self.ITerm < -self.windup_guard):
                self.ITerm = -self.windup_guard
            elif (self.ITerm > self.windup_guard):
                self.ITerm = self.windup_guard

            self.DTerm = 0.0
            if delta_time > 0:
                self.DTerm = delta_error / delta_time

            # Remember last time and last error for next calculation
            self.last_time = current_time
            self.last_error = error

            # print("P=", self.PTerm, " I=", self.ITerm, " D=", self.DTerm)
            self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time


class PIDEvaluator:
    def __init__(self):
        pass

    def run(self, env, pi):
        actuals = []
        desireds = []
        rewards = []

        pi.reset()
        ob = env.reset()

        print('PID Evaluation started')
        pbar = tqdm.tqdm(total=60000)
        while True:
            desired = env.omega_target
            actual = env.omega_actual
            # PID only needs to calculate error between desired and actual y_e
            ac = pi.action(ob, env.sim_time, desired, actual)
            ob, reward, done, info = env.step(ac)
            pbar.update(1)

            if pbar.n >= 1000:
                break

            actuals.append(actual)
            desireds.append(desired)
            rewards.append(reward)

            if done:
                break

        pbar.close()
        env.close()
        return desireds, actuals, rewards

    def main(self, env_id: str, seed: int, r, p, y):
        print('[Evaluator.main] Starting new session with seed {}'.format(seed))
        print('[Evaluator.main] Environment id is "{}"'.format(env_id))

        env = gym.make(env_id)
        rank = MPI.COMM_WORLD.Get_rank()
        workerseed = seed + 1000000 * rank
        env.seed(workerseed)
        pi = PIDPolicy(r, p, y)
        desireds, actuals, rewards = self.run(env, pi)
        rewards = np.array(rewards)
        print('\nResults summary:\n\t--sum  {}\n\t--mean {}'.format(np.sum(rewards), np.mean(rewards)))
        title = "PID Step Response in Environment {}".format(env_id)
        plot_title = 'Session seed {}'.format(seed)
        # self.plot_step_response(plot_title, np.array(desireds), np.array(actuals), title=title)
        return rewards

    def plot_step_response(self, plot_title, desired, actual, title=None, step_size=0.001, threshold_percent=0.1):
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

        plt.savefig('{}.png'.format(plot_title))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate a PID controller")
    parser.add_argument('--env-id', help="The Gym environement ID", type=str,
                        default="AttFC_GyroErr-MotorVel_M4_Con-v0")
    parser.add_argument('--seed', help='RNG seed', type=int, default=9832)

    args = parser.parse_args()
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, "../configs/iris.config")

    print("Loading config from ", config_path)
    os.environ["GYMFC_CONFIG"] = config_path

    evaluator = PIDEvaluator()
    evaluator.main(args.env_id, args.seed)
