import argparse
import gym
import gymfc
import numpy as np
import tqdm
from mpi4py import MPI
import math
import os
import time


class Policy(object):
    def action(self, state, sim_time=0, desired=np.zeros(3), actual=np.zeros(3)):
        pass

    def reset(self):
        pass


class Agent(Policy):
    def __init__(self):
        self.r = [39, 6, 0]
        self.p = [35, 8, 0]
        self.y = [4, 35, 0]

        # self.r = [23, 40, 7]
        # self.p = [33, 30, 75]
        # self.y = [49, 50, 2]
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

    def __init__(self, pid_roll=[40, 40, 30], pid_pitch=[58, 50, 35], pid_yaw=[80, 45, 20], itermLimit=150):

        # init gains and scale
        self.Kp = [pid_roll[0], pid_pitch[0], pid_yaw[0]]
        self.Kp = [self.PTERM_SCALE * p for p in self.Kp]

        self.Ki = [pid_roll[1], pid_pitch[1], pid_yaw[1]]
        self.Ki = [self.ITERM_SCALE * i for i in self.Ki]

        self.Kd = [pid_roll[2], pid_pitch[2], pid_yaw[2]]
        self.Kd = [self.DTERM_SCALE * d for d in self.Kd]

        self.itermLimit = itermLimit

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
        PID_MIXER_SCALING = 1000.0
        pidSumLimit = 10000.  # 500
        pidSumLimitYaw = 100000.  # 1000.0#400
        motorOutputMixSign = 1
        motorOutputRange = self.maxthrottle - self.minthrottle  # throttle max - throttle min
        motorOutputMin = self.minthrottle

        currentMixer = [
            [1.0, -1.0, 0.598, -1.0],  # REAR_R
            [1.0, -0.927, -0.598, 1.0],  # RONT_R
            [1.0, 1.0, 0.598, 1.0],  # REAR_L
            [1.0, 0.927, -0.598, -1.0],  # RONT_L
        ]
        mixer_index_throttle = 0
        mixer_index_roll = 1
        mixer_index_pitch = 2
        mixer_index_yaw = 3

        scaledAxisPidRoll = self.constrainf(r, -pidSumLimit, pidSumLimit) / PID_MIXER_SCALING
        scaledAxisPidPitch = self.constrainf(p, -pidSumLimit, pidSumLimit) / PID_MIXER_SCALING
        scaledAxisPidYaw = self.constrainf(y, -pidSumLimitYaw, pidSumLimitYaw) / PID_MIXER_SCALING
        scaledAxisPidYaw = -scaledAxisPidYaw

        # Find roll/pitch/yaw desired output
        motor_count = 4
        motorMix = [0] * motor_count
        motorMixMax = 0
        motorMixMin = 0
        # No additional throttle, in air mode
        throttle = 0
        motorRangeMin = 1000
        motorRangeMax = 2000

        for i in range(motor_count):
            mix = (scaledAxisPidRoll * currentMixer[i][1] +
                   scaledAxisPidPitch * currentMixer[i][2] +
                   scaledAxisPidYaw * currentMixer[i][3])

            if mix > motorMixMax:
                motorMixMax = mix
            elif mix < motorMixMin:
                motorMixMin = mix
            motorMix[i] = mix

        motorMixRange = motorMixMax - motorMixMin
        # print("range=", motorMixRange)

        if motorMixRange > 1.0:
            for i in range(motor_count):
                motorMix[i] /= motorMixRange
            # Get the maximum correction by setting offset to center when airmode enabled
            throttle = 0.5

        else:
            # Only automatically adjust throttle when airmode enabled. Airmode logic is always active on high throttle
            throttleLimitOffset = motorMixRange / 2.0
            throttle = self.constrainf(throttle, 0.0 + throttleLimitOffset, 1.0 - throttleLimitOffset)

        motor = []
        for i in range(motor_count):
            motorOutput = motorOutputMin + (motorOutputRange * (
                    motorOutputMixSign * motorMix[i] + throttle * currentMixer[i][mixer_index_throttle]))
            motorOutput = self.constrainf(motorOutput, motorRangeMin, motorRangeMax);
            motor.append(motorOutput)

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

    def __init__(self, P=1.2, I=1, D=0.001):

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

        delta_time = (current_time - self.last_time) / current_time
        delta_error = (error - self.last_error) / error

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

    def run(self, env, pi, ticks_count: int):
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

            if ticks_count != -1:
                if pbar.n >= ticks_count:
                    break

            actuals.append(actual)
            desireds.append(desired)
            rewards.append(reward)

            if done:
                break

        pbar.close()
        env.close()
        return desireds, actuals, rewards

    def main(self, env_id: str, seed: int, r=None, p=None, y=None, ticks_count=-1):
        if y is None:
            y = [4, 50, 0.0]
        if p is None:
            p = [10, 10, 0.005]
        if r is None:
            r = [2, 10, 0.005]
        print('[Evaluator.main] Starting new session with seed {}'.format(seed))
        print('[Evaluator.main] Environment id is "{}"'.format(env_id))

        env = gym.make(env_id)
        rank = MPI.COMM_WORLD.Get_rank()
        workerseed = seed + 1000000 * rank
        env.seed(workerseed)
        pi = Agent()
        desireds, actuals, rewards = self.run(env, pi, ticks_count)
        rewards = np.array(rewards)
        print('\nResults summary:\n\t--sum  {}\n\t--mean {}'.format(np.sum(rewards), np.mean(rewards)))
        title = "PID Step Response in Environment {}".format(env_id)
        plot_title = 'Session seed {}'.format(seed)
        # self.plot_step_response(plot_title, np.array(desireds), np.array(actuals), title=title)
        return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate a PID controller")
    parser.add_argument('--env-id', help="The Gym environement ID", type=str,
                        default="AttFC_GyroErr-MotorVel_M4_Con-v0")
    parser.add_argument('--seed', help='RNG seed', type=int, default=9832)
    parser.add_argument('--ticks', help='Ticks count', type=int, default=-1)

    args = parser.parse_args()
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, "../configs/iris.config")

    print("Loading config from ", config_path)
    os.environ["GYMFC_CONFIG"] = config_path

    evaluator = PIDEvaluator()
    evaluator.main(args.env_id, args.seed, ticks_count=args.ticks)
