import os
import gym
import gymfc


current_dir = os.path.dirname(__file__)
config_path = os.path.join(current_dir,
                           "../configs/iris.config")
print("Loading config from ", config_path)
os.environ["GYMFC_CONFIG"] = config_path

env = gym.make('AttFC_GyroErr-MotorVel_M4_Con-v0')
env = RewScale(env, 0.1)
env.reset()