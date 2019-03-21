import os
import gymfc
import gym
from tensorflow.python.keras.layers import Input, Dense, Activation
from drlbox.trainer import make_trainer
# import nest_asyncio
# nest_asyncio.apply()

class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale

    def reward(self, r):
        return r * self.scale
    
'''
Input arguments:
    observation_space: Observation space of the environment;
    num_hid_list:      List of hidden unit numbers in the fully-connected net.
'''
def make_feature(observation_space, num_hid_list):
    inp_state = Input(shape=observation_space.shape)
    feature = inp_state
    for num_hid in num_hid_list:
        feature = Dense(num_hid)(feature)
        feature = Activation('relu')(feature)
    return inp_state, feature


if __name__ == '__main__':
    current_dir = os.getcwd()
    config_path = os.path.join(current_dir, "../configs/iris.config")
    os.environ["GYMFC_CONFIG"] = config_path
    env = gym.make('AttFC_GyroErr-MotorVel_M4_Con-v0')
#     env = RewScale(env, 0.1)
    trainer = make_trainer(
        algorithm='a3c',
        env_maker=lambda: env,
        feature_maker=lambda obs_space: make_feature(obs_space, [64, 128, 64]),
        num_parallel=5,
        train_steps=1000,
        verbose=True,
        )
    trainer.run()
