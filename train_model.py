# coding=utf-8
import argparse, os
import tensorflow as tf
from PIL import Image
import keras.backend as K
import numpy as np
from keras.optimizers import Adam
from keras.backend import set_session
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from ICMDP_Env import ClassifyEnv
from  get_model import get_pbcnn_model
from data_pre import load_data, get_imb_data
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
# parser.add_argument('--data',choices=['mnist', 'cifar10','famnist','imdb'], default='famnist')
# parser.add_argument('--model', choices=['image', 'text'], default='image')
# parser.add_argument('--imb-rate',type=float, default=0.05)
# parser.add_argument('--min-class', type=str, default='456')
# parser.add_argument('--maj-class', type=str, default='789')
parser.add_argument('--training-steps', type=int, default=983431)
args = parser.parse_args()

# train_path count: 1465359
# test_path count: 732689
# valid_path count: 244225
# train_us_os count: 983431



# TODO(zenglinfan) load dataset
x_train_path = 'D:\\ids2018\\tfrecord\\train_us_os\\features.npy'
y_train_path = 'D:\\ids2018\\tfrecord\\train_us_os\\labels.npy'
x_test_path = 'D:\\ids2018\\tfrecord\\test\\features.npy'
y_test_path = 'D:\\ids2018\\tfrecord\\test\\labels.npy'

x_train = np.memmap(x_train_path, dtype='float32', mode='r', shape=(983431, 20, 256, 1))
y_train = np.memmap(y_train_path, dtype='int64', mode='r', shape=(983431,))


print(f"x_train shape: {x_train.shape}")
print(f"y_train.shape: {y_train.shape}")

in_shape = x_train.shape[1:]
num_classes = 15
mode = 'train'
env = ClassifyEnv(mode, x_train, y_train)
nb_actions = num_classes
training_steps = args.training_steps
model = get_pbcnn_model()

INPUT_SHAPE = in_shape
print(model.summary())


class ClassifyProcessor(Processor):
    def process_observation(self, observation):
        # if args.model == 'text':
        #     return observation
        # img = observation.reshape(INPUT_SHAPE)
        # processed_observation = np.array(img)
        # return processed_observation
        return observation

    def process_state_batch(self, batch):
        # if args.model == 'text':
        #     return batch.reshape((-1, INPUT_SHAPE[1]))
        batch = batch.reshape((-1,) + INPUT_SHAPE)
        # processed_batch = batch.astype('float32') / 1.
        return batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


memory = SequentialMemory(limit=100000, window_length=1)
processor = ClassifyProcessor()
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=100000)
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, enable_double_dqn = True, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=0.5, target_model_update=10000,
               train_interval=4)
dqn.compile(Adam(learning_rate=.00025), metrics=['mae'])

dqn.fit(env, nb_steps=training_steps, log_interval=60000)


# env.mode = 'test'
# dqn.test(env, nb_episodes=1, visualize=False)
x_test = np.memmap(x_test_path, dtype='float32', mode='r', shape=(732689, 20, 256, 1))
y_test = np.memmap(y_test_path, dtype='int64', mode='r', shape=(732689,))
env = ClassifyEnv(mode, x_test, y_test)
env.mode = 'test'
dqn.test(env, nb_episodes=1, visualize=False)

