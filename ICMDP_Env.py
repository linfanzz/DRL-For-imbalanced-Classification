# encoding=utf-8
import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from imblearn.metrics import geometric_mean_score

class ClassifyEnv(gym.Env):

    def __init__(self, mode, trainx, trainy, ):  # mode means training or testing
        self.mode = mode

        self.Env_data = trainx
        self.Answer = trainy
        self.id = np.arange(trainx.shape[0])

        self.game_len = self.Env_data.shape[0]

        self.num_classes = len(set(self.Answer))
        self.action_space = spaces.Discrete(self.num_classes)
        self.observation_space = spaces.Box(low=0, high=1, shape=(20,256,1), dtype=np.float32)
        print(self.action_space)
        self.step_ind = 0
        self.y_pred = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # return: (states, observations)
    def reset(self):
        if self.mode == 'train':
            np.random.shuffle(self.id)
        self.step_ind = 0
        self.y_pred = []
        return self.Env_data[self.id[self.step_ind]]

    def step(self, a):
        self.y_pred.append(a)
        y_true_cur = []
        info = {}
        terminal = False
        if a == self.Answer[self.id[self.step_ind]]:
            reward = 1
        else:
            reward = -1
        self.step_ind += 1

        if self.step_ind == self.game_len - 1:
            y_true_cur = self.Answer[self.id]
            info['gmean'], info['fmeasure'] = self.My_metrics(np.array(self.y_pred),
                                                              np.array(y_true_cur[:self.step_ind]))
            terminal = True

        return self.Env_data[self.id[self.step_ind]], reward, terminal, info
    @staticmethod
    def My_metrics(y_pre, y_true):
        # TODO(zenglf): 计算多分类问题的G_mean, F1, precision, recall, TP, TN, FP, FN
        confusion_mtx = confusion_matrix(y_true, y_pre, labels=range(15))
        print('\n')
        print(classification_report(y_true=y_true, y_pred=y_pre, labels=range(15)))

        G_mean = geometric_mean_score(y_true, y_pre, average='macro')

        Recall, Precision, F_measure, _ = precision_recall_fscore_support(y_true, y_pre, average='macro')

        res = 'G-mean:{}, F_measure:{}\n'.format(G_mean, F_measure)
        print(res)
        print()
        return G_mean, F_measure



# test
if __name__=='__main__':
    y_true = [2, 0, 2, 2, 0, 1]
    y_pre = [0, 0, 2, 2, 0, 2]
    g_mean, f1 = ClassifyEnv.My_metrics(y_pre=y_pre, y_true=y_true)
    print(f"g_mean: {g_mean}")
    print(f"f1: {f1}")
