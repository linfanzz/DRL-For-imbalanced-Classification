# encoding=utf-8
import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from imblearn.metrics import geometric_mean_score
from get_model import _pkt_num, _pkt_bytes, _num_class
from data_pre import generate_ds



class ClassifyEnv(gym.Env):

    def __init__(self, mode, data_path):  # mode means training or testing
        self.mode = mode

        self.data_path = data_path
        self.ds = generate_ds(self.data_path)
        self.numpy_iter = self.ds.as_numpy_iterator()
        self.Answer = []

        self.num_classes = _num_class
        self.action_space = spaces.Discrete(self.num_classes)
        self.observation_space = spaces.Box(low=0, high=1, shape=(_pkt_num,_pkt_bytes,1), dtype=np.float32)
        print(self.action_space)
        print(self.observation_space)
        self.step_ind = 0
        self.y_pred = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_feature_label_done(self):
        # return feature, lable, done
        feature = self.next_feature
        label = self.next_label
        done = False
        try:
            numpy = self.numpy_iter.next()
            self.next_feature = numpy[0]
            self.next_feature = self.next_feature.reshape(_pkt_num, _pkt_bytes, 1)
            self.next_label = numpy[1]
        except StopIteration:
            done = True
        return feature, label, done
        
    
    # return: (states, observations)
    def reset(self):
        self.ds = generate_ds(self.data_path)
        self.numpy_iter = self.ds.as_numpy_iterator()
        # 因为迭代器特性，第一条数据需要特殊处理
        numpy = self.numpy_iter.next()
        self.next_feature = numpy[0]
        self.next_feature = self.next_feature.reshape(_pkt_num, _pkt_bytes, 1)
        self.next_label = numpy[1]
        # 第一条数据处理完毕

        feature, label, _ = self.get_feature_label_done()
        self.step_ind = 0
        self.Answer = []
        self.Answer.append(label)
        self.y_pred = []

        return feature

    def step(self, action):
        self.y_pred.append(action)
        info = {}
        feature, label, terminal = self.get_feature_label_done()
        self.Answer.append(label)

        if action == self.Answer[self.step_ind]:
            reward = 1
        else:
            reward = -1
        self.step_ind += 1

        if terminal:
            info['gmean'], info['fmeasure'] = self.My_metrics(np.array(self.y_pred),
                                                              np.array(self.Answer[:self.step_ind]))

        return feature, reward, terminal, info
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

    test_path='D:\\ids2018\\tfrecord\\test'
    env = ClassifyEnv('test', data_path=test_path)
    observation = env.reset()
    print(observation.shape)
    print(observation.dtype)