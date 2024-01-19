# encoding=utf-8
import numpy as np
import math
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
        if mode == 'train':
            self.get_equal_reward()
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
            reward = self.reward_set[self.Answer[self.id[self.step_ind]]]
        else:
            reward = -self.reward_set[self.Answer[self.id[self.step_ind]]]
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
        labels=['ftp-bruteforce', 'ddos-hoic', 'dos-goldeneye', 'ddos-loic-http', 'sql-injection','dos-hulk', 'bot', 'ssh-bruteforce', 'bruteforce-xss', 'dos-slowhttptest','bruteforce-web', 'dos-slowloris', 'benign', 'ddos-loic-udp', 'infiltration']
        print(classification_report(y_true=y_true, y_pred=y_pre, labels=range(15), target_names=labels, digits=4))

        G_mean = geometric_mean_score(y_true, y_pre, average='macro')

        Recall, Precision, F_measure, _ = precision_recall_fscore_support(y_true, y_pre, average='macro')

        res = 'Precision:{}, Recall:{}, G-mean:{}, F_measure:{}\n'.format(Precision, Recall, G_mean, F_measure)
        print(res)
        print()
        return G_mean, F_measure
    def get_l2_reward(self):
        _, nums_cls = np.unique(self.Answer, return_counts=True)
        print(f"nums_cls: {nums_cls}")
        raw_reward_set = 1 / nums_cls
        print(f"raw_reward: {raw_reward_set}")
        self.reward_set = np.round(raw_reward_set / np.linalg.norm(raw_reward_set), 6)
        print("\nReward for each class.")
        for cl_idx, cl_reward in enumerate(self.reward_set):
            print("\t- Class {} : {:.6f}".format(cl_idx, cl_reward))

    def get_equal_reward(self):
        self.reward_set = np.full(15, 1)
        print("\nReward for each class.")
        for cl_idx, cl_reward in enumerate(self.reward_set):
            print("\t- Class {} : {:.6f}".format(cl_idx, cl_reward))

    def get_ir_reward(self):
        _, nums_cls = np.unique(self.Answer, return_counts=True)
        min_class_nums = np.min(nums_cls)
        self.reward_set = np.full(15,0.1)
        for cl_idx, nums in enumerate(nums_cls):
            self.reward_set[cl_idx] = math.sqrt(min_class_nums / nums)
        for cl_idx, cl_reward in enumerate(self.reward_set):
            print("\t- Class {} : {:.6f}".format(cl_idx, cl_reward))

    def get_ir_min_reward(self):
        _, nums_cls = np.unique(self.Answer, return_counts=True)
        min_class_nums = np.min(nums_cls)
        self.reward_set = np.full(15,0.1)
        for cl_idx, nums in enumerate(nums_cls):
            self.reward_set[cl_idx] = max(math.sqrt(min_class_nums / nums), 1/15)
        for cl_idx, cl_reward in enumerate(self.reward_set):
            print("\t- Class {} : {:.6f}".format(cl_idx, cl_reward))


# test
if __name__=='__main__':
    x_test = np.load(r'D://ids2018//tfrecord//valid//features.npy')
    y_test = np.load(r'D://ids2018//tfrecord//valid//labels.npy')
    env = ClassifyEnv('train', x_test, y_test)

    # env.My_metrics([1,2,0,1,13,7],[1,2,0,0,12,5])
