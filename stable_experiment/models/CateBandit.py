import math
import numpy as np
import random

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from joblib import Parallel, delayed

# Categorical Reward Distribution（多点分布/Multinoulli Distribution） & Dynamic Futurity @ single arm ver.
class CateDynamicBandit:
    def __init__(self, J_distribution, df: int=None, num_steps=1000, seed=None, J_static=False):
        """
        初始化单臂老虎机模型, J值静态/动态版

        :param J_distribution: J值的分布。
                                对于离散分布，传入 ('discrete', [J_values], [probabilities])；
                                对于连续分布，传入 ('chisquare', df)
        :param df: 卡方分布的自由度（仅当 J_distribution 为 'chisquare' 时使用）
        :param num_steps: 每条轨迹的步数
        :param seed: 随机种子（可选）
        """
        self.J_distribution = J_distribution
        self.num_steps = num_steps
        self.df = df
        self.seed = seed
        self.J_change_cum = 0  # 记录 J 值变化的次数 
        self.J_static = J_static  # J是否静态

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

                # 预处理离散分布的累积分布函数 (CDF)
        if self.J_distribution[0] == 'discrete':
            J_values, probabilities = self.J_distribution[1], self.J_distribution[2]
            self.J_values = np.array(J_values)
            self.cumulative_probs = np.cumsum(probabilities)
            if not np.isclose(self.cumulative_probs[-1], 1.0):  # 验证输入的离散概率分布是正确的
                raise ValueError("Probabilities must sum to 1.")
            
        elif self.J_distribution[0] == 'chisquare':
            if not isinstance(self.J_distribution[1], (int, float)) or self.J_distribution[1] <= 0:
                raise ValueError("For 'chisquare' distribution, df must be a positive number.")
        
        ## TODO 待添加的其它离散/连续分布

        else:
            raise ValueError("Unsupported J_distribution type.")

        # 抽取初始的 J 并计算对应的 p
        self.J = self.sample_J()
        self.p = self.calculate_p(self.J)
        print(f"{self.J_distribution[0]} Distribution Initialized with J={self.J}, calculated p={self.p:.4f}")
        print(f"{self.J_distribution[0]} 分布 初始化 J={self.J}, 计算得到单局win rate p={self.p}")


    def sample_J(self):
        pass

    def calculate_p(self, J):
        pass

    def simulate_single_path(self, j_state_init=0):
        pass

    def simulate_multiple_paths(self, num_paths=1000, j_state_init=0, n_jobs=-1):
        pass

    def plot_paths_with_mean(self, rewards_array):
        pass

    def plot_paths_with_mean_init_stationarity(self, rewards_array, j_states: list):
        pass