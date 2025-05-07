import numpy as np
import matplotlib.pyplot as plt
import random

class Solver:
    """
    MAB识别算法基本框架, 基类
    """

    def __init__(self, bandit, env=None, **kwargs):
        """
        :param bandit: MAB模型对象
        :param env: 环境对象
        :param kwargs: 其他参数
        """
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 记录每个臂被拉动的次数
        self.regret = 0. # 记录累积遗憾值
        self.actions = []  # 维护一个列表, 记录每一步的动作选择
        self.regrets = []  # 维护一个列表, 记录每一步的累计regret
        self.rewards = [[] for _ in range(self.bandit.K)]  # 维护一个列表, 记录每一步的奖励以及归属的臂
        self.env = env  # 当前状态，历史累计连输
        self.kwargs = kwargs

    # def update_regret(self, k):
    #     """
    #     更新累计遗憾值和每一步的累计regret列表
    #     :param k: 被拉动的臂
    #     :return: None
    #     """
    #     # self.regret += self.bandit.get_optimal_reward() - self.bandit.get_reward(k)  # 这个是实际观测的，好像不是mu?
    #     self.regret += self.bandit.get_optimal_mu() - self.bandit.get_mu_p()[k]  # 这个是给定的mu        
    #     self.regrets.append(self.regret)

    # def run_one_step(self):
    #     # 每个子算法的class具体实现
    #     raise NotImplementedError("This method should be overridden by subclasses.")
    
    def take_action(self, **kwargs):
        # 采样action的方法
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def update(self, **kwargs):
        # 更新Q V 值的方法
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    # def run(self, num_steps):
    #     """
    #     单条path, 运行一定次数
    #     :param num_steps: 单条path运行的步数
    #     :return: None
    #     """
    #     for step in range(num_steps):
    #         action = self.run_one_step()  # 选择动作手臂
    #         self.counts[action] += 1  # 更新被拉动的臂的次数
    #         self.update_regret(action)  # 更新遗憾值 在给定\mu的情形下
    #         self.actions.append(action)