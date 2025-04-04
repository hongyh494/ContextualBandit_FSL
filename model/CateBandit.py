import math
import numpy as np
import random

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from joblib import Parallel, delayed

class CategoricalDynamicBandit:
    """
    Category/Multi-point奖励分布的多臂老虎机模型, J值静态/动态版

    Params:
        K: 拉动的臂数
        J_distribution: J值的分布。
                            对于静态取值，传入 ('static', J_value)；
                            对于离散分布，传入 ('discrete', [J_values], [probabilities])；
                            对于连续分布，传入 ('chisquare', df)
        df: 卡方分布的自由度（仅当 J_distribution 为 'chisquare' 时使用）
        num_steps: 每条轨迹的步数
        seed: 随机种子（可选）

    """
        
    def __init__(self, K: int=10, 
            J_distribution: str = 'discrete', df: int = None, num_steps: int = 1000,
            seed=None):


        self.K = K  # 总臂数
        self.num_steps = num_steps  # 单个path的步数
        self.loss_state = 0  # 连续输的状态, 后面是否需要定义考虑？
        self.seed = seed
        self.J_distribution = J_distribution # J值静态/动态分布
        self.df = df
        self.J_change_cum = 0  # 记录 J 值变化的次数 
        self.J_discrete = None  # 是否是离散分布 调整reward的触发P有关
        self.J_active_times = 0  # futurity 触发的次数

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Initialize arms and their corresponding Bernoulli reward probabilities
        self.mu = 0.5 + np.array(range(1, K+1))/10  # Mean reward probabilities for each arm
        self.p = np.random.uniform(size=K)/5   # Actual probabilities for each arm


        # Dynamic J 要调整
        if self.J_distribution[0] != 'static':
            for i in range(K):
                self.J = self.sample_J()
                # adjust p 
                self.mu[i] = self.calculate_p(self.J)  # Calculate mean reward probability
                self.p[i] = self.mu[i]  # Initialize actual probabilities with mean values

        # Static J
        if self.J_distribution[0] == 'static':
            
            self.J = self.J_distribution[1]
            print(f"Static J Initialized with J={self.J}")

        # Dynamic J
        else:
            # 预处理J采样自离散分布的累积分布函数 (CDF)
            if self.J_distribution[0] == 'discrete':

                # 标识为离散分布
                self.J_discrete = True

                # 解包J的值分布，概率分布
                J_values, probabilities = self.J_distribution[1], self.J_distribution[2]
                self.J_values = np.array(J_values)
                self.cumulative_probs = np.cumsum(probabilities)
                if not np.isclose(self.cumulative_probs[-1], 1.0):  # 验证输入的离散概率分布是正确的
                    raise ValueError("Probabilities must sum to 1.")
                
            elif self.J_distribution[0] == 'chisquare':

                # 标识为连续分布
                self.J_discrete = False

                if not isinstance(self.J_distribution[1], (int, float)) or self.J_distribution[1] <= 0:
                    raise ValueError("For 'chisquare' distribution, df must be a positive number.")
            
            ## TODO 待添加的其它离散/连续分布
            elif self.J_distribution[0] == 'triangular':

                # 标识为连续分布
                self.J_discrete = False

            else:
                raise ValueError("Unsupported J_distribution type.")

            # init J -> clac p
            self.J = self.sample_J()
            self.p = self.calculate_p(self.J)
            print(f"{self.J_distribution[0]} Distribution Initialized with J={self.J}, calculated p={self.p:.4f}")
            print(f"{self.J_distribution[0]} 分布 初始化 J={self.J}, 计算得到单局win rate p={self.p}")

    def step(self, action):
        """
        进行一步操作，返回奖励和下一个状态
        :param action: 被拉动的臂
        :return: 奖励和下一个状态
        """
        reward = self.get_reward(action)
        next_state = self.get_next_state(action)  # TODO 连输状态J
        return reward, next_state

    def sample_J(self):
        """
        TODO 
        抽取J值
        :return: 抽取的J值
        """
        if self.J_discrete:
            rand_num = random.random()
            for i, prob in enumerate(self.cumulative_probs):
                if rand_num < prob:
                    return self.J_values[i]
        else:
            return np.random.chisquare(self.df)
        
    def calculate_p(self, J):
        pass

    def get_reward(self, action: int = None):
        """
        摇一次臂, 获取reward, 更新loss_state
        动态版还要判断更新J值
        :return: _description_
        :rtype: _type_
        """

        if action is None or action < 0 or action >= len(self.p):
            raise ValueError("Invalid action. Action must be within the range of available arms.")

        if random.random() < self.p[action]:
            reward = self.mu[action]  # 根据均值生成奖励
            self.loss_state = 0  # 重置loss_state
        else:
            reward = 0
            self.loss_state += 1

        if self.loss_state >= self.J:
            # !!! 触发Futurity
            # print(f"J值变化了 {self.J_change_cum} 次")
            # print(f"Futurity 在第 {self.steps_now} 次变化")
            reward = self.J  # 达到J值，返回J作为奖励
            self.loss_state = 0  # 重置loss_state
            self.J_active_times += 1  # 更新触发次数
        
        return reward

    def get_optimal_reward(self):
        return None

    def get_mu(self):
        """
        返回所有臂的奖励均值列表

        :return: 所有手臂奖励均值列表
        :rtype: np.array
        """

        return self.mu*self.p

    def get_optimal_mu(self):
        """
        返回Optimal臂的奖励均值列表

        :return: 所有手臂奖励均值列表
        :rtype: np.array
        """

        return np.max(self.mu*self.p)
