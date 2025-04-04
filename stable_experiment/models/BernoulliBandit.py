import math
import numpy as np
import random

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from joblib import Parallel, delayed

# Bernoulli Reward Distribution (两点分布) Dynamic Futurity @ single arm ver.
class BernoulliDynamicBandit:
    def __init__(self, J_distribution, df=2, num_steps=1000, seed=None):
        """
        初始化单臂老虎机模型, J值变化版

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
        self.J_discrete = None  # 是否是离散分布 调整reward的触发P有关

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # 预处理离散分布的累积分布函数 (CDF)
        if self.J_distribution[0] == 'discrete':

            # 标识为离散分布
            self.J_discrete = True

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

        # 抽取初始的 J 并计算对应的 p
        self.J = self.sample_J()
        self.p = self.calculate_p(self.J)
        print(f"{self.J_distribution[0]} Distribution Initialized with J={self.J}, calculated p={self.p:.4f}")
        print(f"{self.J_distribution[0]} 分布 初始化 J={self.J}, 计算得到单局win rate p={self.p}")

    def sample_J(self):
        """
        根据 J_distribution 抽取一个新的 J 值。

        :return: 抽取到的 J 值（整数）
        """
        if self.J_distribution[0] == 'discrete':
            J = random.choices(self.J_distribution[1], weights=self.J_distribution[2])[0]
            return int(J)
        
        elif self.J_distribution[0] == 'chisquare':
            sampled_J = np.random.chisquare(self.df) + 3
            return sampled_J  # 向上取整确保 J 至少为 1 TODO 连续型的如何保证不落入<1的情况

            sampled_J = np.random.chisquare(self.df)
            return sampled_J  # 如果不取整，J值可以是小数，但是不影响计算p值?
        
        elif self.J_distribution[0] == 'triangular':
            sampled_J = np.random.triangular(3, 6, 9)
            return sampled_J
        
        else:
            raise ValueError("Unsupported J_distribution type.")
        
        
    def calculate_p(self, J):
        """
        计算 p 值，使得渐进期望为零。

        方程：$\mu_A + J * p * (1 - p)^J / (1 - (1 - p)^J) = 0$
        2p - 1 + J * p * (1 - p)^J / (1 - (1 - p)^J) = 0

        :param J: 当前的 J 值
        :return: 计算得到的 p 值, 为Bernoulli 成功获利概率
        """

        # from paper Markovian
        def asymptotic_expectation_zero(p, J):
            # 算亏钱的时候的期望
            # mu = 2 * p - 1
            # numerator = (J-1) * p * (1 - p)**(J-1)
            # denominator = 1 - (1 - p)**(J-1)

            # 不算亏钱的时候的期望 计算得到的期望
            mu = 2 * p 
            numerator = (J) * p * (1 - p)**(J)
            denominator = 1 - (1 - p)**(J)
            asymptotic_mu_star = mu + numerator / denominator -1
            return asymptotic_mu_star

        # from paper CZJ
        # def asymptotic_expectation_zero(p, J):
        #     mu = p
        #     numerator = J * p * (1 - p)**J
        #     denominator = 1 - (1 - p)**J
        #     asymptotic_mu_star = p * mu + numerator / denominator
        #     return asymptotic_mu_star

        # 使用 fsolve 进行求解 hardcode initial guess
        p_initial_guess = 0.5
        p_solution, = fsolve(asymptotic_expectation_zero, p_initial_guess, args=(J,))
        return p_solution

    def simulate_single_path(self, j_state_init=0):
        """
        模拟单条路径。

        :return: 累积奖励轨迹 (numpy 数组）
        """
        J = self.sample_J()
        p = self.calculate_p(J)
        total_reward = 0
        consecutive_losses = j_state_init
        rewards = np.zeros(self.num_steps)

        # 这个reward 获利设定和mu有关, 间接影响p的公平性设定,reward此处先假设0,2,然后每一把1块费用
        # TODO 对于不同的reward分布设定，reward需要sampling.

        for step in range(self.num_steps):
            if np.random.rand() < p:
                reward = 1
                consecutive_losses = 0
            else:
                reward = -1
                consecutive_losses += 1

            # Futurity mechnism !!!

            # 算亏钱的时候 sota接近的期望 pass
            # if consecutive_losses >= J:
            #     reward = J+2  # 添加补偿
            #     consecutive_losses = 0

            # p不算亏钱的时候 sota接近的期望 ！
            if consecutive_losses >= J:
                if self.J_discrete:
                    reward = J-1  # 离散型J添加离散型补偿
                else:
                    # reward = math.ceil(J-1)  # 连续型J添加离散型补偿
                    reward = J  # 连续型J添加连续型补偿 貌似在chi-square下还可以 但也会漂移

                consecutive_losses = 0

                # 更新 J 和 p
                J = self.sample_J()
                p = self.calculate_p(J)  # 离散型 无需取整
                # p = self.calculate_p(math.ceil(J))  #连续型J添加离散型补偿
                self.J_change_cum += 1

            total_reward += reward
            rewards[step] = total_reward

        # 检查 J 是否发生变化, 如果离散多J分布or连续分布，J值未发生变化，可能存在问题
        if self.J_change_cum == 0 and (len(self.J_distribution[1])>1 or self.J_distribution[0] != 'discrete'):
            print(f"分布{self.J_distribution[0]} J值未发生变化, 存在问题")
        
        return rewards


    def simulate_multiple_paths(self, num_paths=1000, j_state_init=0, n_jobs=-1):
        """
        并行模拟多条路径。

        :param num_paths: 模拟的轨迹数
        :param j_state_init: 初始计数时的 累计损失j 状态
        :param n_jobs: 并行作业数，-1 表示使用所有可用的 CPU 核心
        :return: 所有轨迹的奖励数组（num_paths x num_steps）
        """

        # 初始化一个二维数组来存储所有路径的奖励
        results = np.zeros((num_paths, self.num_steps))

        # 使用 tqdm 添加进度条
        for i in tqdm(range(num_paths), desc="Simulating paths"):
            results[i, :] = np.array(self.simulate_single_path(j_state_init=j_state_init))

        # 验证results dimensions, 应是二维数组, 模拟轨迹数 x 每条轨迹的步数
        assert results.shape == (num_paths, self.num_steps)

        return results
    
    # @staticmethod  # 静态方法，不需要实例化即可调用
    # def plot_paths_with_mean(rewards_array):
    #     """
    #     绘制所有路径的奖励，并在图上标出每个时间步的平均奖励。

    #     :param rewards_array: 所有轨迹的奖励数组（num_paths x num_steps）
    #     """
    #     num_paths, num_steps = rewards_array.shape

    #     plt.figure(figsize=(12, 8))

    #     # 绘制每条路径
    #     for i in range(num_paths):
    #         plt.plot(rewards_array[i, :], color='lightgray', alpha=0.5)

    #     # 计算并绘制平均路径
    #     mean_rewards = np.mean(rewards_array, axis=0)
    #     plt.plot(mean_rewards, color='red', linewidth=2, label='Mean Reward')

    #     plt.title("Paths and Mean Reward")
    #     plt.xlabel("Steps")
    #     plt.ylabel("Cumulative Reward")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    #  适合用来看单个j的情况下的paths
    #  只负责把所有的paths和mean reward画出来，对不同的j_state_init都一视同仁，
    def plot_paths_with_mean(self, rewards_array, plot_show=True):
        """
        绘制所有路径的奖励，并在图上标出每个时间步的平均奖励。

        :param rewards_array: 所有轨迹的奖励数组(num_paths x num_steps)
        """
        num_paths, num_steps = rewards_array.shape

        plt.figure(figsize=(12, 8))

        # 绘制每条路径
        for i in tqdm(range(num_paths), desc="Plotting paths"):
            plt.plot(rewards_array[i, :], color='lightgray', alpha=0.5)

        # 计算并绘制平均路径
        mean_rewards = np.mean(rewards_array, axis=0)

        # 如果只看某个j_state状态下的paths
        if plot_show:
            plt.plot(mean_rewards, color='red', linewidth=2, label='Mean Reward')
            plt.title("Paths and Mean Reward")
            plt.xlabel("Steps")
            plt.ylabel("Cumulative Reward")
            plt.legend()
            plt.grid(True)
            plt.show()


        return mean_rewards

    # 2.24 突然想到 应该是对初始状态平稳下的初始i分别统计模拟\pi(i, j) for j in range(0, J-1)
    def plot_paths_with_mean_init_stationarity(self, rewards_array, j_states: list):
        """
        绘制所有路径的奖励，并在图上标出每个时间步的平均奖励。

        :param rewards_array: 所有轨迹的奖励数组(num_paths x num_steps)
        """
        num_paths, num_steps = rewards_array.shape

        plt.figure(figsize=(12, 8))

        # 绘制每种初始状态(主要是初始时已经累计j次损失)下的路径
        for j_init_state in tqdm(j_states):
            j_iniit_state = self.plot_paths_with_mean(rewards_array, j_state_init=j_init_state, plot_show=False)

        # 计算并绘制平均路径
        mean_rewards = np.mean(rewards_array, axis=0)
        plt.plot(mean_rewards, color='red', linewidth=2, label='Mean Reward')

        plt.title("Paths and Mean Reward")
        plt.xlabel("Steps")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid(True)
        plt.show()
