import os
import sys
import argparse
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from scipy.optimize import fsolve
from joblib import Parallel, delayed

from algorithm.classic import EpsilonGreedy, UCB, ThompsonSampling, GaussianTS, plot_results
from model.BernBandit import BernoulliDynamicBandit
from model.CateBandit import CategoricalDynamicBandit

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # 测试BernoulliDynamicBandit模型
    # 创建一个10臂的Bernoulli动态老虎机模型

    seed = int(2)
    K = 10  # 拉动的臂数
    J_distribution = ('static', 5)  # 离散分布
    # J_distribution = ('discrete', [0, 1], [0.5, 0.5])  # 离散分布

    bandit_10_arm = BernoulliDynamicBandit(K=K, 
                                        J_distribution=J_distribution, 
                                        num_steps=1000, 
                                        seed=seed)
    
    np.set_printoptions(precision=4, suppress=True)

    print("所有手臂均值：", bandit_10_arm.get_mu())
    print("所有手臂真实概率p：", bandit_10_arm.p)
    print("所有手臂真实奖励mu：", bandit_10_arm.mu)

    # 测试EpsilonGreedy算法

    random.seed(seed)
    epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.05)
    epsilon_greedy_solver.run(5000)
    print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
    plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])
    print("Futurity Mechanism激发次数", bandit_10_arm.J_active_times)

    # 测试UCB算法
    random.seed(seed)
    ucb_solver = UCB(bandit_10_arm, coef=1)
    ucb_solver.run(5000)
    print('UCB算法的累积懊悔为：', ucb_solver.regret)
    plot_results([ucb_solver], ["UCB"])
    print("Futurity Mechanism激发次数", bandit_10_arm.J_active_times)

    # 测试ThompsonSampling算法
    random.seed(seed)
    # thompson_solver = ThompsonSampling(bandit_10_arm)
    # thompson_solver.run(5000)
    # print('Thompson Sampling算法的累积懊悔为：', thompson_solver.regret)
    # plot_results([thompson_solver], ["ThompsonSampling"])
    # print("Futurity Mechanism激发次数", bandit_10_arm.J_active_times)

    # 测试GaussianTS算法
    random.seed(seed)
    gaussian_solver = GaussianTS(bandit_10_arm)
    gaussian_solver.run(5000)
    print('Gaussian TS算法的累积懊悔为：', gaussian_solver.regret)
    plot_results([gaussian_solver], ["GaussianTS"])
    print("Futurity Mechanism激发次数", bandit_10_arm.J_active_times)

    plt.show()

