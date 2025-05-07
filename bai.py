"""
本脚本用于
1. 017:可视化不同手臂的随着alpha的估计的变化图
2. 估计不同arm的Hit Frequency
"""

import warnings
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from scipy.optimize import fsolve
from joblib import Parallel, delayed

from algorithm.BAI.classic import EpsilonGreedy, UCB, ThompsonSampling, GaussianTS, ExponentialTS, PoissonTS, \
    plot_regret_single, plot_mu_alpha_single, plot_p_estimate_single
from algorithm.BAI.SOTA import PPOSolver, EpsilonGreedyMedian, UCBMedian# DDPG, DQN, A2C, TRPO
from algorithm.BAI.identify import convex_idtf, p_estimate_idtf
from model.BernBandit import BernoulliDynamicBandit
from model.CateBandit import CategoricalDynamicBandit
from utils import get_trans_reward

warnings.filterwarnings("ignore")


def run_bai_experiment(ax, seed, num_steps, bandit_10_arm):

    print("所有手臂均值mu_p：", bandit_10_arm.get_mu_p())
    print("所有手臂真实概率p：", bandit_10_arm.p)
    print("所有手臂真实奖励mu：", bandit_10_arm.mu)

    random.seed(seed)
    alpha_list = np.linspace(0, 5, num=20)
    idtf_solver = convex_idtf(bandit_10_arm, alpha_list)
    idtf_solver.run(num_steps)
    # import pdb; pdb.set_trace()
    print('BAI-IDTF算法的累积懊悔为：', idtf_solver.regret)
    # plot_regret_single(ax, solvers=[idtf_solver], solver_names=["idtf"])
    plot_mu_alpha_single(ax, solvers=[idtf_solver], solver_names=["idtf"], alpha_list=alpha_list)
    print("Futurity Mechanism激发次数", bandit_10_arm.J_active_times)
    print("估计的均值mu是", idtf_solver.estimated_mu)


def run_p_estimate_experiment(ax, seed, num_steps, bandit_10_arm):

    print("所有手臂均值mu_p：", bandit_10_arm.get_mu_p())
    print("所有手臂真实概率p：", bandit_10_arm.p)
    print("所有手臂真实奖励mu：", bandit_10_arm.mu)

    random.seed(seed)
    t_list = np.array(range(num_steps//bandit_10_arm.K))  # TODO 这是每个手臂分配的次数
    idtf_solver = p_estimate_idtf(bandit_10_arm, t_list)
    idtf_solver.run(num_steps)
    # import pdb; pdb.set_trace()
    print('P-IDTF算法的累积懊悔为：', idtf_solver.regret)
    # plot_regret_single(ax, solvers=[idtf_solver], solver_names=["idtf"])
    plot_p_estimate_single(ax, solvers=[idtf_solver], solver_names=["idtf"], t_list=t_list)
    print("Futurity Mechanism激发次数", bandit_10_arm.J_active_times)
    print("估计的均值mu是", idtf_solver.estimated_mu)
    print("估计的Hit Frequency是", idtf_solver.estimated_hit)



if __name__ == "__main__":
    
    # 测试BernoulliDynamicBandit模型
    # 创建一个10臂的Bernoulli动态老虎机模型

    seed = int(3)
    K = 10  # 拉动的臂数
    num_steps = 1000*K  # 每条轨迹的步数

    # !!! 都可以做出来 但是categorical还没尝试
    # J_distribution = ('static', 3)  # 离散分布
    J_distribution = ('discrete', [2, 3, 4, 5, 6], [0.2]*5)  # 离散分布

    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    ax = ax.flatten()

    # Seed 1
    seed1 = seed
    bandit_bernoulli_dyna = BernoulliDynamicBandit(K=K, 
                                              J_distribution=J_distribution, 
                                              num_steps=num_steps, 
                                              seed=seed1)
    
    bandit_categorical_dyna = CategoricalDynamicBandit(K=K, 
                                              J_distribution=J_distribution, 
                                              num_steps=num_steps, 
                                              seed=seed1)
    

    run_bai_experiment(ax[0], seed1, num_steps, bandit_bernoulli_dyna)
    run_bai_experiment(ax[1], seed1, num_steps, bandit_categorical_dyna)
    print(" ============ \n Bern Futurity J value:happens", bandit_bernoulli_dyna.J_history)
    print("Cate Futurity J value:happens", bandit_categorical_dyna.J_history, "\n ============ ")

    non_J_distribution = ('static', 3)  # 无奖励时的离散分布
    bandit_bernoulli_static = BernoulliDynamicBandit(K=K, 
                                              J_distribution=non_J_distribution, 
                                              num_steps=num_steps, 
                                              seed=seed1)
    bandit_categorical_static = CategoricalDynamicBandit(K=K,
                                                  J_distribution=non_J_distribution,
                                                  num_steps=num_steps,
                                                  seed=seed1)
    
    run_bai_experiment(ax[2], seed1, num_steps, bandit_bernoulli_static)
    run_bai_experiment(ax[3], seed1, num_steps, bandit_categorical_static)

    ax[0].set_title("Bernoulli Bandit (Dynamic J)")
    ax[1].set_title("Categorical Bandit (Dynamic J)")
    ax[2].set_title("Bernoulli Bandit (Static)")
    ax[3].set_title("Categorical Bandit (Static)")
    print(" ============ \n Bern Futurity J value:happens", bandit_bernoulli_static.J_history)
    print("Cate Futurity J value:happens", bandit_categorical_static.J_history, "\n ============ ")
    plt.tight_layout()
    plt.show()


    # hyh estimate   # 估计不同arm的Hit Frequency

    fig2, ax2 = plt.subplots(2, 2, figsize=(12, 6))
    ax2 = ax2.flatten()

    run_p_estimate_experiment(ax2[0], seed1, num_steps, bandit_bernoulli_dyna)
    run_p_estimate_experiment(ax2[1], seed1, num_steps, bandit_categorical_dyna)
    run_p_estimate_experiment(ax2[2], seed1, num_steps, bandit_bernoulli_static)
    run_p_estimate_experiment(ax2[3], seed1, num_steps, bandit_categorical_static)
    print(" ============ \n Bern Futurity J value:happens", bandit_bernoulli_dyna.J_history)
    print("Cate Futurity J value:happens", bandit_categorical_dyna.J_history, "\n ============ ")

    ax2[0].set_title("Bernoulli Bandit (Dynamic J)")
    ax2[1].set_title("Categorical Bandit (Dynamic J)")
    ax2[2].set_title("Bernoulli Bandit (Static)")
    ax2[3].set_title("Categorical Bandit (Static)")
    print(" ============ \n Bern Futurity J value:happens", bandit_bernoulli_static.J_history)
    print("Cate Futurity J value:happens", bandit_categorical_static.J_history, "\n ============ ")
    plt.tight_layout()
    plt.show()
