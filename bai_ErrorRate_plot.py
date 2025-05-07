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
import seaborn as sns

warnings.filterwarnings("ignore")
# Set seaborn style for plots
sns.set_theme(style="darkgrid")
# 启用 LaTeX 样式（可选）
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def run_p_estimate_experiment(ax, seed, num_steps, bandit_10_arm):

    print("所有手臂均值mu_p：", bandit_10_arm.get_mu_p())
    print("所有手臂真实概率p：", bandit_10_arm.p)
    print("所有手臂真实奖励mu：", bandit_10_arm.mu)

    random.seed(seed)
    t_list = np.array(range(num_steps//bandit_10_arm.K))  # TODO 这是输入每个手臂分配的次数
    idtf_solver = p_estimate_idtf(bandit_10_arm, t_list)
    idtf_solver.run(num_steps)
    # import pdb; pdb.set_trace()
    print('P-IDTF算法的累积懊悔为：', idtf_solver.regret)
    # plot_regret_single(ax, solvers=[idtf_solver], solver_names=["idtf"])
    plot_p_estimate_single(ax, solvers=[idtf_solver], solver_names=["idtf"], t_list=t_list)
    print("Futurity Mechanism激发次数", bandit_10_arm.J_active_times)
    print("估计的均值mu是", idtf_solver.estimated_mu)
    print("估计的Hit Frequency是", idtf_solver.estimated_hit)



def plot_error_rate(ax, bandit_10_arm, label="Categorical Bandit", axes=None):
    """
    传进来某个老虎机的实例，在ax上画图
    横坐标是budget/T_0 也就是num_steps。纵坐标是probability of error，一张图里画一个点；
    不同的图画不同K的K臂老虎机，一个老虎机做100次试验，看看是否识别出来了最大的手臂,
    bandit_10_arm.p p大的是最优的
    idtf_solver.estimated_hit估计最大值的索引是最优的臂
    比较就看这两个的argmax是不是一样的，一样的记为1，不一样的记为0，然后bandit_10_arm.reset()后重复摇100次，统计错分率。

    """

    K = bandit_10_arm.K  # 拉动的臂数
    seed_nums = 3000 # 每个预算重复num次实验
    num_steps_list = range(200*K, 2000*K, 200*K)    # 每个手臂给的预算探索次数*K
    # num_steps_list = range(2*K, 20*K, 2*K)     # 每个手臂给的预算探索次数*K debug
    error_rate_list = []  # 错分率列表,从1到T_0


    for num_steps in tqdm(num_steps_list, desc=f"{label[:4]} Fut{bandit_10_arm.J_distribution[0]} || Budgets progress"):

        error_count = 0

        # for seed in tqdm(range(seed_nums), desc=(f"{seed_nums}"+r" Trials")):  # 每个预算重复100次实验
        for seed in range(seed_nums):  # 每个预算重复100次实验
            bandit_10_arm.reset()  # 重置老虎机状态

            t_list = np.array(range(num_steps // bandit_10_arm.K))
            idtf_solver = p_estimate_idtf(bandit_10_arm, t_list)
            idtf_solver.run(num_steps)

            # !!!!注意这里estimated.shape = (K, num_steps//K)
            if np.argmax(bandit_10_arm.p) != np.argmax(np.mean(idtf_solver.estimated_hit,axis=1)):
                error_count += 1
        error_rate = error_count / seed_nums
        error_rate_list.append(error_rate)
        print(f"\n K={K} Arms || Budgets={num_steps//K} || Error Rate: {100*error_rate:.2f}% ")

    ax.plot([steps / K for steps in num_steps_list], error_rate_list, label=label, marker='*')
    # ax.set_xlabel(r"Budget / T_0")
    # ax.set_ylabel(r"Probability of Error(e_{T_0})")
    ax.set_xlabel(r"Budgets per Arm ($T_0 // K$)")
    ax.set_ylabel(r"Probability of Error($e_{T_0}$)")
    ax.legend()


def run_error_plot(ax, K=5, seed=int(2),
                   J_distribution=('discrete', [2, 3, 4, 5, 6], [0.2]*5), 
                   non_J_distribution=('static', 3)):
    
    num_steps = 1000*K  # 每条轨迹的步数

    # Seed 1
    seed1 = seed
    bandit_bernoulli_dyna = BernoulliDynamicBandit(K=K, 
                                              J_distribution=J_distribution, 
                                              num_steps=num_steps, 
                                              seed=seed1, error_plot=True)
    
    bandit_categorical_dyna = CategoricalDynamicBandit(K=K, 
                                              J_distribution=J_distribution, 
                                              num_steps=num_steps, 
                                              seed=seed1, error_plot=True)
    
    print(f"=========================================\n Futurity={J_distribution[0]} \n")
    print("============ \n Bern Futurity J value:happens", bandit_bernoulli_dyna.J_history)
    print("Cate Futurity J value:happens", bandit_categorical_dyna.J_history, "\n ============ ")

    bandit_bernoulli_static = BernoulliDynamicBandit(K=K, 
                                              J_distribution=non_J_distribution, 
                                              num_steps=num_steps, 
                                              seed=seed1, error_plot=True)
    bandit_categorical_static = CategoricalDynamicBandit(K=K,
                                                  J_distribution=non_J_distribution,
                                                  num_steps=num_steps,
                                                  seed=seed1, error_plot=True)
    
    for bandit in [bandit_bernoulli_dyna, bandit_bernoulli_static]:
        plot_error_rate(ax, bandit, label=f"Bern \& J {bandit.J_distribution[0]}", axes=ax)

    for bandit in [bandit_categorical_dyna, bandit_categorical_static]:
        plot_error_rate(ax, bandit, label=f"Cate \& J {bandit.J_distribution[0]}", axes=ax)
    
    ax.set_title(r"$K={}$ Arms Futurity Bandits".format(K))


if __name__ == "__main__":
    
    # 测试BernoulliDynamicBandit模型
    # 创建一个10臂的Bernoulli动态老虎机模型

    seed = int(2)

    ### 终稿作图

    # !!! 都可以做出来 但是categorical还没尝试
    # J_distribution = ('static', 3)  # 离散分布
    J_distribution = ('discrete', [2, 3, 4, 5, 6], [0.2]*5)  # 离散分布
    non_J_distribution = ('static', 3)  # 无奖励时的离散分布

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))  # 不同的K再换
    # fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax = ax.flatten()

    for i, K in enumerate([3, 5, 10, 15]):
        run_error_plot(ax[i], K=K, seed=i, 
                       J_distribution=J_distribution, non_J_distribution=non_J_distribution)
    
    # ax[0].set_title("10 Arm Futurity Bandits")
    plt.savefig("error_rate_plot.png", dpi=600)
    plt.tight_layout()
    plt.show()