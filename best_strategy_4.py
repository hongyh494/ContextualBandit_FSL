import os
import sys
import argparse
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# nn dependencies
import torch
from scipy.optimize import fsolve
from joblib import Parallel, delayed

# offline dependencies
from algorithm.BestStrategy.rl_utils import ReplayBuffer

from algorithm.BestStrategy.policy_based import Sarsa, DQN
from model.BernBandit import BernoulliDynamicBandit
from model.CateBandit import CategoricalDynamicBandit
from utils import plot_accum_reward, print_agent

warnings.filterwarnings('ignore')

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4" # 设置OpenMP线程数

def apply_sarsa(ax, seed, bandit_10_arm, max_J=10,
                num_episodes=100, num_steps=20000, alpha=0.1, gamma=1, epsilon=0.1):
    """应用Sarsa算法
    Algo Hyper-params设定
    :param bandit_10_arm: Bandit 实例
    :type bandit_10_arm: _type_
    :param num_steps: 每条轨迹的步数
    :type num_steps: int
    :param num_episodes: 训练的回合数/agent在环境中运行的nums/轨迹数
    :type num_episodes: int
    :param max_J: 设定最大J值，状态就有0，..., J-1这几种
    :type max_J: int
    :param alpha: 学习率
    :type alpha: float
    :param gamma: 折扣因子
    :type gamma: float
    :param epsilon: greedy探索率
    :type epsilon: float
    """

    # agent实例化
    agent = Sarsa(bandit_10_arm, max_J=max_J, epsilon=epsilon, alpha=alpha, gamma=gamma)
    # 每条episode的累积回报的序列 size=num_episodes
    return_list = []

    for i in range(10):   # 显示10个进度条
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条里的序列数目
                episode_return = 0
                state = bandit_bernoulli.reset()   # bandit : 重置老虎机状态
                action = agent.take_action(state)  # algorithm : 选择动作手臂
                step_i = 0                         # 步数计数器 每条序列采样的长度
                while step_i < num_steps:
                    # agent选择动作手臂
                    reward, next_state = bandit_bernoulli.step(action)  # bandit : 摇臂，得到t奖励和t+1状态
                    next_action = agent.take_action(next_state)         # algorithm : 采样得到t+1应该选取的动作手臂
                    episode_return += reward                            
                    agent.update(state, action, reward, next_state, next_action)  # algorithm : 更新Q_table
                    state = next_state
                    action = next_action

                    step_i += 1

                while step_i < num_steps:  # 采样到最大步数
                    # agent选择动作手臂
                    reward, next_state = bandit_bernoulli.step(action)
                    step_i += 1
                    episode_return += reward                            

                return_list.append(episode_return)   # 记录每条序列的总采样回报

                if (i_episode + 1) % 10 == 0:   # 每10条序列打印一次这10条的平均回报
                    pbar.set_postfix({
                        'Episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'Average Return': '%d' % np.mean(return_list[-10:])
                    })
                pbar.update(1)


    # 计算滑动平均ACCUM REWARDS-不同算法之间比较
    plot_accum_reward(ax, bandit_10_arm,  num_steps, return_list, method='Sarsa', window_size=10)    

    # 打印最佳策略
    print_agent(agent, max_J, action_meaning=0, disaster=[], end=[])


def apply_dqn(ax, seed, bandit_10_arm, state_dim, hidden_dim, action_dim, 
              num_episodes=100, num_steps=20000, # 训练数据
              max_J=20, gamma=1, learning_rate=0.001, epsilon=0.1, device='cuda'):

    batch_size = 64
    buffer_size = 10000
    minimal_size = 3000
    target_update = 20 # 10步更新一次
    replay_buffer = ReplayBuffer(buffer_size)
    # state_dim = max_J # 纯状态 one-hot
    # state_dim = 2 # [状态, 动作]
    state_dim = 1 # [连输状态]
    action_dim = bandit_10_arm.K # 动作维度

    # agent实例化
    agent = DQN(bandit=bandit_10_arm, state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim, target_update=target_update,
                 max_J=max_J, gamma=gamma, learning_rate=learning_rate, epsilon=epsilon, device=device)
    
    # 每条episode的累积回报的序列 size=num_episodes
    return_list = []

    for i in range(10):   # 显示10个进度条
            with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):  # 每个进度条里的序列数目
                    episode_return = 0
                    state = bandit_bernoulli.reset()   # bandit : 重置老虎机状态
                    step_i = 0                         # 步数计数器 每条序列采样的长度
                    while step_i < num_steps:
                        action = agent.take_action(state)  # algorithm : 选择动作手臂
                        reward, next_state = bandit_bernoulli.step(action)  # bandit : 摇臂，得到t奖励和t+1状态
                        replay_buffer.add(state, action, reward, next_state)
                        episode_return += reward                            
                        state = next_state  # 一个数据组需要(s, a, r, s')这几个数据

                        # 当buffer数据量超过一定量minimal_size时，开始QNet训练
                        if (replay_buffer.size() > minimal_size) & (step_i % 10 ==0):
                            b_s, b_a, b_r, b_ns = replay_buffer.sample(batch_size)
                            transition_dict = {'states': b_s, 
                                               'actions': b_a, 
                                               'next_states': b_ns, 
                                               'rewards': b_r}
                            agent.update(transition_dict)
                        
                        step_i += 1

                    return_list.append(episode_return)   # 记录每条序列的总采样回报

                    if (i_episode + 1) % 10 == 0:   # 每10条序列打印一次这10条的平均回报
                        pbar.set_postfix({
                            'Episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                            'Average Return': '%d' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)

    # 计算滑动平均ACCUM REWARDS-不同算法之间比较
    plot_accum_reward(ax, bandit_10_arm,  num_steps, return_list, method='DQN', window_size=10)

    # 打印最佳策略
    print_agent(agent, max_J, action_meaning=0, disaster=[], end=[])
    


# 根据PPO里面能获得的老虎机的信息,
#  state就是我们的连输状态loss_state, a就是我们摇动的某一个老虎机，
# 写一个策略迭代的class，大概要维护一个Q_table[state, action]。
# 要求实现能经过多轮迭代后在不同的状态state下打印出best_action的函数,；同时还要可视化

if __name__ == "__main__":
    
    # 测试BernoulliDynamicBandit模型
    # 创建一个10臂的Bernoulli动态老虎机模型

    seed = int(2)
    K = 3  # 拉动的臂数
    num_steps = 10000  # 每条轨迹的步数
    num_episodes=100  # 训练的回合数/agent在环境中运行的nums/轨迹数
    max_J=5  # 设定最大J值，状态就有0，..., J-1这几种
    J_distribution = ('static', 4)  # 离散分布
    # J_distribution = ('discrete', [2, 3, 4, 5, 6], [0.2]*5)  # 离散分布


    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    ax = ax.flatten()


    # Seed 1
    seed1 = seed
    bandit_bernoulli = BernoulliDynamicBandit(K=K, 
                                              J_distribution=J_distribution, 
                                              num_steps=num_steps, 
                                              seed=seed1)
    

    # bandit_categorical = CategoricalDynamicBandit(K=K,
    #                                               J_distribution=J_distribution,
    #                                               num_steps=num_steps,
    #                                               seed=seed1)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    apply_sarsa(ax[0], seed1, bandit_bernoulli, gamma=0.9,
                 num_episodes=num_episodes, num_steps=num_steps, max_J=max_J)

    # 只有连输状态的dqn
    # apply_dqn(ax[0], seed1, bandit_bernoulli, state_dim=1, hidden_dim=32, action_dim=K, 
    #           num_episodes=num_episodes, num_steps=num_steps, # 训练数据
    #           max_J=max_J, gamma=0.9, learning_rate=0.01, epsilon=0.1, device=device)

    # [连输状态, 动作]的dqn
    # apply_pair_dqn(ax[0], seed1, bandit_bernoulli, state_dim=2, hidden_dim=32, action_dim=K, 
    #           num_episodes=num_episodes, num_steps=num_steps, # 训练数据
    #           max_J=max_J, gamma=0.9, learning_rate=0.01, epsilon=0.1, device=device)

    # 绘制Futurity Baseline \mu*
    # ax.axhline(y=num_steps, color='r', linestyle='--', label='Compensation Baseline')
    ax[0].axhline(y=num_steps, linestyle='--', label='Compensation Baseline')

    # [state, action] 动作对的dqn, 输出一个标量Q(s,a)
    plt.tight_layout()
    plt.show()
