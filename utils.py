"""
    此为一个Python模块，包含对Futurity和Bandit的奖励函数的放大区分。
"""

import numpy as np

def trans_func(reward, alpha, mode='linear'):
    """
    调整奖励值的函数。用于对大额奖励（J等）进行线性、指数或对数放大。
    
    参数:
        reward (float): 奖励值。
        alpha (float): 调整参数。
        mode (str): 调整模式，可以是 'linear'、'exponential' 或 'logarithmic'。
    
    返回:
        float: 调整后的奖励值。
    """
    if mode == 'linear':
        return -max(-(reward - alpha), 0) + alpha
    if mode == 'exponential':
        return reward ** alpha 
    # elif mode == 'exponential':
    #     return reward * (1 + alpha * (reward - beta))
    # elif mode == 'logarithmic':
    #     return reward * (1 + alpha * np.log(reward - beta))
    else:
        raise ValueError("Invalid mode. Choose from 'linear', 'exponential', or 'logarithmic'.")


def get_trans_reward(reward, alpha_list:np.array, mode='linear'):
    """我希望返回一个，对alpha_list里的每个alpha都要得到一个经过trans_func函数变形的np.array

    :param reward: _description_
    :type reward: _type_
    :param alpha_list: _description_
    :type alpha_list: _type_
    """

    return np.array([trans_func(reward, alpha, mode) for alpha in alpha_list])

# best_strategy里面画图用
def plot_accum_reward(ax, bandit_10_arm,  num_steps, return_list, method='Sarsa', window_size=10):

    episodes_list = list(range(len(return_list)))

    # 计算滑动平均-不同算法之间比较
    moving_avg = np.convolve(return_list, np.ones(window_size)/window_size, mode='valid')
    moving_avg_episodes = episodes_list[:len(moving_avg)]
    moving_avg = moving_avg[::10] # 每10个取一个
    moving_avg_episodes = np.array(moving_avg_episodes[::10]) + 10 # 每10个取一个

    # import pdb; pdb.set_trace()
    ax.scatter(episodes_list, return_list,  s=10, alpha=0.7)
    # ax.scatter(episodes_list, return_list, color='gray', s=10, alpha=0.7)

    # 绘制滑动平均线
    if method == 'Sarsa':
        # 绘制滑动平均线
        ax.plot(moving_avg_episodes, moving_avg, label=f'MA-{method}', marker='^',markersize=5, 
                linestyle='--', linewidth=2.5)
    elif method == 'DQN':
        # 绘制滑动平均线
        ax.plot(moving_avg_episodes, moving_avg, label=f'MA-{method}', marker='o', markersize=5, 
                linestyle='--',  linewidth=2.5)
    elif method == 'K Best':
        # 绘制滑动平均线
        ax.plot(moving_avg_episodes, moving_avg, label=f'MA-K Threshold', marker='o', markersize=5, 
                linestyle='--', linewidth=2.5)
        
    # ax.set_title(r"动态补偿老虎机下不同策略收益对比")
    # ax.set_title('{} on {} J'.format(method, bandit_10_arm.J_distribution[0]))

    ax.set_xlabel(r'Episodes', fontsize=20)
    # ax.set_title(r"Strategies on Dynamic Futurity Slot Machine", fontsize=22)
    ax.set_title(r"Strategies on Static Futurity Slot Machine", fontsize=22)

    ax.legend(fontsize=15, loc='upper right')
    # ax.set_title(r"动态补偿老虎机下不同策略收益对比")


def print_agent(agent, max_J, action_meaning, disaster=[], end=[]):
    """打印agent的最优动作策略

    :param agent: _description_
    :type agent: _type_
    :param max_J: _description_
    :type max_J: _type_
    :param action_meaning: _description_
    :type action_meaning: _type_
    :param disaster: _description_, defaults to []
    :type disaster: list, optional
    :param end: _description_, defaults to []
    :type end: list, optional
    """
    print('\n======================================')

    for i in range(max_J):
        a = agent.best_action(i)
        print(f"状态 {i} 下的最优Arm序号为：{np.argmax(a)} \n", end="")
    print('======================================')



if __name__ == "__main__":

    reward = 5
    alpha_list = np.array(range(1, 10))
    adjusted_reward = get_trans_reward(reward, alpha_list=alpha_list, mode='linear')
    print(adjusted_reward)  # 输出结果

