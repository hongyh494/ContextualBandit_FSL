import numpy as np
import matplotlib.pyplot as plt
import random
from algorithm.BAI.classic import Solver


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


class convex_idtf(Solver):
    """
    017identify empirical算法
    """

    def __init__(self, bandit, alpha_list:np.array ,**kwargs):
        """
        :param bandit: MAB模型对象
        :param alpha_list: alpha的列表, 不同阈值表现不一样
        :param kwargs: 其他参数
        """
        super(convex_idtf, self).__init__(bandit, **kwargs)
        self.alpha_list = alpha_list
        self.alpha_len = len(alpha_list)
        self.estimated_mu = np.zeros(self.bandit.K)  # 初始化每个臂的均值估计为0
        self.estimated_alpha_mu = np.zeros((self.bandit.K, self.alpha_len)) # 横轴代表不同的arm, 纵轴代表不同的alpha下的mean估计

    def run_one_step(self):
        """
        选择动作手臂
        更新counts和estimated_mu, 历史连输J(老虎机内部记录), 返回被拉动的臂,
        :return: 被拉动的臂
        """
        
        if random.random() < 1:
            action = random.randint(0, self.bandit.K - 1)  # 随机选择一个臂
        else:
            action = np.argmax(self.estimated_mu)  # 选择当前估计奖励最大的臂

        # 摇臂得到奖励 可能来自mu可能来自J, 更新了J
        reward = self.bandit.get_reward(action)

        # 更新均值估计
        # self.estimated_mu[action] += (reward - self.estimated_mu[action]) / self.counts[action]
        self.estimated_mu[action] += 1./ (self.counts[action] + 1) * (reward - self.estimated_mu[action])

        # 更新alpha截断均值估计
        alpha_reward = get_trans_reward(reward, alpha_list=self.alpha_list, mode='linear')
        # alpha_reward = get_trans_reward(reward, alpha_list=self.alpha_list*3, mode='exponential')
        # import pdb; pdb.set_trace()
        self.estimated_alpha_mu[action, :] += 1./ (self.counts[action] + 1) * (alpha_reward - self.estimated_alpha_mu[action, :])
        
        return action
    

class p_estimate_idtf(Solver):
    """
    hyh identify empirical算法
    """

    def __init__(self, bandit, t_list:np.array ,**kwargs):
        """
        :param bandit: MAB模型对象
        :param t_list: 表示每个手臂的探测次数的序列，随着t增大估计应该越准
        :param kwargs: 其他参数
        """
        super(p_estimate_idtf, self).__init__(bandit, **kwargs)
        self.t_list = t_list
        self.t_len = len(t_list)
        self.estimated_mu = np.zeros(self.bandit.K)  # 初始化每个臂的均值估计为0
        self.estimated_hit = np.zeros((self.bandit.K, self.t_len)) # 横轴代表不同的arm, 纵轴代表不同的alpha下的mean估计

    def run_one_step(self, step, action):
        """
        !!! 注意由于先选手臂因此step 和action轮询已经给定，所以更新run函数
        p_estimate_idtf 是均匀选择动作手臂先。因此不会犯错。
        更新counts和estimated_mu, 历史连输J(老虎机内部记录), 返回被拉动的臂,
        :return: 被拉动的臂的Reward
        """

        # 摇臂得到奖励 可能来自mu可能来自J, 更新了J
        reward = self.bandit.get_reward(action)

        if step == 0 :
            # 第一个点 直接0 or 1
            self.estimated_hit[action, step] = np.sign(reward)
        
        else : 
            self.estimated_hit[action, step] = self.estimated_hit[action, step-1] + 1./ (step + 1) * (np.sign(reward) - self.estimated_hit[action, step-1])
        
        # 更新均值估计
        # self.estimated_mu[action] += (reward - self.estimated_mu[action]) / self.counts[action]
        self.estimated_mu[action] += 1./ (self.counts[action] + 1) * (reward - self.estimated_mu[action])

        return reward
    
    def run(self, num_steps):
        """
        !!! 注意由于先选手臂因此step 和action轮询已经给定，所以更新run函数
        单条path, 运行一定次数
        :param num_steps: 单条path运行的步数
        :return: None
        """

        for action in range(self.bandit.K):
            for step in range(num_steps//self.bandit.K):
                reward = self.run_one_step(step, action)  # 选择动作手臂
                self.counts[action] += 1  # 更新被拉动的臂的次数
                self.update_regret(action)  # 更新遗憾值 在给定\mu的情形下
                self.actions.append(action)

        return None

