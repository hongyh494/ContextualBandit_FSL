import numpy as np
import matplotlib.pyplot as plt
import random
from algorithm.BestStrategy.base import Solver

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



# Sarsa算法 ON-POLICY
class Sarsa(Solver):
    """
    Sarsa算法
    """

    def __init__(self, bandit, env=None, max_J=20, epsilon=0.1, alpha=0.1, gamma=1, **kwargs):
        """
        :param bandit: MAB模型对象
        :param env: 环境对象
        :param kwargs: 其他参数
        """
        super(Sarsa, self).__init__(bandit, env, **kwargs)

        self.Q_table = np.zeros([max_J, bandit.K])  # Q_table
        self.n_actions = bandit.K # 动作空间大小
        self.alpha = alpha   # 学习率
        self.gamma = gamma    # 折扣因子
        self.epsilon = epsilon # greedy探索率
   
    def take_action(self, state):
        """
        选择动作手臂
        历史连输J(老虎机内部记录), 返回被拉动的臂,
        :return: 被拉动的臂
        """

        if np.random.rand() < self.epsilon:
            # 探索
            action = np.random.choice(self.n_actions)
        else:
            # 利用
            action = np.argmax(self.Q_table[state, :])
        
        # 摇臂得到奖励reward
        # reward = self.bandit.get_reward(action)

        return action
    

    def best_action(self, state):
        """
        用于打印策略
        :param state: 当前状态
        :return: 最优动作
        """
        Q_max = np.max(self.Q_table[state, :])
        a = [0 for _ in range(self.n_actions)]
        for i in range(self.n_actions):   # 若两个动作的Q值相同，则都记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1

        return a
    
    def update(self, state, action, reward, next_state, next_action):
        """
        更新Q值
        :param state: 当前状态
        :param action: 当前动作
        :param reward: 奖励
        :param next_state: 下一个状态
        :param next_action: 下一个动作
        """
        # 更新Q_table
        # best_next_action = np.argmax(self.Q_table[next_state, :])
        # td_target = reward + self.gamma * self.Q_table[next_state, best_next_action]
        # td_error = td_target - self.Q_table[state, action]
        # self.Q_table[state, action] += self.alpha * td_error

        td_error = reward + self.gamma * self.Q_table[next_state, next_action] - self.Q_table[state, action]
        self.Q_table[state, action] += self.alpha * td_error


# DQN算法 

class Qnet(torch.nn.Module):
    """ 1 hidden layer QNet """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQN(Solver):
    def __init__(self, bandit, state_dim, hidden_dim, action_dim, target_update,
                 env=None, max_J=20, gamma=1, learning_rate=0.001, epsilon=0.1, device='cuda',**kwargs):
        """
        :param bandit: MAB模型对象
        :param env: 环境对象
        :param kwargs: 其他参数
        """
        super(DQN, self).__init__(bandit, env, **kwargs)

        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)

        self.action_dim = action_dim # 动作空间大小/bandit.K
        self.gamma = gamma    # 折扣因子
        self.epsilon = epsilon  # 贪心比例
        self.target_update = target_update # 目标网络更新频率

        # adam 优化器
        self.optimizer = optim.Adam(self.q_net.parameters(), 
                                    lr=learning_rate)
        
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        """
        选择动作手臂
        历史连输J(老虎机内部记录), 返回被拉动的臂,
        :return: 被拉动的臂
        """

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action


    def best_action(self, state):
        """
        用于打印策略
        :param state: 当前状态
        :return: 最优动作
        """

        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    
    def update(self, transition_dict):
        """_summary_

        :param transition_dict: {'states':..., 'actions':, 'next_states', 'rewards', 'dones'}
        :type transition_dict: _type_
        """

        # 一个batch的数据
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).unsqueeze(1).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        

        q_values = self.q_net(states).gather(1, actions)  # 通过action索引dim = 0的Q值
        # 下个状态的最大Q值
        # import pdb; pdb.set_trace()
        max_next_q_values = self.target_q_net(next_states.unsqueeze(1)).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values # TD误差目标

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1