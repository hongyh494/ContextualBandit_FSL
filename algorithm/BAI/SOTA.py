import numpy as np
import matplotlib.pyplot as plt
import random
from algorithm.BAI.classic import Solver

import torch
import torch.nn as nn
import torch.optim as optim

class PPONetwork(nn.Module):
    """PPO策略网络和价值网络"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.actor(x), self.critic(x)

class PPOSolver(Solver):
    """PPO算法实现"""
    def __init__(self, bandit, env=None, gamma=0.99, clip_eps=0.2, lr=3e-4, 
                 batch_size=64, epochs=4, **kwargs):
        super().__init__(bandit, env, **kwargs)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        
        # 检查是否有GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络和优化器
        self.net = PPONetwork(1, self.bandit.K).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.memory = []
    
    def get_state(self):
        # TODO: 这里可以根据需要定义状态
        return torch.FloatTensor([self.bandit.loss_state]).to(self.device)
    
    def run_one_step(self):
        state = self.get_state()
        
        # 采样动作
        probs, _ = self.net(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        
        # 执行并保存轨迹
        reward = self.bandit.get_reward(action)
        self.memory.append((state, action, reward))
        
        # 当积累足够经验后更新
        if len(self.memory) >= self.batch_size:
            self.update()
            self.memory = []
        return action
    
    def update(self):
        """PPO更新步骤"""
        states, actions, rewards = zip(*self.memory)
        states = torch.stack(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        # 计算折扣回报
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 标准化回报
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # 旧概率
        old_probs, old_values = self.net(states)
        old_probs = old_probs.detach()
        old_log_probs = torch.log(old_probs.gather(1, actions.unsqueeze(1))).squeeze()
        
        # 多epoch优化
        for _ in range(self.epochs):
            new_probs, values = self.net(states)
            new_log_probs = torch.log(new_probs.gather(1, actions.unsqueeze(1))).squeeze()
            
            # 计算比率和优势
            ratios = torch.exp(new_log_probs - old_log_probs)
            advantages = returns - values.squeeze().detach()
            
            # 裁剪目标函数
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_eps, 1+self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 价值函数损失
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # 总损失
            loss = actor_loss + 0.5 * critic_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


# 0409 中位数算法
class EpsilonGreedyMedian(Solver):
    """
    Epsilon Greedy算法
    """

    def __init__(self, bandit, env=None, epsilon=0.1, init_mu=1.0, **kwargs):
        """
        :param bandit: MAB模型对象
        :param env: 环境对象
        :param epsilon: 探索概率
        :param init_mu: 初始化的均值
        :param kwargs: 其他参数
        """
        super(EpsilonGreedyMedian, self).__init__(bandit, env, **kwargs)
        self.epsilon = epsilon  # 超参数-exploration概率
        self.estimated_median = np.array([init_mu] * self.bandit.K)  # 初始化每个臂的均值估计
   
    def run_one_step(self):
        """
        选择动作手臂
        更新counts和estimated_median, 历史连输J(老虎机内部记录), 返回被拉动的臂,
        :return: 被拉动的臂
        """
        if random.random() < self.epsilon:
            action = random.randint(0, self.bandit.K - 1)  # 随机选择一个臂
        else:
            action = np.argmax(self.estimated_median)  # 选择当前估计奖励最大的臂
        
        # 摇臂得到奖励 可能来自mu可能来自J, 更新了J
        reward = self.bandit.get_reward(action)
        # 更新self.rewards
        self.rewards[action].append(reward)

        # 更新估计中位数估计！！
        # self.estimated_median[action] += (reward - self.estimated_median[action]) / self.counts[action]
        self.estimated_median[action] = np.median(self.rewards[action])
        return action
    

class UCBMedian(Solver):
    """
    UCB算法
    """

    def __init__(self, bandit, env=None, coef: float=2, init_mu: float=1.0, **kwargs):
        """
        :param bandit: MAB模型对象
        :param env: 环境对象
        :param coef: 超参数
        :param kwargs: 其他参数
        """
        super(UCBMedian, self).__init__(bandit, env, **kwargs)
        self.total_count = 0  # 总的拉动次数-估计上界有用
        self.coef = coef  # 超参数-不确定性Upper Bounds的比重
        self.estimated_median = np.array([init_mu] * self.bandit.K)  # 初始化每个臂的均值估计

    def run_one_step(self):
        """
        选择动作手臂
        更新counts和estimated_median, 历史连输J(老虎机内部记录), 返回被拉动的臂,
        :return: 被拉动的臂
        """

        self.total_count += 1  # 更新总的拉动次数

        if np.min(self.counts) == 0:
            action = np.argmin(self.counts)  # 如果有未被拉动的臂，选择其中一个
        else:
            action = np.argmax(self.estimated_median + self.coef * np.sqrt(np.log(self.total_count) / (2 * self.counts + 1)))  # UCB公式
        
        # 摇臂得到奖励 可能来自mu可能来自J, 更新了J
        reward = self.bandit.get_reward(action)
        # 更新self.rewards
        self.rewards[action].append(reward)
        
        # 更新中位数估计！！ plus uncertainty
        # self.estimated_median[action] += 1./ (self.counts[action] + 1) * (reward - self.estimated_median[action])
        self.estimated_median[action] = np.median(self.rewards[action])
        
        return action