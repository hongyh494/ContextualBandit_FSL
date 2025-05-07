import numpy as np
import matplotlib.pyplot as plt
import random
from algorithm.BestStrategy.base import Solver

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
        # TODO: 这里可以根据需要定义状态 / [Ation , loss_state] 动作对，输出一个价值
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