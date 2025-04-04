import numpy as np
import matplotlib.pyplot as plt
from utils.calc_p import asymptotic_expectation_zero, objective
from scipy.optimize import fsolve
from numba import jit, prange
import random


# Dynamic Futurity @ single arm ver.
class DynamicSlotMachine:
    def __init__(self, p, J_initial, num_steps=1000, J_distribution=None):
        """
        初始化单臂老虎机模型
        
        :param p: 成功奖励的概率
        :param J_initial: 初始的J值
        :param num_steps: 每条轨迹的步数
        :param J_distribution: J值的分布，用于补偿后更新J（默认为None，表示J是固定的）
        """
        self.p = p  # 成功奖励的概率
        self.J = J_initial  # 当前的J值
        self.num_steps = num_steps  # 每条轨迹的步数
        self.J_distribution = J_distribution  # 用于更新J的分布（可选）

    def update_J(self):
        """
        更新J值。如果提供了J的分布，则从中抽样生成新的J值。
        """
        if self.J_distribution:
            if isinstance(self.J_distribution, list):  # 离散分布
                # 注，若J为单个值，则也需要传入一个[1]的概率分布list
                self.J = random.choices(self.J_distribution[0], self.J_distribution[1])[0]
            elif isinstance(self.J_distribution, tuple):  # 连续分布
                dist_type, *params = self.J_distribution
                if dist_type == 'chisquare':
                    self.J = np.random.chisquare(*params)
        else:
            pass
        # 同时也要更新p值，以确保公平性
        # self.p = fsolve(objective, self.p, args=(self.J,))[0]  # initial guess is self.p
        self.p = fsolve(objective, self.p, args=(self.J,))[0]  # initial guess is 0.5?

    def simulate(self):
        """
        模拟一个玩家玩单臂老虎机的过程
        
        :return: 玩家奖励轨迹
        """
        rewards = []
        consecutive_losses = 0  # 记录连续失败次数
        total_reward = 0  # 当前的总奖励

        # 进行 num_steps 次抽奖
        for step in range(self.num_steps):
            # 根据概率p生成奖励 -1 或 1
            reward = 1 if np.random.rand() < self.p else -1

            # 如果是负奖励，增加连续失败次数
            if reward == -1:
                consecutive_losses += 1
            else:
                consecutive_losses = 0

            # 如果连续失败次数达到 J，则补偿
            if consecutive_losses == self.J:
                reward += self.J  # 添加补偿
                self.update_J()  # 触发补偿后更新J的值

            # 更新总奖励
            total_reward += reward
            rewards.append(total_reward)

        return np.array(rewards)

# 并行模拟多个轨迹
@jit(nopython=True, parallel=True)
def parallel_simulate(num_paths, num_steps, p, J_initial, J_distribution):
    all_rewards = np.zeros((num_paths, num_steps))

    for i in prange(num_paths):
        model = DynamicSlotMachine(p, J_initial, num_steps, J_distribution)
        all_rewards[i] = model.simulate()

    return all_rewards


if __name__ == "__main__":

    # 假设我们已知 J 值
    J = 3
    p_initial_guess = 0.5
    p_solution = fsolve(objective, p_initial_guess, args=(J,))
    p = p_solution[0]

    print(f"Calculated p: {p}")

    # Step 3: 设置参数
    J_initial = 3  # 初始的J值
    num_steps = 1000  # 每条轨迹的步数
    num_paths = 100  # 模拟的轨迹数

    # J_distribution可以是离散分布或连续分布
    # J 离散分布
    J_distribution = ([5], [1])  # 例如4的概率是0.2, 5的概率是0.3...
    # J_distribution = ([4, 6], [0.6, 0.4])  # 例如4的概率是0.2, 5的概率是0.3...
    # J_distribution = ([4, 5, 7, 10], [0.2, 0.3, 0.3, 0.2])  # 例如4的概率是0.2, 5的概率是0.3...

    # 或者使用连续分布，比如卡方分布
    # J_distribution = ('chisquare', 2)  # 自由度为2的卡方分布

    # Step 5: 运行模拟并计算均值
    all_rewards = parallel_simulate(num_paths, num_steps, p, J_initial, J_distribution)

    # 计算所有轨迹的均值
    mean_rewards = np.mean(all_rewards, axis=0)


    # 绘制均值轨迹
    plt.plot(mean_rewards)
    plt.title(f"Average Reward for {num_paths} Paths (p={p}, Initial J={J_initial})")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.show()
