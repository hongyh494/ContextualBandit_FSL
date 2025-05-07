import numpy as np
import matplotlib.pyplot as plt
import random

class Solver:
    """
    MAB识别算法基本框架, 基类
    """

    def __init__(self, bandit, env=None, **kwargs):
        """
        :param bandit: MAB模型对象
        :param env: 环境对象
        :param kwargs: 其他参数
        """
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 记录每个臂被拉动的次数
        self.regret = 0. # 记录累积遗憾值
        self.actions = []  # 维护一个列表, 记录每一步的动作选择
        self.regrets = []  # 维护一个列表, 记录每一步的累计regret
        self.rewards = [[] for _ in range(self.bandit.K)]  # 维护一个列表, 记录每一步的奖励以及归属的臂
        self.env = env  # 当前状态，历史累计连输
        self.kwargs = kwargs

    def update_regret(self, k):
        """
        更新累计遗憾值和每一步的累计regret列表
        :param k: 被拉动的臂
        :return: None
        """
        # self.regret += self.bandit.get_optimal_reward() - self.bandit.get_reward(k)  # 这个是实际观测的，好像不是mu?
        self.regret += self.bandit.get_optimal_mu() - self.bandit.get_mu_p()[k]  # 这个是给定的mu        
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 每个子算法的class具体实现
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def run(self, num_steps):
        """
        单条path, 运行一定次数
        :param num_steps: 单条path运行的步数
        :return: None
        """
        for step in range(num_steps):
            action = self.run_one_step()  # 选择动作手臂
            self.counts[action] += 1  # 更新被拉动的臂的次数
            self.update_regret(action)  # 更新遗憾值 在给定\mu的情形下
            self.actions.append(action)


class EpsilonGreedy(Solver):
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
        super(EpsilonGreedy, self).__init__(bandit, env, **kwargs)
        self.epsilon = epsilon  # 超参数-exploration概率
        self.estimated_mu = np.array([init_mu] * self.bandit.K)  # 初始化每个臂的均值估计
   
    def run_one_step(self):
        """
        选择动作手臂
        更新counts和estimated_mu, 历史连输J(老虎机内部记录), 返回被拉动的臂,
        :return: 被拉动的臂
        """
        if random.random() < self.epsilon:
            action = random.randint(0, self.bandit.K - 1)  # 随机选择一个臂
        else:
            action = np.argmax(self.estimated_mu)  # 选择当前估计奖励最大的臂
        
        # 摇臂得到奖励 可能来自mu可能来自J, 更新了J
        reward = self.bandit.get_reward(action)

        # 更新均值估计
        # self.estimated_mu[action] += (reward - self.estimated_mu[action]) / self.counts[action]
        self.estimated_mu[action] += 1./ (self.counts[action] + 1) * (reward - self.estimated_mu[action])
        return action
    

class UCB(Solver):
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
        super(UCB, self).__init__(bandit, env, **kwargs)
        self.total_count = 0  # 总的拉动次数-估计上界有用
        self.coef = coef  # 超参数-不确定性Upper Bounds的比重
        self.estimated_mu = np.array([init_mu] * self.bandit.K)  # 初始化每个臂的均值估计

    def run_one_step(self):
        """
        选择动作手臂
        更新counts和estimated_mu, 历史连输J(老虎机内部记录), 返回被拉动的臂,
        :return: 被拉动的臂
        """

        self.total_count += 1  # 更新总的拉动次数

        if np.min(self.counts) == 0:
            action = np.argmin(self.counts)  # 如果有未被拉动的臂，选择其中一个
        else:
            action = np.argmax(self.estimated_mu + self.coef * np.sqrt(np.log(self.total_count) / (2 * self.counts + 1)))  # UCB公式
        
        # 摇臂得到奖励 可能来自mu可能来自J, 更新了J
        reward = self.bandit.get_reward(action)

        # 更新均值估计
        self.estimated_mu[action] += 1./ (self.counts[action] + 1) * (reward - self.estimated_mu[action])
        
        return action
    

class ThompsonSampling(Solver):
    """
    Thompson Sampling算法
    """

    def __init__(self, bandit, env=None, **kwargs):
        """
        :param bandit: MAB模型对象
        :param env: 环境对象
        :param kwargs: 其他参数
        """
        super(ThompsonSampling, self).__init__(bandit, env, **kwargs)
        self.alpha = np.ones(self.bandit.K)  # Beta分布的alpha参数
        self.beta = np.ones(self.bandit.K)  # Beta分布的beta参数

    def run_one_step(self):
        """
        选择动作手臂
        更新counts和estimated_mu, 历史连输J(老虎机内部记录), 返回被拉动的臂,
        :return: 被拉动的臂
        """
        theta = np.random.beta(self.alpha, self.beta)  # 从Beta分布中抽取样本
        action = np.argmax(theta)  # 选择当前估计奖励最大的臂
        
        # 摇臂得到奖励 可能来自mu可能来自J, 更新了J
        reward = self.bandit.get_reward(action)

        # 更新Beta分布的参数
        self.alpha[action] += reward
        self.beta[action] += (1 - reward)
        
        return action
    
class GaussianTS(Solver):
    """
    Gaussian Thompson Sampling算法
    """

    def __init__(self, bandit, env=None, **kwargs):
        """
        :param bandit: MAB模型对象
        :param env: 环境对象
        :param kwargs: 其他参数
        """
        super(GaussianTS, self).__init__(bandit, env, **kwargs)
        self.mu = np.zeros(self.bandit.K)  # 均值
        self.sigma = np.ones(self.bandit.K)  # 标准差

    def run_one_step(self):
        """
        选择动作手臂
        更新counts和estimated_mu, 历史连输J(老虎机内部记录), 返回被拉动的臂,
        :return: 被拉动的臂
        """
        theta = np.random.normal(self.mu, self.sigma)  # 从正态分布中抽取样本
        action = np.argmax(theta)  # 选择当前估计奖励最大的臂
        
        # 摇臂得到奖励 可能来自mu可能来自J, 更新了J
        reward = self.bandit.get_reward(action)

        # 更新均值和标准差
        self.mu[action] += (reward - self.mu[action]) / (self.counts[action] + 1)
        self.sigma[action] = np.sqrt((self.sigma[action]**2 * self.counts[action] + (reward - self.mu[action])**2) / (self.counts[action] + 1))
        
        return action
    
class ExponentialTS(Solver):
    """
    Exponential Thompson Sampling算法 (时间间隔型奖励，如用户停留时间)
    """
    def __init__(self, bandit, env=None, **kwargs):
        super(ExponentialTS, self).__init__(bandit, env, **kwargs)
        self.alpha = np.ones(self.bandit.K)  # Gamma分布形状参数 (先验α=1)
        self.beta = np.ones(self.bandit.K)   # Gamma分布逆尺度参数 (先验β=1)

    def run_one_step(self):
        # 从Gamma后验中采样λ (指数分布的参数)
        lambda_samples = np.random.gamma(self.alpha, 1 / self.beta)
        action = np.argmin(lambda_samples)  # 注意：指数分布越小越好
        
        reward = self.bandit.get_reward(action)
        
        # 更新Gamma后验参数
        self.alpha[action] += 1
        self.beta[action] += reward
        
        return action

class PoissonTS(Solver):
    """
    Poisson Thompson Sampling算法 (计数型奖励，如点击次数)
    """
    def __init__(self, bandit, env=None, **kwargs):
        super(PoissonTS, self).__init__(bandit, env, **kwargs)
        self.alpha = np.ones(self.bandit.K)  # Gamma分布形状参数 (先验α=1)
        self.beta = np.ones(self.bandit.K)   # Gamma分布逆尺度参数 (先验β=1)

    def run_one_step(self):
        # 从Gamma后验中采样λ (泊松分布的参数)
        lambda_samples = np.random.gamma(self.alpha, 1 / self.beta)
        action = np.argmax(lambda_samples)
        
        reward = self.bandit.get_reward(action)
        
        # 更新Gamma后验参数
        self.alpha[action] += reward
        self.beta[action] += 1
        
        return action
    
def plot_regret_single(ax, solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        ax.plot(time_list, solver.regrets, label=solver_names[idx])
    ax.set_xlabel('Time steps')
    ax.set_ylabel('Cumulative regrets')
    ax.set_title('%d-armed bandit' % solvers[0].bandit.K)
    ax.legend()

def plot_mu_alpha_single(ax, solvers, solver_names, alpha_list):
    """ 生成不同alpha的不同估计的图像
    """
    for solver in solvers:
        for arm in range(solver.bandit.K):
            for idx, solver in enumerate(solvers):
                estimated_alpha_mu = np.array(solver.estimated_alpha_mu)  # 转换为numpy数组
                ax.plot(alpha_list, estimated_alpha_mu[arm, :], label=f"{solver_names[idx]} - Arm {arm}")
                # 标注数值点
                # for alpha, mu in zip(alpha_list, estimated_alpha_mu[arm, :]):
                ax.scatter(alpha_list, estimated_alpha_mu[arm, :],  s=10)  # 用红色空心圆标记点
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Estimated Mean')
    ax.set_title('Estimated Mean vs Alpha for Different Arms')
    ax.legend()

def plot_p_estimate_single(ax, solvers, solver_names, t_list):
    """ 生成不同alpha的不同估计的图像
    """
    for solver in solvers:
        for arm in range(solver.bandit.K):
            for idx, solver in enumerate(solvers):
                estimated_hit = np.array(solver.estimated_hit)  # 转换为numpy数组
                ax.plot(t_list[10:], estimated_hit[arm, :][10:], label=f"{solver_names[idx]} - Arm {arm}")
                # 标注数值点
                # for alpha, mu in zip(alpha_list, estimated_alpha_mu[arm, :]):
                ax.scatter(t_list[10:], estimated_hit[arm, :][10:], s=10)  # 用红色空心圆标记点
    ax.set_xlabel('pullss')
    ax.set_ylabel('Estimated P')
    ax.set_title('Estimated P vs pulls for Different Arms')
    ax.legend()
    

def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    # plt.show()

if __name__ == "__main__":
    
    # 测试代码 BernoulliBandit 无J版
    class Bandit:
        def __init__(self, K):
            self.K = K
            self.mu = np.random.uniform(size=K)  # 随机生成K个臂的均值
            self.J = np.zeros(K)  # 初始化J为0

        def get_reward(self, action):
            reward = np.random.binomial(1, self.mu[action])  # 根据均值生成奖励
            return reward

        def get_optimal_mu(self):
            return np.max(self.mu)

        def get_mu(self):
            return self.mu

    bandit_10_arm = Bandit(10)
    # bandit_10_arm.mu = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])/50  # 设置均值

    # 测试EpsilonGreedy算法

    random.seed(1)
    epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.1)
    epsilon_greedy_solver.run(5000)
    print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
    plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])


    # 测试UCB算法
    random.seed(1)
    ucb_solver = UCB(bandit_10_arm, coef=1)
    ucb_solver.run(5000)
    print('UCB算法的累积懊悔为：', ucb_solver.regret)
    plot_results([ucb_solver], ["UCB"])

    # 测试ThompsonSampling算法
    random.seed(1)
    thompson_solver = ThompsonSampling(bandit_10_arm)
    thompson_solver.run(5000)
    print('Thompson Sampling算法的累积懊悔为：', thompson_solver.regret)
    plot_results([thompson_solver], ["ThompsonSampling"])

    plt.show()