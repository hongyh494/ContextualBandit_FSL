import numpy as np
from scipy.optimize import fsolve


# 定义渐进期望函数，确保它为零
def asymptotic_expectation_zero(p, J):
    # 期望 reward μ = 2p - 1
    mu = 2 * p - 1
    # 计算渐进期望 μ*
    numerator = p * (1 - p)**J
    denominator = 1 - (1 - p)**J
    asymptotic_mu_star = mu + J * numerator / denominator
    # 返回渐进期望 - 0（我们要求解这个值为零）
    return asymptotic_mu_star


# 定义目标函数，求解渐进期望为零时的p值
def objective(p, J):
    return asymptotic_expectation_zero(p, J)


def calculate_p(J, p_initial_guess=0.5):
    """p的initial guess在J<4的时候有讲究, 详情请见p_distribution.ipynb

    :param J: _description_
    :type J: _type_
    :param p_initial_guess: _description_, defaults to 0.5
    :type p_initial_guess: float, optional
    :return: _description_
    :rtype: _type_
    """
    p_solution = fsolve(objective, p_initial_guess, args=(J,))
    return p_solution[0]

if __name__ == "__main__":
    
    # 给定J值（例如J=2）
    J = 2

    # 初始猜测p值
    p_initial_guess = 0.5

    # 使用fsolve进行求解，使渐进期望为零
    p_solution = fsolve(objective, p_initial_guess, args=(J))

    print(f"Estimated p for J={J} (asymptotic expectation = 0): {p_solution[0]}")

