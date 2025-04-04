import numpy as np
from scipy.optimize import fsolve

""" TODO
用来当J取Dynamic时，计算p的函数，保持\mu^*不变=0单臂平稳，同一个\mu^*的p值会有俩个解

:return: 两个p值
:rtype: [float, float]
"""

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