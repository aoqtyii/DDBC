# _*_ coding utf-8 _*_
"""
@File : geo_coordination.py
@Author: yxwang
@Date : 2021/1/12
@Desc :
"""

from collections import Counter

from scipy.optimize import minimize
import numpy as np
import random
import matplotlib.pyplot as plt


# 1. 2D inhomogeneous PPP分布
# 2. 对于非齐次泊松点过程的模拟，首先模拟一个均匀的泊松点过程，然后根据确定性函数适当地变换这些点
# 3. 模拟联合分布的随机变量的标准方法是使用马尔可夫链蒙特卡洛；应用MCMC方法就是简单地将随机点处理操作重复应用于所有点
#    将使用基于Thinning的通用但更简单的方法(Thinning是模拟非均匀泊松点过程的最简单，最通用的方法)

# plt.close('all')

class geographical_coordination:
    def __init__(self, xMin, xMax, yMin, yMax, n_of_nodes, num_Sim=1, s=0.7):
        self.xMin = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax
        self.xDelta = xMax - xMin
        self.yDelta = yMax - yMin
        self.xMin_norm = -1
        self.xMax_norm = 1
        self.yMin_norm = -1
        self.yMax_norm = 1
        self.xDelta_norm = self.xMax_norm - self.xMin_norm
        self.yDelta_norm = self.yMax_norm - self.yMin_norm
        self.areaTotal_norm = self.xDelta_norm * self.yDelta_norm
        self.areaTotal = self.xDelta * self.yDelta

        self.num_Sim = num_Sim
        self.s = s

        # 是指定区域范围内的所有节点数
        self.n_of_nodes = n_of_nodes

        self.resultsOpt = None
        self.lambdaNegMin = None
        self.lambdaMax = None
        self.numbPointsRetained = None
        self.numbPoints = None

        self.xxRetained = []
        self.yyRetained = []
        self.xxThinned = []
        self.yyThinned = []

    # 强度函数
    def fun_lambda(self, x, y):
        return 20 * np.exp(-(x ** 2 + y ** 2) / self.s ** 2)

    # 定义 thinning prob 函数
    def fun_p(self, x, y):
        return self.fun_lambda(x, y) / self.lambdaMax

    # 负lambda
    def fun_neg(self, x):
        return -self.fun_lambda(x[0], x[1])

    def geographical_coordinates(self):
        # xy0 = [(self.xMin + self.xMax) / 2, (self.yMin + self.yMax) / 2]
        xy0 = [(self.xMin_norm + self.xMax_norm) / 2, (self.yMin_norm + self.yMax_norm) / 2]

        # 找到最大的lambda值
        self.resultsOpt = minimize(self.fun_neg, xy0,
                                   bounds=((self.xMin_norm, self.xMax_norm), (self.yMin_norm, self.yMax_norm)))
        self.lambdaNegMin = self.resultsOpt.fun
        self.lambdaMax = -self.lambdaNegMin

        # thinning过后保留的点的数量
        self.numbPointsRetained = np.zeros(self.n_of_nodes)

        while True:
            # 模拟PPP
            # 泊松过程产生的点的数量
            self.numbPoints = np.random.poisson(self.areaTotal_norm * self.lambdaMax)
            # 获取泊松点, numbPoints > n_of_IIot points
            if self.numbPoints >= self.n_of_nodes:
                # 泊松点的坐标
                xx = np.random.uniform(0, self.xDelta_norm, (self.numbPoints, 1)) + self.xMin_norm
                yy = np.random.uniform(0, self.yDelta_norm, (self.numbPoints, 1)) + self.yMin_norm

                # 计算空间独立的thinning probabilities
                p = self.fun_p(xx, yy)

                # 为thinning生成伯努利变量
                booleRetained = np.random.uniform(0, 1, (self.numbPoints, 1)) < p

                num_of_generate_points = Counter(booleRetained.T[0]).get(True)
                if num_of_generate_points >= self.n_of_nodes:
                    self.xxRetained = xx[booleRetained]
                    self.yyRetained = yy[booleRetained]
                    break

        length = self.xxRetained.shape[0]
        sample_list = [i for i in range(length)]
        sample_list = random.sample(sample_list, self.n_of_nodes)
        sample_list = sorted(sample_list)
        self.xxRetained = self.xxRetained[sample_list]*self.xDelta/2
        self.yyRetained = self.yyRetained[sample_list]*self.yDelta/2
        # return self.xxRetained, self.yyRetained, self.s
        return self
