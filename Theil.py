# _*_ coding utf-8 _*_
"""
@File : Theil.py
@Author: yxwang
@Date : 2021/1/12
@Desc :
"""

import numpy as np


class Theil:
    def __init__(self, K, Tl_norm_C=0, Tl_norm_G=0, Tl_norm_L=0):
        self.Tl_norm_L = Tl_norm_L
        self.Tl_norm_G = Tl_norm_G
        self.Tl_norm_C = Tl_norm_C
        self.K = K

    def get_Theil_Computation(self, C):
        C_norm = np.sum(C) / self.K
        self.Tl_norm_C = np.sum(C / C_norm * np.log(C / C_norm)) / (self.K * np.log(self.K))
        return self.Tl_norm_C

    def get_Theil_Stake(self, G):
        G_norm = np.sum(G) / self.K
        self.Tl_norm_G = np.sum(G / G_norm * np.log(G / G_norm)) / (self.K * np.log(self.K))
        return self.Tl_norm_G

    # def get_Theil_Lambda(self, L, n, average_A):
    #     self.Tl_norm_L = np.sum(average_A * L * np.log(average_A * L * n / self.K)) / (self.K * np.log(n))
    #     return self.Tl_norm_L

    def get_Theil_Lambda(self, L, n):
        self.Tl_norm_L = np.sum(L * np.log(L * n / self.K)) / (self.K * np.log(n))
        return self.Tl_norm_L

    def get_Theil_norm(self, delta, Tl_norm_C, Tl_norm_G, Tl_norm_L):
        """
        :param delta: 共识算法类型
        :param Tl_norm_C: 计算能力去中心化程度
        :param Tl_norm_G: 权利去中心化程度
        :param Tl_norm_L: 地理位置去中心化
        :return: Tl_norm: 综合去中心化程度
        """
        if delta in [0, 1, 2]:  # PoW, DPoS, PoS
            alpha = 0.5
            beta = 0.5
            gamma = 0
        elif delta in [3, 4]:  # PBFT, Quorum
            alpha = 0
            beta = 0.5
            gamma = 0.5
        else:
            alpha = 0.3
            beta = 0.3
            gamma = 0.3
        return alpha * Tl_norm_C + beta * Tl_norm_L + gamma * Tl_norm_G
