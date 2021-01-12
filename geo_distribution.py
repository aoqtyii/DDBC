# _*_ coding utf-8 _*_
"""
@File : geo_distribution.py
@Author: yxwang
@Date : 2021/1/12
@Desc :
"""
import numpy as np


class geo_distribution:
    def __init__(self, area, X, Y):
        self.area = area
        self.X = X
        self.Y = Y

    def get_lambda0(self, M):
        if self.area <= 1:
            n = 4
        elif self.area <= 10:
            n = 16
        elif self.area <= 100:
            n = 36
        elif self.area <= 1000:
            n = 64
        else:
            n = 100
        sqrt_n = np.int(np.sqrt(n))
        lambda0 = np.ones(n)
        X = np.asarray(
            [np.float(-self.X) / 2 + i * np.float(self.X) / 2 for i in range(sqrt_n + 1)])
        Y = np.asarray(
            [np.float(self.Y) / 2 - i * np.float(self.Y) / 2 for i in range(sqrt_n + 1)])
        X_range = np.array([[0 * i, 0] for i in range(sqrt_n)])
        Y_range = np.array([[0 * i, 0] for i in range(sqrt_n)])

        for i in range(sqrt_n):
            X_range[i] = [X[i], X[i + 1]]
            Y_range[i] = [Y[i], Y[i + 1]]

        flg_x = 0
        flg_y = 0
        XY = np.asarray([[0, 0, 0, 0 * i] for i in range(n)])
        for i in range(n):
            if i != 0 and i % sqrt_n == 0:
                flg_y += 1
                flg_x = 0
            XY[i] = np.concatenate((X_range[flg_x], Y_range[flg_y]))
            flg_x += 1

        for i in range(n):
            lambda0[i] = \
                np.where((XY[i][0] < M[:, 0]) & (M[:, 0] < XY[i][1]) & (M[:, 1] < XY[i][2]) & (XY[i][3] < M[:, 1]))[
                    0].size
        return lambda0, n
