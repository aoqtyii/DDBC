# _*_ coding utf-8 _*_
"""
@File : state_to_array.py
@Author: yxwang
@Date : 2021/1/13
@Desc :
"""

import numpy as np


class state_to_array:
    def __init__(self, state):
        self.state = state

    def to_array(self):
        s = []
        s += (list(v.reshape(-1)) for k, v in self.state.items())
        s_array = []

        for i in range(4):
            for j in range(len(s[i])):
                if i == 0:
                    s_array.append(s[i][j])
                elif i == 1:
                    if s[i][j] < 0:
                        s_array.append(s[i][j] / 500)
                    else:
                        s_array.append(s[i][j] / 500)
                elif i == 2:
                    s_array.append(s[i][j])
                else:
                    s_array.append((s[i][j] - 10) / 90)
        return np.array(s_array)
