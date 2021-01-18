# _*_ coding utf-8 _*_
"""
@File : action_to_dict.py
@Author: yxwang
@Date : 2021/1/13
@Desc :
"""
import numpy as np
from constant.const import constInstance as ci


class action_to_dict:
    def __init__(self, action):
        self.action = action

    def to_dict(self):
        # 传递的action参数是通过神经网络得到的输出，由于最后一层使用sigmoid function使得输出在(0,1)之间
        l = []
        keyList = ['no_block_producer', 'no_consensus_algorithm', 'block_size', 'block_interval']
        valueList = []

        action_dict = {}
        for i in range(len(self.action)):
            if i < ci.N_OF_NODES:
                l.append(1 if self.action[i] > 0.5 else 0)
                if i == ci.N_OF_NODES - 1:
                    valueList.append(np.array(l))
                continue
            valueList.append(np.array(self.action[i]))

        for i in range(len(keyList)):
            action_dict[keyList[i]] = valueList[i]

        # 保证通过网络输出的action的取值在限制范围内
        action_dict['no_consensus_algorithm'] = int(action_dict['no_consensus_algorithm'] * 3) \
            if action_dict['no_consensus_algorithm'] is not None \
            else 0
        action_dict['block_size'] = action_dict['block_size'] * ci.BLOCK_SIZE_LIMIT+300 \
            if action_dict['block_size'] != 0 \
            else action_dict['block_size'] * ci.BLOCK_SIZE_LIMIT+300
        action_dict['block_interval'] = action_dict['block_interval'] * ci.MAX_BLOCK_INTERVAL+0.5 \
            if action_dict['block_interval'] != 0 \
            else (action_dict['block_interval'] + 0.001) * ci.MAX_BLOCK_INTERVAL + 0.5
        print(action_dict)
        return action_dict
