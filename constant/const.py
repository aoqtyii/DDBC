# _*_ coding utf-8 _*_
"""
@File : const.py
@Author: yxwang
@Date : 2021/1/13
@Desc :
"""
import numpy as np


class _const:
    class ConstError(TypeError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError("Can't rebind const (%s)" % name)
        self.__dict__[name] = value


# sys.modules[__name__] = _const()
constInstance = _const()
constInstance.AREA_OF_NODES = (1000, 1000)  # 将节点散落的区域假定为规整矩形
constInstance.N_OF_NODES = 40  # 所有的IIoT节点，N，其中包括block_producer
constInstance.N_OF_BLOCK_PRODUCER = 21  # 常选择21，block_producer的个数，暂时将生产者数量固定
constInstance.AVERAGE_TRANSACTION_SIZE = 200  # 200B
# ---------------------------------------------------------
# 权益及算力的分配应当服从高斯分布
# STAKE_OF_NODES = (10, 50)  # 对不同的节点权益分配,范围(0, 1)
constInstance.STAKE_OF_NODES = (0, 1)  # 对不同的节点权益分配,范围(0, 1)
constInstance.POWER_OF_NODES = (0, 1)
# ---------------------------------------------------------
constInstance.COMPUTING_RESOURCE_OF_NODE = \
    np.ones(constInstance.N_OF_NODES, ) * 10  # 暂定为10~30GHz，(generating # MACs和verifying MACs)
constInstance.BLOCK_SIZE_LIMIT = 8 * 1024  # 最大区块的大小限制设定为8M，为统一单位定为B
constInstance.MAX_BLOCK_INTERVAL = 10  # 最大区块打包间隔时间10s
constInstance.ETA_S = 0.2
constInstance.ETA_L = 0.3  # 基尼系数中对去中心化程度的最大限制设定为0.2与0.3
constInstance.BATCH_SIZE = 3  # 如在PBFT算法中，一个massage中，primary节点处理的request
