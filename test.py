# _*_ coding utf-8 _*_
"""
@File : test.py
@Author: yxwang
@Date : 2021/1/12
@Desc :
"""
import gym
env = gym.make('BlockChain-v0')
# reset的作用
env.reset()

for _ in range(3):
    action = env.action_space.sample()
    print(action)
    print("\n")
    print(env.step(action))
    print("\n")
