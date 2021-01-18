# _*_ coding utf-8 _*_
"""
@File : DDPG.py
@Author: yxwang
@Date : 2021/1/12
@Desc :
"""

import tensorflow as tf
from tensorflow import keras
import huskarl as hk
import gym

if __name__ == "__main__":
    # Setup Block_chain environment
    create_env = lambda: gym.make('BlockChain-v0')
    dummy_env = create_env()

    action_nodes_size = dummy_env.n_of_nodes
    action_size = action_nodes_size + 3
    state_shape = (dummy_env.n_of_block_producer * 2 + 3,)

    # print("action_size = {}".format(action_size))
    # print("action_nodes_size = {} 总结点数".format(action_nodes_size))
    # print("state_shape = {}".format(state_shape))

    # Build a simple actor model which is the function estimator of action according to state
    inputs = tf.keras.Input(shape=state_shape, name='state_input')
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x1 = tf.keras.layers.Dense(64, activation='relu')(x)

    output1 = tf.keras.layers.Dense(action_nodes_size, activation='sigmoid')(x1)
    x2 = tf.keras.layers.Dense(32, activation='relu')(x1)
    x2 = tf.keras.layers.Dense(16, activation='relu')(x2)

    output2 = tf.keras.layers.Dense(1, activation='sigmoid')(x2)

    x3 = tf.keras.layers.Dense(32, activation='relu')(x1)
    x3 = tf.keras.layers.Dense(16, activation='relu')(x3)
    output3 = tf.keras.layers.Dense(1, activation='sigmoid')(x3)

    x4 = tf.keras.layers.Dense(32, activation='relu')(x1)
    x4 = tf.keras.layers.Dense(16, activation='relu')(x4)
    output4 = tf.keras.layers.Dense(1, activation='sigmoid')(x4)

    output = tf.concat([output1, output2, output3, output4], axis=1)
    actor = tf.keras.Model(inputs=inputs, outputs=output)

    # Build a simple critic model which is the function estimator of value
    action_input = tf.keras.Input(shape=(action_size,), name='action_input')
    state_input = tf.keras.Input(shape=state_shape, name='state_input')
    x = tf.keras.layers.Concatenate()([action_input, state_input])
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='linear')(x)
    critic = tf.keras.Model(inputs=[action_input, state_input], outputs=x)

    # Create Deep Deterministic Policy Gradient agent
    agent = hk.agent.DDPG(actor=actor, critic=critic, nsteps=1)

    # Create simulation, train and then test
    sim = hk.Simulation(create_env, agent)
    sim.train(max_steps=100, visualize=False, plot=None)
