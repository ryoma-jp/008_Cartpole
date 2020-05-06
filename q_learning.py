#! -*- coding: utf-8 -*-

"""
  [q_learning]
    python3 q_learning.py --help
"""

#---------------------------------
# モジュールのインポート
#---------------------------------
import sys
import argparse
import gym

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#---------------------------------
# メモ
#---------------------------------
"""
■動的にグラフを更新するサンプル
	https://blog.amedama.jp/entry/2018/07/13/001155
"""

#---------------------------------
# 定数定義
#---------------------------------
DIGIT_NUM = 16

#---------------------------------
# 関数
#---------------------------------

"""
  関数名: ArgParser
  説明：引数を解析して値を取得する
"""
def ArgParser():
	parser = argparse.ArgumentParser(description='DQN')
	
	# --- 引数を追加 ---
#	parser.add_argument('--str', dest='str', type=str, help='この引数の説明', required=True)
#	parser.add_argument('--int', dest='int', type=int, default=-1, help='この引数の説明', required=True)
#	parser.add_argument('--flag', dest='flag', action='store_true', help='この引数の説明', required=True)
	
	args = parser.parse_args()
	
	return args

"""
  関数名: DigitizeState
  説明：Cartpoleのパラメータをデジタル化する
"""
def DigitizeState(observation):
	def bins(clip_min, clip_max, num):
		return np.linspace(clip_min, clip_max, num + 1)[1:-1]
	
	cart_pos, cart_vel, pole_angle, pole_vel = observation
	digitized = \
		np.digitize(cart_pos, bins=bins(-2.4, 2.4, DIGIT_NUM)) * DIGIT_NUM**0 + \
		np.digitize(cart_vel, bins=bins(-3.0, 3.0, DIGIT_NUM)) * DIGIT_NUM**1 + \
		np.digitize(pole_angle, bins=bins(-0.5, 0.5, DIGIT_NUM)) * DIGIT_NUM**2 + \
		np.digitize(pole_vel, bins=bins(-2.0, 2.0, DIGIT_NUM)) * DIGIT_NUM**3
	
	return digitized
	
#---------------------------------
# クラス
#---------------------------------
class Q_Learning():
	# --- Qテーブルの初期化 ---
	def _init_qtable(self, shape):
		self.qtable = np.random.rand(np.prod(shape))
		self.qtable = self.qtable.reshape(shape)
		
		print(self.qtable)
		print(self.qtable.shape)
		print(shape)
		
		return
	
	# --- コンストラクタ ---
	def __init__(self, param_shape, fix_seed=True):
		if (fix_seed):
			print('random seed = 1234')
			random.seed(1234)
			np.random.seed(seed=1234)
		self._init_qtable(param_shape)
		return
	
	# --- 行動を決定する (ε-greedy法) ---
	def get_action(self, next_state, episode):
		epsilon = 0.5 * (1 / (episode + 1))
		if epsilon <= np.random.uniform(0, 1):
			next_action = np.argmax(self.qtable[next_state])
		else:
			next_action = np.random.choice([0, 1])
		return next_action
	
	# --- Qテーブル更新 ---
	def update_qtable(self, state, action, reward, next_state):
		gamma = 0.99
		alpha = 0.5
		next_Max_Q=max(self.qtable[next_state][0], self.qtable[next_state][1])
		self.qtable[state, action] = (1 - alpha) * self.qtable[state, action] + \
										alpha * (reward + gamma * next_Max_Q)
		
		return
		
	
#---------------------------------
# メイン処理
#---------------------------------
if __name__ == '__main__':
	# --- 引数処理 ---
	args = ArgParser()
	
	# --- Q学習オブジェクト作成 ---
	q_learning = Q_Learning([DIGIT_NUM**4, 2])
	env = gym.make('CartPole-v0')
	
	print('[Spec]')
	print(' * Observation: (Cart Position, Cart Velocity, Pole Anble, Pole Velocity At Tip)')
	print(' * Actions: 0-Push cart to the left, 1-Push cart to the right')
	
	log_header = ['episode', 'iter', 'episode_reward']
	log_data = None
	for episode in range(2000):
		observation = env.reset()
		observation_digitized = DigitizeState(observation)
		episode_reward = 0
		
		for t in range(1000):  
			env.render()
			
#			action = env.action_space.sample()
			
#			sum_obs = np.sum(observation)
#			if (sum_obs > 0):
#				action = 1
#			else:
#				action = 0
			
#			if (observation[2] > 0):
#				action = 1
#			else:
#				action = 0
			
			action = q_learning.get_action(observation_digitized, episode)
			observation, reward, done, info = env.step(action)
			
			if (done):
				if (t < 195):
					# 失敗
					reward = -200
				else:
					# 成功
					reward = 1
			else:
				reward = 1
			episode_reward += reward
			
			# Qテーブル更新
			next_observation_digitized = DigitizeState(observation)
			q_learning.update_qtable(observation_digitized, action, reward, next_observation_digitized)
			observation_digitized = next_observation_digitized
			
			if (done):
				print('Episode {} Done: t={}, episode_reward={}'.format(episode, t, episode_reward))
				if (log_data is None):
					log_data = np.array([episode, t, episode_reward])
				else:
					log_data = np.vstack((log_data, [episode, t, episode_reward]))
				break
				
	env.close()
	
	pd.DataFrame(log_data).to_csv('log.csv', header=log_header, index=False)
	
	fig = plt.figure(0)
	plt.plot(log_data[:, 0], log_data[:, 2])
	plt.savefig('log_graph.png')
	plt.show()
	plt.close(fig)
	
	
