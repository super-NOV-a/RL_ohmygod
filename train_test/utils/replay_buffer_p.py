import torch
import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, args):
        self.device = args.device
        self.N_drones = args.N_drones
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.share_prob = args.share_prob
        self.count = 0
        self.current_size = 0
        self.buffer_obs_n, self.buffer_a_n, self.buffer_r_n, self.buffer_s_next_n, self.buffer_done_n = [], [], [], [], []
        self.priorities = []

        for agent_id in range(self.N_drones):
            self.buffer_obs_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_a_n.append(np.empty((self.buffer_size, args.action_dim_n[agent_id])))
            self.buffer_r_n.append(np.empty((self.buffer_size, 1)))
            self.buffer_s_next_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_done_n.append(np.empty((self.buffer_size, 1)))
            self.priorities.append(np.zeros(self.buffer_size))

    def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n):
        for agent_id in range(self.N_drones):
            target_agent_id = agent_id
            if np.random.rand() < self.share_prob:
                target_agent_id = np.random.choice(self.N_drones)
            self.buffer_obs_n[target_agent_id][self.count] = obs_n[agent_id]
            self.buffer_a_n[target_agent_id][self.count] = a_n[agent_id]
            self.buffer_r_n[target_agent_id][self.count] = r_n[agent_id]
            self.buffer_s_next_n[target_agent_id][self.count] = obs_next_n[agent_id]
            self.buffer_done_n[target_agent_id][self.count] = done_n[agent_id]
            self.priorities[target_agent_id][self.count] = max(
                self.priorities[target_agent_id]) + 1  # Set initial priority

        self.count = (self.count + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self):
        if self.current_size < self.batch_size:
            raise ValueError("Not enough samples in the replay buffer to sample a batch.")

        # 使用当前有效的优先级生成概率
        priorities = np.concatenate(self.priorities)[:self.current_size]
        probabilities = priorities ** 0.6  # 温度系数，控制优先级影响
        probabilities /= probabilities.sum()  # 归一化

        # 确保根据 probabilities 进行采样
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False, p=probabilities)

        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = [], [], [], [], []

        for agent_id in range(self.N_drones):
            batch_obs_n.append(torch.tensor(self.buffer_obs_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_a_n.append(torch.tensor(self.buffer_a_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_r_n.append(torch.tensor(self.buffer_r_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_obs_next_n.append(
                torch.tensor(self.buffer_s_next_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_done_n.append(
                torch.tensor(self.buffer_done_n[agent_id][index], dtype=torch.float).to(self.device))

        return batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n

