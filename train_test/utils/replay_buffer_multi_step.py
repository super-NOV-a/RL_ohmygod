import torch
import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, args):
        self.device = args.device
        self.N_drones = args.N_drones  # The number of agents
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.n_step = args.n_step  # N-step returns
        self.gamma = args.gamma  # Discount factor for n-step returns
        self.share_prob = args.share_prob
        self.count = 0
        self.current_size = 0
        self.buffer_obs_n, self.buffer_a_n, self.buffer_r_n, self.buffer_s_next_n, self.buffer_done_n = [], [], [], [], []
        self.store_prob=0.01

        # Initialize buffers
        for agent_id in range(self.N_drones):
            self.buffer_obs_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_a_n.append(np.empty((self.buffer_size, args.action_dim_n[agent_id])))
            self.buffer_r_n.append(np.empty((self.buffer_size, 1)))
            self.buffer_s_next_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_done_n.append(np.empty((self.buffer_size, 1)))

        # For storing n-step transitions
        self.n_step_buffer = [[] for _ in range(self.N_drones)]

    def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n):
        # Store single transition in n_step_buffer for each agent
        for agent_id in range(self.N_drones):
            target_agent_id = agent_id
            if np.random.rand() < self.share_prob:
                target_agent_id = np.random.choice(self.N_drones)

            # Append to n-step buffer
            transition = (obs_n[agent_id], a_n[agent_id], r_n[agent_id], obs_next_n[agent_id], done_n[agent_id])
            self.n_step_buffer[agent_id].append(transition)

            # If buffer is full for n-step, store the processed n-step transition
            if len(self.n_step_buffer[agent_id]) >= self.n_step:
                reward, next_state, done = self._get_n_step_info(self.n_step_buffer[agent_id], self.gamma)
                obs, action = self.n_step_buffer[agent_id][0][:2]  # The first transition's state and action
                save_probability = random.random()  # 生成0到1之间的随机数
                if save_probability < self.store_prob:  # self.store_prob 是一个设定的概率阈值
                    self._store_single_transition(agent_id, obs, action, reward, next_state, done)
                self._store_single_transition(target_agent_id, obs, action, reward, next_state, done)
                self.n_step_buffer[agent_id].pop(0)  # Remove the first transition to make room for new transitions

        # After processing all agents, update count and current_size
        self.count = (self.count + 1) % self.buffer_size  # Increment count after all agents are processed
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def _store_single_transition(self, agent_id, obs, action, reward, next_state, done):
        # Store the processed n-step transition into the main buffer
        self.buffer_obs_n[agent_id][self.count] = obs
        self.buffer_a_n[agent_id][self.count] = action
        self.buffer_r_n[agent_id][self.count] = reward
        self.buffer_s_next_n[agent_id][self.count] = next_state
        self.buffer_done_n[agent_id][self.count] = done

    def _get_n_step_info(self, n_step_buffer, gamma):
        """Return n-step reward, next state and done flag from the n-step buffer"""
        reward, next_state, done = 0, n_step_buffer[-1][3], n_step_buffer[-1][4]
        for transition in reversed(n_step_buffer):
            r = transition[2]
            reward = r + gamma * reward * (1 - done)
        return reward, next_state, done

    def sample(self):
        """Sample random batch from the replay buffer"""
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = [], [], [], [], []
        for agent_id in range(self.N_drones):
            batch_obs_n.append(torch.tensor(self.buffer_obs_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_a_n.append(torch.tensor(self.buffer_a_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_r_n.append(torch.tensor(self.buffer_r_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_obs_next_n.append(
                torch.tensor(self.buffer_s_next_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_done_n.append(torch.tensor(self.buffer_done_n[agent_id][index], dtype=torch.float).to(self.device))

        return batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n
