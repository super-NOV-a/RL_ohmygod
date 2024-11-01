import torch
import numpy as np


class ReplayBuffer(object):
    def __init__(self, args):
        self.device = args.device
        self.N_drones = args.N_drones # The number of agents
        self.n_step = args.n_step  # N-step returns
        self.gamma = args.gamma  # Discount factor
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.count = 0
        self.current_size = 0
        self.buffer_obs_n, self.buffer_a_n, self.buffer_r_n, self.buffer_s_next_n, self.buffer_done_n = [], [], [], [], []
        self.n_step_buffer = [[] for _ in range(self.N_drones)]

        for agent_id in range(self.N_drones):
            self.buffer_obs_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_a_n.append(np.empty((self.buffer_size, args.action_dim_n[agent_id])))
            self.buffer_r_n.append(np.empty((self.buffer_size, 1)))
            self.buffer_s_next_n.append(np.empty((self.buffer_size, args.obs_dim_n[agent_id])))
            self.buffer_done_n.append(np.empty((self.buffer_size, 1)))

    def store_transition(self, obs_n, a_n, r_n, obs_next_n, done_n):
        for agent_id in range(self.N_drones):
            transition = (obs_n[agent_id], a_n[agent_id], r_n[agent_id], obs_next_n[agent_id], done_n[agent_id])

            # 处理未完成状态的 transition
            if not done_n[agent_id]:
                self.n_step_buffer[agent_id].append(transition)

            # 当 done 为 False 且缓冲区达到 n 步时，处理 n 步信息
            if not done_n[agent_id] and len(self.n_step_buffer[agent_id]) >= self.n_step:
                reward, next_state, done = self._get_n_step_info(self.n_step_buffer[agent_id], self.gamma)
                obs, action = self.n_step_buffer[agent_id][0][:2]
                self._store_single_transition(agent_id, obs, action, reward, next_state, done)
                self.n_step_buffer[agent_id].pop(0)

            # 当 done 为 True 时，单独存储这条转移，并清空 n_step_buffer
            if done_n[agent_id]:
                self._store_single_transition(agent_id, obs_n[agent_id], a_n[agent_id], r_n[agent_id],
                                              obs_next_n[agent_id], 1)
                while len(self.n_step_buffer[agent_id]) > 0:  # 存储剩余的 n_step transition
                    reward, next_state, done = self._get_n_step_info(self.n_step_buffer[agent_id], self.gamma)
                    obs, action = self.n_step_buffer[agent_id][0][:2]
                    self._store_single_transition(agent_id, obs, action, reward, next_state, done)
                    self.n_step_buffer[agent_id].pop(0)

    def _store_single_transition(self, agent_id, obs, action, reward, next_state, done):
        self.buffer_obs_n[agent_id][self.count] = obs
        self.buffer_a_n[agent_id][self.count] = action
        self.buffer_r_n[agent_id][self.count] = reward
        self.buffer_s_next_n[agent_id][self.count] = next_state
        self.buffer_done_n[agent_id][self.count] = done

        self.count = (self.count + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def _get_n_step_info(self, n_step_buffer, gamma):
        reward, next_state, done = 0, n_step_buffer[-1][3], n_step_buffer[-1][4]
        for transition in reversed(n_step_buffer):
            r = transition[2]
            reward = r + gamma * reward * (1 - done)
        return reward, next_state, done

    def sample(self,):
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = [], [], [], [], []
        for agent_id in range(self.N_drones):
            batch_obs_n.append(torch.tensor(self.buffer_obs_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_a_n.append(torch.tensor(self.buffer_a_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_r_n.append(torch.tensor(self.buffer_r_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_obs_next_n.append(torch.tensor(self.buffer_s_next_n[agent_id][index], dtype=torch.float).to(self.device))
            batch_done_n.append(torch.tensor(self.buffer_done_n[agent_id][index], dtype=torch.float).to(self.device))

        return batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n

    def get_all_experience(self):
        return (
            [np.array(buffer[:self.current_size]) for buffer in self.buffer_obs_n],
            [np.array(buffer[:self.current_size]) for buffer in self.buffer_a_n],
            [np.array(buffer[:self.current_size]) for buffer in self.buffer_r_n],
            [np.array(buffer[:self.current_size]) for buffer in self.buffer_s_next_n],
            [np.array(buffer[:self.current_size]) for buffer in self.buffer_done_n]
        )

    def transfer_from(self, source_buffer):
        source_obs_n, source_a_n, source_r_n, source_obs_next_n, source_done_n = source_buffer.get_all_experience()
        for i in range(source_buffer.current_size):
            self.store_transition(
                [source_obs_n[j][i] for j in range(len(source_obs_n))],
                [source_a_n[j][i] for j in range(len(source_a_n))],
                [source_r_n[j][i] for j in range(len(source_r_n))],
                [source_obs_next_n[j][i] for j in range(len(source_obs_next_n))],
                [source_done_n[j][i] for j in range(len(source_done_n))]
            )
