import os
import random
import torch
import numpy as np
import argparse
from train_test.utils.matd3_graph import MATD3
import copy
from gym_pybullet_drones.envs.A3O3 import A3o3
from gym_pybullet_drones.envs.B3T3 import B3T3
from gym_pybullet_drones.envs.C3V1_Test import C3V1_Test
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import matplotlib.pyplot as plt
import plotly.graph_objects as go

Env_name = 'c3v1G'  # 模型名称  c3v1G是GAT, c3v1G2是GCN
ENV = '3V1'  # 测试3个中的一个指定环境'3o3', '3T3', '3V1'
Mark = 9244  # todo 测试时指定mark C3V1G最好的是9230,9203,9244，9242
test_times = 300
action = 'vel'
Eval_plot = False  # 是否绘制轨迹图像 该选项同时保存txt和png文件,重复保存会覆盖 该选项会增加运行时间!
Env_gui = True  # 环境gui是否开启 建议关闭 该选项会增加时间
Display = False  # 开启Eval_plot后：绘制图像是否展示 建议关闭，想看去文件夹下面看去
Need_Html = False  # 开启Eval_plot后：是否需要Html图像 建议关闭
Success_Time_Limit = 1000  # 成功时间限制，max: 1000, 不在环境中定义 todo 修改成功条件
Success_FollowDistance = 1  # 成功靠近目标距离: 1。跟踪敌机的距离 胜利条件
Success_AttackDistance = 0.5  # 成功打击距离。打击敌机的距离 胜利条件
Success_KeepDistance = 0.5  # 彼此不碰撞距离。不碰撞距离 成功条件
all_axis = 20


# 作战效能=跟踪敌机的距离限制+打击敌机的距离限制+彼此不碰撞距离条件


class Runner:
    def __init__(self, args):
        self.args = args
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_name = Env_name
        self.number = 3  #
        self.seed = 114542  # 保证一个seed，名称使用记号--mark
        self.mark = Mark  # todo 指定mark
        Load_Steps = 10000000  # self.args.max_train_steps = 1e6
        self.test_times = test_times  # 修改为100次运行
        self.done_count = 0  # 用于记录胜利次数
        self.success_count = 0  # 用于记录成功次数（完美条件）
        # Create env
        if ENV == '3o3':
            self.env_evaluate = A3o3(gui=Env_gui, num_drones=args.N_drones, obs=ObservationType('kin_target'),
                                     act=ActionType(action),
                                     ctrl_freq=30,  # 这个值越大，仿真看起来越慢，应该是由于频率变高，速度调整的更小了
                                     need_target=True, obs_with_act=True, all_axis=int(all_axis/10))
        elif ENV == '3T3':
            self.env_evaluate = B3T3(gui=Env_gui, num_drones=args.N_drones, obs=ObservationType('kin_target'),
                                     act=ActionType(action),
                                     ctrl_freq=30,  # 这个值越大，仿真看起来越慢，应该是由于频率变高，速度调整的更小了
                                     need_target=True, obs_with_act=True, all_axis=int(all_axis/10))
        elif ENV == '3V1':
            self.env_evaluate = C3V1_Test(gui=Env_gui, num_drones=args.N_drones, obs=ObservationType('kin_target'),
                                          act=ActionType(action),
                                          ctrl_freq=30,  # 这个值越大，仿真看起来越慢，应该是由于频率变高，速度调整的更小了
                                          need_target=True, obs_with_act=True, all_axis=all_axis)
        else:
            print("Wrong ENV init!!!")
        self.timestep = 1.0 / 30  # 计算每个步骤的时间间隔 0.003

        self.args.obs_dim_n = [self.env_evaluate.observation_space[i].shape[0] for i in
                               range(self.args.N_drones)]  # obs dimensions of N agents
        self.args.action_dim_n = [self.env_evaluate.action_space[i].shape[0] for i in
                                  range(self.args.N_drones)]  # actions dimensions of N agents
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Set random seed
        self.set_random_seed(self.seed)

        # Create N agents
        if self.args.algorithm == "MATD3":
            print("Algorithm: MATD3")
            self.agent_n = [MATD3(self.args, agent_id) for agent_id in range(args.N_drones)]
        else:
            print("Wrong algorithm!!!")
        # 加载模型参数
        for agent_id in range(self.args.N_drones):
            model_path = "../model/{}/{}_actor_mark_{}_number_{}_step_{}k_agent_{}.pth".format(self.env_name,
                                                                                               self.args.algorithm,
                                                                                               self.mark,
                                                                                               self.number,
                                                                                               int(Load_Steps / 1000),
                                                                                               agent_id)  # agent_id
            self.agent_n[agent_id].actor.load_state_dict(torch.load(model_path))
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.noise_std = self.args.noise_std_init  # Initialize noise_std

    def set_random_seed(self, seed):
        """
        设置固定的随机种子以确保可复现性
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def run(self, ):
        for eval_time in range(self.test_times):
            job_done, success = self.evaluate_policy(Eval_plot, eval_time)
            if job_done:
                # self.done_count += 1
                if success:
                    self.success_count += 1
        self.env_evaluate.close()

        # 计算成功率
        # done_rate = self.done_count / self.test_times
        success_rate = self.success_count / self.test_times
        print(f"作战效能: {success_rate * 100}%")

    def evaluate_policy(self, eval_plot, eval_time):  # 仅测试一次的
        all_states, all_actions, all_rewards, all_target_pos = [], [], [], []
        Job_done = False  # 用于记录本次运行是否成功
        Success = False
        obs_n, _ = self.env_evaluate.reset()
        self.env_evaluate.collision = False
        episode_return = [0 for _ in range(self.args.N_drones)]
        episode_states = []
        # episode_actions = []
        episode_rewards = []
        episode_target_pos = []

        for _ in range(self.args.episode_limit):
            a_n = [agent.choose_action(obs, noise_std=0) for agent, obs in zip(self.agent_n, obs_n)]  # 不添加噪声
            # time.sleep(0.01)
            obs_next_n, r_n, done_n, collided, _ = self.env_evaluate.step(copy.deepcopy(a_n))
            for i in range(self.args.N_drones):
                episode_return[i] += r_n[i]

            # 保存状态、动作和奖励
            episode_target_pos.append(10 * self.env_evaluate.TARGET_POS)
            episode_states.append(10 * obs_n)
            episode_rewards.append(r_n)
            obs_n = obs_next_n

        return Job_done, Success


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3 in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=1000, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=100000,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=1, help="Evaluate times")
    parser.add_argument("--max_action", type=float, default=1.0, help="Max action")

    parser.add_argument("--algorithm", type=str, default="MATD3", help="MADDPG or MATD3")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.2, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=3e5,
                        help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--lr_a", type=float, default=5e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=5e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train model")
    # --------------------------------------MATD3--------------------------------------------------------------------
    parser.add_argument("--policy_noise", type=float, default=0.2, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Clip noise")
    parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")

    parser.add_argument("--mark", type=int, default=3, help="The frequency of policy updates")
    parser.add_argument("--N_drones", type=int, default=3, help="The number of drones")
    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    # todo change mark !!!!!!
    runner = Runner(args)
    runner.run()
