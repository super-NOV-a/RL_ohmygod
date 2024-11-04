import os
import random
import torch
import numpy as np
import argparse
from train_test.utils.matd3_graph import MATD3
import copy
from train_test.gym_pybullet_drones.envs.C3V1_Test import C3V1_Test
from train_test.gym_pybullet_drones.utils.enums import ObservationType, ActionType
import matplotlib.pyplot as plt
import plotly.graph_objects as go

Env_name = 'c3v1G'  # c3v1G是GAT, c3v1G2是GCN
Mark = 9244  # todo 测试时指定mark C3V1G最好的是9230,9203,9244，9242
test_times = 300
action = 'vel'
Eval_plot = True  # 是否绘制轨迹图像 该选项同时保存txt和png文件,重复保存会覆盖 该选项会增加运行时间!
Env_gui = False  # 环境gui是否开启 建议关闭 该选项会增加时间
Display = True  # 开启Eval_plot后：绘制图像是否展示 建议关闭，想看去文件夹下面看去
Episode_limit = 1000
Success_FollowDistance = 1  # 成功靠近目标距离: 1。跟踪敌机的距离 胜利条件
Success_AttackDistance = 0.5  # 成功打击距离。打击敌机的距离 胜利条件
Success_KeepDistance = 0.5  # 彼此不碰撞距离。不碰撞距离 成功条件
# 作战效能=跟踪敌机的距离限制+打击敌机的距离限制+彼此不碰撞距离条件
all_axis = 100
Large = False
if all_axis >= 15:  # 范围越大，花费时间越长，测试需要的步数越多
    Large = True
    Episode_limit = int(35.4 * all_axis + 1470)  # 范围为30，步数为2000， 范围为200，步数为5000


class Runner:
    def __init__(self, args):
        self.args = args
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_name = Env_name
        self.number = 3  #
        self.seed = 1145  # 保证一个seed，名称使用记号--mark
        self.mark = Mark  # todo 指定mark
        Load_Steps = 10000000  # self.args.max_train_steps = 1e6
        self.test_times = test_times  # 修改为100次运行
        self.done_count = 0  # 用于记录胜利次数
        self.success_count = 0  # 用于记录成功次数（完美条件）
        # Set random seed
        self.set_random_seed(self.seed)
        # Create env
        self.env_evaluate = C3V1_Test(gui=Env_gui, num_drones=args.N_drones, obs=ObservationType('kin_target'),
                                      act=ActionType(action),
                                      ctrl_freq=30,  # 这个值越大，仿真看起来越慢，应该是由于频率变高，速度调整的更小了
                                      need_target=True,
                                      obs_with_act=True,
                                      follow_distance=Success_FollowDistance,
                                      acctack_distance=Success_AttackDistance,
                                      keep_distance=Success_KeepDistance,
                                      all_axis=int(all_axis))
        self.timestep = 1.0 / 30  # 计算每个步骤的时间间隔 0.003

        self.args.obs_dim_n = [self.env_evaluate.observation_space[i].shape[0] for i in
                               range(self.args.N_drones)]  # obs dimensions of N agents
        self.args.action_dim_n = [self.env_evaluate.action_space[i].shape[0] for i in
                                  range(self.args.N_drones)]  # actions dimensions of N agents
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_dim_n={}".format(self.args.action_dim_n))
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

    def map_value(self, x):
        if x < -1:  # x 在 [-all_axis, -1] 映射到 [-3, -1]
            return -1 + (x + 1) * 2 / (all_axis - 1)
        elif x > 1:  # x 在 [1, all_axis] 映射到 [1, 3]
            return 1 + (x - 1) * 2 / (all_axis - 1)
        else:  # x 在 [-1, 1] 之间，保持不变
            return x

    def map_observation(self, obs):
        # 映射 obs_n[i][0:3] 位置观测
        obs[0:3] = [self.map_value(x) for x in obs[0:3]]
        # 映射 obs_n[i][12:16] 目标位置观测
        obs[12:16] = [self.map_value(x) for x in obs[12:16]]
        # 映射 obs_n[i][16:24] 队友位置观测
        obs[16:24] = [self.map_value(x) for x in obs[16:24]]
        return obs

    def run(self, ):
        for eval_time in range(self.test_times):
            job_done, success = self.evaluate_policy(Eval_plot, eval_time)
            if job_done:
                self.done_count += 1
                if success:
                    self.success_count += 1
        self.env_evaluate.close()

        # 计算成功率
        done_rate = self.done_count / self.test_times
        success_rate = self.success_count / self.test_times
        print(f"任务胜率: {done_rate * 100}%, 完美成功率: {success_rate * 100}%")

    def evaluate_policy(self, eval_plot, eval_time):  # 仅测试一次的
        all_states, all_actions, all_rewards, all_target_pos = [], [], [], []
        Job_done = False  # 用于记录本次运行是否成功
        Success = False
        obs_n, _ = self.env_evaluate.reset()
        self.env_evaluate.collision = False
        episode_return = [0 for _ in range(self.args.N_drones)]
        episode_states = []
        episode_rewards = []
        episode_target_pos = []

        for _ in range(Episode_limit):
            if Large:
                a_n = [agent.choose_action(self.map_observation(obs.copy()), noise_std=0.005) for agent, obs in
                       zip(self.agent_n, obs_n)]  # 不添加噪声
            else:
                a_n = [agent.choose_action(obs, noise_std=0.005) for agent, obs in
                       zip(self.agent_n, obs_n)]  # 不添加噪声

            # time.sleep(0.01)
            obs_next_n, r_n, done_n, collided, _ = self.env_evaluate.step(copy.deepcopy(a_n))
            for i in range(self.args.N_drones):
                episode_return[i] += r_n[i]

            # 保存状态、动作和奖励
            episode_target_pos.append(10*self.env_evaluate.TARGET_POS)
            episode_states.append(10*obs_n)
            episode_rewards.append(r_n)

            obs_n = obs_next_n
            if any(done_n):  # 如果有一个 done 为 True，则算作成功
                Job_done = True  # 打击成功
                if collided is False:  # 期间没有发生碰撞
                    Success = True
                break

        all_target_pos.append(episode_target_pos)
        all_states.append(episode_states)
        all_rewards.append(episode_rewards)
        print("打击胜利:{}, 成功：{} \t episode_reward:{} \t".format(Job_done, Success, episode_return))

        # 将数据转换为numpy数组
        if eval_plot:
            all_target_pos = np.array(all_target_pos)
            all_states = np.array(all_states)
            all_rewards = np.array(all_rewards)
            # 绘制图
            for a_time in range(self.args.evaluate_times):  # 其实每次 eval 只绘制一次
                self.plot_and_save_results(eval_time, all_states[a_time], all_rewards[a_time],
                                           all_target_pos[a_time], Job_done, Success)
        if not eval_plot and Display:
            all_target_pos = np.array(all_target_pos)
            all_states = np.array(all_states)
            all_rewards = np.array(all_rewards)
            for a_time in range(self.args.evaluate_times):
                self.plot_and_save_results(eval_time, all_states[a_time], all_rewards[a_time],
                                           all_target_pos[a_time], Job_done, Success, False)
        return Job_done, Success

    def plot_and_save_results(self, time_i, states, rewards, target_pos, Job_done, Success, save=True):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        # 计算奖励范围以用于颜色映射
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        # 生成三种颜色映射
        cmaps = [plt.cm.Reds, plt.cm.Greens, plt.cm.Blues]  # 三种渐变色
        norms = [plt.Normalize(min_reward - 2, max_reward) for _ in range(3)]
        # 保存路径数据的目录
        save_dir = f"./agent_paths/{self.env_name}_{self.mark}"
        os.makedirs(save_dir, exist_ok=True)
        save_file_path = os.path.join(save_dir, f"{time_i}_胜{Job_done}_成{Success}.txt")
        # 定义不同的颜色，用于区分目标点和无人机
        colors = ['red', 'green', 'blue', 'yellow']  # 三个无人机和一个目标位置

        # 绘制目标点轨迹（目标位置曲线）
        ax.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2],
                color=colors[3], label='Target Position', linestyle='--')
        # 突出显示每个无人机轨迹的起始点，不添加标签
        ax.scatter(target_pos[0, 0], target_pos[0, 1], target_pos[0, 2],
                   color=colors[3], s=100, marker='o')  # 起始点使用大号标记符号

        for agent_id in range(self.args.N_drones):
            agent_states = states[:, agent_id, :3]  # 提取无人机位置信息
            agent_rewards = rewards[:, agent_id]
            cmap = cmaps[agent_id % len(cmaps)]  # 循环使用渐变色
            norm = norms[agent_id % len(norms)]
            # 减少绘制点数
            step_size = max(1, len(agent_states) // 500)  # 绘制最多 500 个点
            # 绘制无人机轨迹，根据奖励值调整颜色亮度
            for i in range(0, len(agent_states) - 1, step_size):
                color_intensity = norm(agent_rewards[i])
                line_color = cmap(color_intensity)
                ax.plot(agent_states[i:i + 2, 0], agent_states[i:i + 2, 1], agent_states[i:i + 2, 2],
                        color=line_color, label=f'Agent {agent_id}' if i == 0 else "")  # 仅为首次绘制添加标签

            # 突出显示每个无人机轨迹的起始点，不添加标签
            ax.scatter(agent_states[0, 0], agent_states[0, 1], agent_states[0, 2],
                       color=colors[agent_id], s=100, marker='o')  # 起始点使用大号标记符号
        if save:
            with open(save_file_path, 'w') as f:
                # 保存目标点数据
                f.write("# Target Trajectories (x, y, z)\n")
                for i in range(len(target_pos)):
                    f.write(f"{target_pos[i, 0]}, {target_pos[i, 1]}, {target_pos[i, 2]}\n")

                # 保存智能体轨迹数据
                f.write("\n# Agent Trajectories (x, y, z)\n")
                for agent_id in range(self.args.N_drones):
                    f.write(f"\nAgent {agent_id} trajectory:\n")
                    for i in range(len(agent_states)):
                        f.write(f"{agent_states[i, 0]}, {agent_states[i, 1]}, {agent_states[i, 2]}\n")
            plt.savefig(os.path.join(save_dir, f"{time_i}_胜{Job_done}_成{Success}.png"))
            # print(f'已经保存:{self.env_name}_{self.mark}_{time_i}相关文件')

        # 设置标题和轴标签
        ax.set_title('Agent and Target Positions Over Time')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # 计算XYZ轴的最小值和最大值，以统一量程
        all_positions = np.concatenate([states[:, :, :3].reshape(-1, 3), target_pos], axis=0)
        x_min, y_min, z_min = np.min(all_positions, axis=0)
        x_max, y_max, z_max = np.max(all_positions, axis=0)
        # 计算每个轴的范围
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
        # 计算中心点
        mid_x = (x_max + x_min) / 2.0
        mid_y = (y_max + y_min) / 2.0
        mid_z = (z_max + z_min) / 2.0
        # 设置统一的XYZ轴范围
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        # 显示图例和布局调整
        ax.legend()
        plt.tight_layout()
        # 保存静态图像

        if not save:
            plt.show(block=False)  # 非阻塞显示图像
            plt.pause(2)  # 使用plt.pause()实现短暂的展示
            plt.close()  # 关闭图像
        else:
            plt.show()  # 如果保存，则显示图像


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MADDPG and MATD3 in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
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
    parser.add_argument("--N_drones", type=int, default=3, help="The number of drones")
    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    # todo change mark !!!!!!
    runner = Runner(args)
    runner.run()
