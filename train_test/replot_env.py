import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_and_plot_key_points(save_file_path):
    # 打开文件读取内容
    with open(save_file_path, 'r') as f:
        lines = f.readlines()

    # 初始化变量
    target_pos_start = None
    target_pos_end = None
    agent_start_points = []
    reading_target = False
    agent_id = -1

    for line in lines:
        if not line.strip():
            continue

        # 检查是否是目标轨迹
        if line.startswith("# Target Trajectories"):
            reading_target = True
            continue
        elif line.startswith("# Agent Trajectories"):
            reading_target = False
            continue
        elif line.startswith("Agent"):
            agent_id += 1
            continue

        # 解析坐标
        coords = list(map(float, line.strip().split(", ")))

        # 获取目标起始点和终点
        if reading_target:
            if target_pos_start is None:
                target_pos_start = coords
            target_pos_end = coords
        # 获取每个Agent的起始点
        elif agent_id < 3 and len(agent_start_points) < 3:
            if len(agent_start_points) == agent_id:
                agent_start_points.append(coords)

    # 创建图
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制目标起始点和终点
    ax.scatter(*target_pos_start, color='yellow', s=100, marker='o', label='Target Start')
    ax.scatter(*target_pos_end, color='orange', s=100, marker='^', label='Target End')

    # 定义颜色并绘制每个无人机的起始点
    colors = ['r', 'g', 'b']
    for i, start_point in enumerate(agent_start_points):
        ax.scatter(*start_point, color=colors[i], s=100, marker='o', label=f'Agent {i} Start')

    # 设置图例和标题
    ax.legend()
    ax.set_title("Key Points: Target Start/End and Agent Start Points")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 显示图像
    plt.show()


# 调用函数并传入保存文件路径
save_file_path = "2_共享/30_胜True_成True.txt"  # 替换为实际的文件路径
load_and_plot_key_points(save_file_path)
