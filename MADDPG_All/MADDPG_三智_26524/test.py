
#########   测试代码  ###########################

import time
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multi_agent_env import MultiAgentEnv
from matplotlib.patches import FancyArrow
from maddpg import MADDPG
import os
import re
from matplotlib.legend_handler import HandlerPatch 
import matplotlib.patches as mpatches

################     下面是加载模型进行测试的代码,如果不需要加载,不填入模型路径即可        ##################


if __name__ == "__main__":
    # 创建环境
    env = MultiAgentEnv()
    num_agents = len(env.agents)  # 动态获取智能体数量

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maddpg = MADDPG(state_dim, action_dim, num_agents, device)

    # 指定要加载的模型路径（如果有的话）
    model_path = "model/不错的_4356.pth"  # 修改为已有的模型路径
    if os.path.exists(model_path):  
        maddpg.load(model_path)
        print(f"Loaded pre-trained model from: {model_path}")
    else:
        print("No pre-trained model found, starting training from scratch.")

    num_episodes = 2  # 训练轮数(如果再训练也是训练这么多轮)
    start_time = time.time()

    # 将 epsilon 设置为 0(测试不要探索)如果要探索,则注释掉这行代码即可,因为默认有epsilon
    maddpg.set_epsilon(0)

# 创建保存视频和模型的文件夹
video_folder = "videos"
os.makedirs(video_folder, exist_ok=True)

# 获取当前文件夹中最大的视频编号
def get_next_video_counter(folder):
    existing_files = os.listdir(folder)
    pattern = re.compile(r"test_(\d+)_\d+\.mp4")
    max_counter = 0
    for file in existing_files:
        match = pattern.match(file)
        if match:
            counter = int(match.group(1))
            max_counter = max(max_counter, counter)
    return max_counter + 1

# 获取下一个视频编号
video_counter = get_next_video_counter(video_folder)


# 定义需要保存动画的轮次
animation_episodes = [1,2, 3, 4, 5]

def initialize_obstacles(env, ax, obstacle_states_list):
    """动态初始化障碍物图形和轨迹"""
    obstacle_circles = []
    obstacle_trajectories = []
    obstacle_colors = ['black', 'black','black', 'black', 'gray','gray','gray','gray','purple', 'brown','brown','orange']  
    for i, obstacle in enumerate(env.obstacles):
        pos = obstacle.pos.copy()
        color = obstacle_colors[i % len(obstacle_colors)]
        circle = plt.Circle(pos, obstacle.radius, color=color, alpha=0.7, label=f"Obstacle {i+1}")
        ax.add_patch(circle)
        obstacle_circles.append(circle)
        obstacle_trajectories.append([pos])
    return obstacle_circles, obstacle_trajectories

def update_obstacles(obstacle_circles, obstacle_states_list, obstacle_trajectories, frame):
    """更新障碍物位置和轨迹"""
    for i, circle in enumerate(obstacle_circles):
        if obstacle_states_list and len(obstacle_states_list[i]) > frame:
            pos = obstacle_states_list[i][frame]
            circle.center = pos
            if obstacle_trajectories is not None:  # 确保列表存在
                obstacle_trajectories[i].append(pos)

##  给起点小黑球做父类的
class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                        xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Circle(xy=center, 
                            radius=min(width, height) * 0.4)  # 调整半径比例
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

##  图例的智能体（横线）的定义
class HandlerAgentLine(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                        xdescent, ydescent, width, height, fontsize, trans):
        center_x = 0.5 * width - 0.5 * xdescent
        center_y = 0.5 * height - 0.5 * ydescent
        line_length = min(width, height) * 2.2
        line = plt.Line2D(
            [center_x - line_length/2, center_x + line_length/2],
            [center_y, center_y],
            color=orig_handle.get_color(),
            linewidth=1.2,
            transform=trans
        )
        return [line]

##  自定义智能体横线句柄类（用于区分目标点）
class AgentLineHandle(plt.Line2D):
    pass

##  自定义目标点叉号句柄类（用于区分智能体）
class TargetCrossHandle(plt.Line2D):
    pass

##  图例的起点小黑球的定义
class HandlerStartMarker(HandlerCircle):
    def create_artists(self, legend, orig_handle,
                        xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Circle(xy=center, radius=min(width, height) * 0.2)  # 调整半径为 0.2
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

##  图例的目标点叉的定义
class HandlerTarget(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                        xdescent, ydescent, width, height, fontsize, trans):
        size = min(width, height) * 1.0  # 叉号大小
        center_x = 0.5 * width - 0.5 * xdescent
        center_y = 0.5 * height - 0.5 * ydescent
        linewidth = 0.7  # 线宽调整为1.0
        line1 = plt.Line2D(
            [center_x - size/2, center_x + size/2],
            [center_y - size/2, center_y + size/2],
            color='black', linewidth=linewidth, transform=trans
        )
        line2 = plt.Line2D(
            [center_x + size/2, center_x - size/2],
            [center_y - size/2, center_y + size/2],
            color='black', linewidth=linewidth, transform=trans
        )
        return [line1, line2]

# #####用于保存动画的函数
def save_animation(episode, env, maddpg, filename, actions, obstacle_states_list, recorded_states):
    """支持任意数量障碍物的动画保存函数 - 使用记录的状态，固定时长"""
    fig, ax = plt.subplots()
    ax.set_xlim(0, env.map_size)
    ax.set_ylim(0, env.map_size)
    ax.set_title(f"Test {episode + 1} Animation", pad=10)

    states = env.reset()
    num_agents = len(env.agents)  # 获取动态智能体数量

    # 添加x轴和y轴标签
    map_size = env.map_size
    ax.text(map_size*0.5, -0.1*map_size, 'x(m)', fontsize=12, ha='center', va='center')  # x轴标签
    ax.text(-0.1*map_size, map_size*0.5, 'y(m)', fontsize=12, rotation=90, ha='center', va='center')  # y轴标签

    # 动态初始化智能体位置和图形
    agent_positions = [state[:2] for state in states]  # 所有智能体初始位置
    agent_circles = []
    agent_arrows = []
    agent_trajectories = []

    agent_start_markers = []  # 起点标记（黑色小圆，半径1）

    agent_colors = ['blue', 'red', 'green']  # 支持最多5个智能体，可扩展

    # 初始化目标
    target_x, target_y = env.target_pos
    size = 0.3  # 叉的大小（可调整）
    # 左上到右下对角线
    line1, = ax.plot([target_x - size, target_x + size], [target_y - size, target_y + size], 
            color='black', linewidth=1, label="Target")
    # 右上到左下对角线
    line2, = ax.plot([target_x + size, target_x - size], [target_y - size, target_y + size], 
            color='black', linewidth=1)
    target_lines = [line1, line2]  # 保存目标点线条

    target_pos = env.target_pos
    target_marker, = ax.plot(target_pos[0], target_pos[1], 'kx', markersize=8, label="Target")

    # 如果初始状态没有终止状态字段，添加默认值
    if "agent_dones" not in recorded_states[0]:
        recorded_states[0]["agent_dones"] = [False] * num_agents

    # 遍历所有障碍物
    all_obstacle_pos = [obs.pos for obs in env.obstacles]

    # # 为不同障碍物轨迹分配颜色
    obstacle_colors = ['black', 'black','black', 'gray', 'gray','gray','gray','gray','purple', 'brown','brown','orange']  # 障碍物颜色列表
        
    for i in range(num_agents):
        pos = agent_positions[i]
        color = agent_colors[i % len(agent_colors)]
        agent_radius = env.agents[i]["radius"]  # 从智能体字典获取半径
        circle = plt.Circle(pos, agent_radius, color=color, alpha=0.7)
        ax.add_patch(circle)
        agent_circles.append(circle)
        
        # 起点标记（黑色小圆，半径1）
        start_marker = plt.Circle(pos, 0.1, color='black', alpha=0.9)
        agent_start_markers.append(start_marker)
        ax.add_patch(start_marker)

        arrow = FancyArrow(pos[0], pos[1], 0, 0, 
                            width=agent_radius*0.2, 
                            head_width=agent_radius*0.5, 
                            head_length=agent_radius*0.8, 
                            color=color, alpha=0.7)
        ax.add_patch(arrow)
        agent_arrows.append(arrow)
        
        agent_trajectories.append([pos])

    # ------------------------------ 自定义图例处理器------------------------------
    # 动态生成智能体图例句柄
    agent_handles = [
        AgentLineHandle([0], [0], color=agent_colors[i % len(agent_colors)], lw=2.0)
        for i in range(num_agents)
    ]

    # 固定添加起点和目标点图例句柄
    legend_handles = agent_handles + [
        mpatches.Circle((0, 0), 1, color='black', alpha=0.9),  # 起点标记
        TargetCrossHandle([0], [0], color='black', linestyle='None', marker='x', markersize=8)  # 目标点
    ]

    # 动态生成智能体图例标签
    agent_labels = [f"Agent {i+1}" for i in range(num_agents)]

    # 固定添加起点和目标点图例标签
    legend_labels = agent_labels + ["Start", "Target"]

    # 配置图例，使用不同的处理器映射
    ax.legend(
        legend_handles,
        legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=8,
        frameon=True,
        framealpha=0.3,
        borderaxespad=0.5,
        handler_map={
            AgentLineHandle: HandlerAgentLine(),  # 智能体横线处理器
            mpatches.Circle: HandlerStartMarker(),  # 起点处理器
            TargetCrossHandle: HandlerTarget()  # 目标点处理器
        }
    )
    
    # 动态初始化障碍物（原有逻辑不变）
    obstacle_circles, obstacle_trajectories = initialize_obstacles(env, ax, obstacle_states_list)

    # 固定帧数为环境的最大步数（固定时长）
    total_frames = env.time_limit
    
    # 确保动作和障碍物状态列表长度匹配记录的状态数
    if actions is not None and len(actions) < len(recorded_states) - 1:
        # 用最后一帧数据填充
        last_action = actions[-1]
        actions = actions + [last_action] * (len(recorded_states) - len(actions) - 1)

    # 填充每个障碍物的状态到完整长度
    if obstacle_states_list:
        for i in range(len(obstacle_states_list)):
            if len(obstacle_states_list[i]) < len(recorded_states):
                last_obs = obstacle_states_list[i][-1] if obstacle_states_list[i] else all_obstacle_pos[i]
                obstacle_states_list[i] = obstacle_states_list[i] + [last_obs] * (len(recorded_states) - len(obstacle_states_list[i]))
    
    def update(frame):
        """使用记录的状态更新动画帧，支持固定时长"""
        # 获取当前帧记录的状态（如果存在）
        if frame < len(recorded_states):
            state = recorded_states[frame]
            
            # 更新智能体位置
            for i in range(num_agents):
                agent_pos = state["agent_positions"][i]
                agent_circles[i].center = agent_pos
                agent_trajectories[i].append(agent_pos)
                
                # 检查智能体是否终止
                if state["agent_dones"][i]:
                    # 设置圆圈透明度
                    agent_circles[i].set_alpha(0.35)
                    # 如果已终止，确保移除箭头
                    if agent_arrows[i] in ax.patches:
                        agent_arrows[i].remove()
                    agent_arrows[i] = FancyArrow(0, 0, 0, 0, alpha=0) # 占位符，确保后续逻辑正常
                else:
                    # 未终止的智能体保持正常透明度
                    agent_circles[i].set_alpha(0.7)

                    # 更新箭头（仅当有动作时）
                    if frame > 0 and actions and frame - 1 < len(actions):
                        action = actions[frame - 1][i]
                        speed = np.linalg.norm(action)
                        if speed > 1e-8:
                            # 移除旧箭头（如果存在）
                            if agent_arrows[i] in ax.patches:
                                agent_arrows[i].remove()
                            
                            # 创建新箭头
                            direction = action / speed
                            x, y = agent_pos
                            dx = direction[0] * env.agents[i]["radius"] * 1.5
                            dy = direction[1] * env.agents[i]["radius"] * 1.5
                            
                            new_arrow = FancyArrow(
                                x, y, dx, dy,
                                width=env.agents[i]["radius"]*0.2,
                                head_width=env.agents[i]["radius"]*0.5,
                                head_length=env.agents[i]["radius"]*0.8,
                                color=agent_colors[i % len(agent_colors)],
                                alpha=0.7
                            )
                            ax.add_patch(new_arrow)
                            agent_arrows[i] = new_arrow
                        else:
                            # 速度为0，隐藏箭头
                            if agent_arrows[i] in ax.patches:
                                agent_arrows[i].remove()
                            agent_arrows[i] = FancyArrow(0, 0, 0, 0, alpha=0)  # 占位符
                    else:
                        # 第一帧或没有动作，隐藏箭头
                        if agent_arrows[i] in ax.patches:
                            agent_arrows[i].remove()
                        agent_arrows[i] = FancyArrow(0, 0, 0, 0, alpha=0)
            
            # 更新障碍物位置
            for i, obs_circle in enumerate(obstacle_circles):
                if i < len(state["obstacle_positions"]):
                    obs_pos = state["obstacle_positions"][i]
                    obs_circle.center = obs_pos
                    obstacle_trajectories[i].append(obs_pos)
        
        # 复制线条列表并逐个移除
        for line in list(ax.lines):
            if line not in target_lines:  # 新增判断：跳过目标点线条
                line.remove()
        
        # 绘制智能体轨迹
        for i in range(num_agents):
            ax.plot(*zip(*agent_trajectories[i]), color=agent_colors[i % len(agent_colors)], alpha=0.7)
        
        # 绘制障碍物轨迹
        for i, trajectory in enumerate(obstacle_trajectories):
            color = obstacle_colors[i % len(obstacle_colors)]
            ax.plot(*zip(*trajectory), color=color, alpha=0.7)
        
        return agent_circles + [arrow for arrow in agent_arrows if arrow in ax.patches] + obstacle_circles + target_lines

    # 生成动画（固定帧数为最大步数）
    ani = animation.FuncAnimation(fig, update, frames=range(total_frames), interval=30, blit=True)

    ani.save(filename, writer="ffmpeg", dpi=200)   ####   dpi=200是调整像素点的,值变大就更清晰(画面像素密度)
    ax.legend(loc="upper right")
    plt.close(fig)


# 测试循环
num_agents = len(env.agents)
total_steps = 0  # 定义 total_steps 并初始化为 0
for episode in tqdm(range(num_episodes), desc="Training Progress", unit="episode"):
    states = env.reset()
    dones = [False] * num_agents
    # 初始化动画轮次的奖励记录列表
    recorded_states = []  # 新增：保存每一步的环境状态

    # 仅在需要保存动画的轮次记录动作、障碍物状态和奖励细节
    num_obstacles = len(env.obstacles)
    if episode + 1 in animation_episodes:
        episode_actions = []
        episode_obstacle_states = [[] for _ in range(num_obstacles)]  # 创建障碍物状态列表
        recorded_states = []  # 初始化状态记录列表
        
        # 记录初始状态
        initial_state = {
            "agent_positions": [agent["pos"].copy() for agent in env.agents],
            "obstacle_positions": [obs.pos.copy() for obs in env.obstacles],
            "agent_speeds": [agent["speed"].copy() for agent in env.agents],
            "agent_arrived": [agent["arrived"] for agent in env.agents],
            "agent_arrived_by_target": [agent["arrived_by_target"] for agent in env.agents],
            "agent_prev_distances": [agent["prev_distance_to_target"] for agent in env.agents],
            "current_time": 0,
            "target_reached": False,
            "agent_dones": [False] * num_agents  # 新增：终止状态
        }
        recorded_states.append(initial_state)
    else:
        episode_actions = None
        episode_obstacle_states = None
        recorded_states = None

    while not all(dones):  # 只要有一个智能体未终止，环境步进就会继续
        # 动作噪声随着训练进行逐渐减小(这里的1.0就是初始噪声系数)
        actions = maddpg.act(states, dones, noise=0)  # 增加动作噪声以促进探索

        # 记录所有智能体和障碍物的状态
        if episode_actions is not None:
            episode_actions.append(actions)
            for i, obstacle in enumerate(env.obstacles):
                episode_obstacle_states[i].append(obstacle.pos.copy())
        
        # 执行一步环境
        next_states, rewards, dones, reward_details, full_state = env.step(actions, dones)
        
        # 记录当前状态（添加终止状态）
        if recorded_states is not None:
            # 创建包含终止状态的完整状态记录
            state_to_record = full_state.copy()
            state_to_record["agent_dones"] = dones.copy()  # 添加终止状态
            recorded_states.append(state_to_record)
        
        states = next_states
        total_steps += 1  # 每次环境步进后递增 total_steps
    
    elapsed_time = time.time() - start_time
    remaining_episodes = num_episodes - (episode + 1)
    estimated_remaining_time = (elapsed_time / (episode + 1)) * remaining_episodes

    tqdm.write(f"正在生成 Testing Animation{episode + 1}/{num_episodes}")

    # 保存指定轮次的动画        
    if episode_actions is not None:
        video_filename = f"{video_folder}/test_{video_counter}_{episode + 1}.mp4"
        save_animation(episode, env, maddpg, video_filename, episode_actions, episode_obstacle_states, recorded_states)
        video_counter += 1

env.close()



