
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
import matplotlib.patches as mpatches  # 新增补丁模块


if __name__ == "__main__":
    # 创建环境
    env = MultiAgentEnv()
    num_agents = len(env.agents)  # 动态获取智能体数量（此时为3）

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maddpg = MADDPG(state_dim, action_dim, num_agents, device)

    # 指定要加载的模型路径（如果有的话）
    model_path = "model/model_1437_episode_2600_animation_rewards_416.pth"  # 修改为已有的模型路径
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
log_folder = "log"
model_folder = "model"  # 保存模型的文件夹
plots_folder = "plots"  # 保存奖励图的文件夹
os.makedirs(video_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)
os.makedirs(log_folder, exist_ok=True)


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
animation_episodes = [1,2,3,4,5,6]

def initialize_obstacles(env, ax, obstacle_states_list):
    """动态初始化障碍物图形和轨迹"""
    obstacle_circles = []
    obstacle_trajectories = []
    obstacle_colors = ['black', 'black','brown','brown','brown','black', 'gray', 'purple', 'orange']  # 障碍物颜色列表
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
def save_animation(episode, env, maddpg, filename, actions, obstacle_states_list):

    """支持任意数量障碍物的动画保存函数"""
    fig, ax = plt.subplots()
    ax.set_xlim(0, env.map_size)
    ax.set_ylim(0, env.map_size)
    ax.set_title(f"Episode {episode + 1} Animation", pad=10)

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

    # 遍历所有障碍物
    all_obstacle_pos = [obs.pos for obs in env.obstacles]

    # # 为不同障碍物轨迹分配颜色
    obstacle_colors = ['black', 'black','brown','brown','brown','black', 'gray', 'purple', 'orange']  
        
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
    # 动态生成智能体图例句柄（数量 = num_agents，颜色循环使用 agent_colors）
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
            TargetCrossHandle: HandlerTarget()      # 目标点处理器
        }
    )

    # 动态初始化障碍物（原有逻辑不变）
    obstacle_circles, obstacle_trajectories = initialize_obstacles(env, ax, obstacle_states_list)


    # 初始化 dones 和 states
    dones = [False] * num_agents

    # 固定帧数为环境时间限制
    total_frames = env.time_limit
    
    # 确保动作和障碍物状态列表长度匹配总帧数
    if actions is not None and len(actions) < total_frames:
        # 用最后一帧数据填充到time_limit帧
        last_action = actions[-1]

        last_obstacle = [obs_states[-1] for obs_states in obstacle_states_list] if obstacle_states_list else all_obstacle_pos

        actions = actions + [last_action] * (total_frames - len(actions))


        # 填充每个障碍物的状态到完整长度
        if obstacle_states_list:
            for i in range(len(obstacle_states_list)):
                if len(obstacle_states_list[i]) < total_frames:
                    obstacle_states_list[i] = obstacle_states_list[i] + [last_obstacle[i]] * (total_frames - len(obstacle_states_list[i]))

        
    def update(frame):
        nonlocal states, dones, agent_positions, agent_trajectories, agent_arrows  # 新增 nonlocal 声明
        
        if all(dones):
            # 更新障碍物位置（原有逻辑）
            update_obstacles(obstacle_circles, obstacle_states_list, obstacle_trajectories, frame)
            return agent_circles + agent_arrows + obstacle_circles + target_lines
        
        # 获取当前帧动作（支持任意数量智能体）
        action = actions[frame] if actions is not None else [maddpg.act([state], [done]) for state, done in zip(states, dones)]


        # 执行环境步（支持任意数量智能体）
        next_states, _, new_dones, _ = env.step(action, dones)

        # 动态更新所有智能体位置和轨迹
        for i in range(num_agents):
            if not new_dones[i]:
                agent_positions[i] = next_states[i][:2]
                agent_trajectories[i].append(agent_positions[i])
                agent_circles[i].center = agent_positions[i]            # # 更新智能体图形位置
                agent_radius = env.agents[i]["radius"]  # 从智能体字典获取半径


                # 更新箭头
                speed = np.linalg.norm(action[i]) if len(action) > i else 0
                if speed > 1e-8:
                    # 移除旧箭头（如果存在）
                    if agent_arrows[i] in ax.patches:
                        agent_arrows[i].remove()
                    
                    # 创建新箭头
                    direction = action[i] / speed
                    x, y = agent_positions[i]
                    dx = direction[0] * agent_radius * 1.5
                    dy = direction[1] * agent_radius * 1.5
                    
                    new_arrow = FancyArrow(
                        x, y, dx, dy,
                        width=agent_radius*0.2,
                        head_width=agent_radius*0.5,
                        head_length=agent_radius*0.8,
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
                # 智能体已终止，隐藏箭头和降低圆透明度
                if agent_arrows[i] in ax.patches:
                    agent_arrows[i].remove()
                agent_circles[i].set_alpha(0.35)

        
        # 更新障碍物（原有逻辑）
        update_obstacles(obstacle_circles, obstacle_states_list, obstacle_trajectories, frame)

       
        # 复制线条列表并逐个移除
        for line in list(ax.lines):
            if line not in target_lines:  # 新增判断：跳过目标点线条
                line.remove()


        
        # 绘制智能体轨迹
        for i in range(num_agents):
            ax.plot(*zip(*agent_trajectories[i]), color=agent_colors[i % len(agent_colors)], alpha=0.7, label=f"Agent {i+1} Trajectory")
        
        # 绘制障碍物轨迹
        for i, trajectory in enumerate(obstacle_trajectories):
            color = obstacle_colors[i % len(obstacle_colors)]
            ax.plot(*zip(*trajectory), color=color, alpha=0.7, label=f"Obstacle {i+1} Trajectory")

        
        states = next_states
        dones = new_dones
        return agent_circles + agent_arrows + obstacle_circles + target_lines

        # # 立即同步终止状态
        # current_dones[:] = new_dones
    
    # 生成动画
    ani = animation.FuncAnimation(fig, update, frames=range(total_frames), interval=30, blit=True)

    ani.save(filename, writer="ffmpeg", dpi=200)   ####   dpi=200是调整像素点的,值变大就更清晰(画面像素密度)
    plt.close(fig)



# 测试循环

num_agents = len(env.agents)
total_steps = 0  # 定义 total_steps 并初始化为 0
for episode in tqdm(range(num_episodes), desc="Training Progress", unit="episode"):
    states = env.reset()

    dones = [False] * num_agents

    # 仅在需要保存动画的轮次记录动作、障碍物状态和奖励细节
    num_obstacles = len(env.obstacles)
    if episode + 1 in animation_episodes:
        episode_actions = []
        episode_obstacle_states = [[] for _ in range(num_obstacles)]  # 创建障碍物状态列表

    else:
        episode_actions = None
        episode_obstacle_states = None

    while not all(dones):  # 只要有一个智能体未终止，环境步进就会继续
        # 动作噪声随着训练进行逐渐减小(这里的1.0就是初始噪声系数)
        actions = maddpg.act(states, dones, noise=0)  # 因为是测试,所以噪声为0

        # 记录所有智能体和障碍物的状态
        if episode_actions is not None:
            episode_actions.append(actions)
            for i, obstacle in enumerate(env.obstacles):
                episode_obstacle_states[i].append(obstacle.pos.copy())
        
        next_states, rewards, dones, reward_details = env.step(actions, dones)  # 解包环境状态和终止状态,奖励用不上,但是得有个接收的东西,所以放这

        
        states = next_states
        total_steps += 1  # 每次环境步进后递增 total_steps
    

    elapsed_time = time.time() - start_time
    remaining_episodes = num_episodes - (episode + 1)

    tqdm.write(f"正在生成 Testing Animation{episode + 1}/{num_episodes}")

    # 保存指定轮次的动画        
    if episode_actions is not None:
        video_filename = f"{video_folder}/test_{video_counter}_{episode + 1}.mp4"
        save_animation(episode, env, maddpg, video_filename, episode_actions, episode_obstacle_states)
        video_counter += 1

env.close()



