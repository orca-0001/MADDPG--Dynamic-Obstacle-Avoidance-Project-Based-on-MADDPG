

################     精简版       ################################
###########    先试试能不能复现好的效果,奖励函数改成一样的了 ################
##########  减缓了障碍物的速度,增大了障碍物的范围和惩罚值   ###############

######  有避障效果,但是智能体在目标点附近会徘徊一阵子,尝试增大后退惩罚减少线性奖励  ###############

#######  增加智能体速度,调高障碍物的速度,障碍物惩罚和范围不变看看   ##########
#######   降低智能体速度,增加障碍物惩罚试试     ##############
####   再试一次,上次的效果不错,还有滑行的问题找到一个治标的方法,暂时不管了,下次来尝试随意增加智能体了  ##############


#######    改了智能体的前一步的距离更新,然后相对距离起作用了,删除了奖励文档的直接生成而不是在log里面   ######

#####  改了一下障碍物的运动轨迹,防止碰撞,移动右上角智能体初始位置,

#######3     0.9改0.8
######    400000个经验(1000轮左右)才开始训练,经验池总量3000000,修改了强制移动立即停止的bug,原来没有强制移动自然也加不进目标点奖励和智能体间的奖励,还好修复了

#####   30000轮,500000经验开始学习(0.2,,,0.1),效果不错,就是有点曲折,在目标点不够直接

###  增加了图例和提高了动画清晰度,再试一次,0.6,0.2   
#        
#### 最好的效果,改进最全(复件) 缩小十倍
##   改变智能体和障碍物的大小,e这个也是,只缩小线性奖励,预训练,重新训练试试
###  已经训练出一个不错的模型,椭圆障碍物是(11,11)的速度是0,02,直线运动那个要往右移两个单位(虽然不影响)

import gym
from gym import spaces
import numpy as np
from scipy.optimize import fsolve

class Obstacle:
    def __init__(self, obstacle_type, start_pos, **kwargs):
        """
        障碍物类
        :param obstacle_type: 'ellipse'（椭圆运动）或 'back_and_forth'（两点来回运动）
        :param start_pos: 初始位置坐标
        :param kwargs: 其他参数（椭圆中心、半轴长度、目标点等）
        """
        self.type = obstacle_type
        self.radius = np.float32(0.5)
        self.pos = np.array(start_pos, dtype=np.float32)

        # 椭圆运动参数（关键修改）
        if self.type == 'ellipse':
            self.center = np.array(kwargs.get('center', [13.50, 13.50]), dtype=np.float32)
            self.a = np.float32(kwargs.get('a', 2.0))
            self.b = np.float32(kwargs.get('b', 0.5))

            self.radius = np.float32(kwargs.get('radius', 0.5))  # 支持自定义半径

            self.angular_speed = np.float32(kwargs.get('angular_speed', 0.03))
            self.theta = np.deg2rad(kwargs.get('theta', 135))  # 允许自定义旋转角度
            
            # 计算初始相位 phi，使得初始位置为 start_pos（右长半轴端点，phi=派）
            self.phi = np.pi  # 初始相位为0，对应右长半轴端点
            self.pos = self._calculate_ellipse_position(self.phi)  # 用参数方程计算初始位置
            self.pos = np.clip(self.pos, self.radius, 20 - self.radius)  # 边界裁剪

        # 两点来回运动参数
        elif self.type == 'back_and_forth':        
            # 明确起点和终点
            self.start = np.array(start_pos, dtype=np.float32)
            self.end = np.array(kwargs.get('target', start_pos), dtype=np.float32)
            self.target = self.end  # 初始目标为终点
            self.speed = np.float32(kwargs.get('speed', 0.05))
            self.direction = self._get_direction()
            self.radius = np.float32(kwargs.get('radius', 0.5))  # 支持自定义半径

        # 边界裁剪初始化
        self.pos = np.clip(self.pos, self.radius, 20 - self.radius)

    def _calculate_ellipse_position(self, phi):
        """根据相位 phi 计算椭圆上的位置"""
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        
        x = self.center[0] + self.a * cos_phi * cos_theta - self.b * sin_phi * sin_theta
        y = self.center[1] + self.a * cos_phi * sin_theta + self.b * sin_phi * cos_theta
        return np.array([x, y], dtype=np.float32)


    def _get_direction(self):
        dir_vec = self.target - self.pos
        norm = np.linalg.norm(dir_vec)    
        return dir_vec / norm if norm > 1e-8 else self.direction  # 安全除法


    def update_position(self, t):
        """更新障碍物位置"""
        if self.type == 'ellipse':
            self._update_ellipse_position(t)
        elif self.type == 'back_and_forth':
            self._update_back_and_forth_position()
    
    def _update_ellipse_position(self, t):
        """椭圆运动更新逻辑"""
        phi = self.angular_speed * t + np.pi  # 初始相位为 π（180°），对应右侧顶点且参数角随时间变化
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        
        x = self.center[0] + self.a * cos_phi * cos_theta - self.b * sin_phi * sin_theta
        y = self.center[1] + self.a * cos_phi * sin_theta + self.b * sin_phi * cos_theta
        self.pos = np.array([x, y])
        self.pos = np.clip(self.pos, self.radius, 20 - self.radius)  # 边界裁剪

    def _update_back_and_forth_position(self):
        distance = np.linalg.norm(self.pos - self.target)
        
        if distance > self.radius + 1e-6:
            self.pos += self.direction * self.speed
            self.pos = np.clip(self.pos, self.radius, 20 - self.radius)
        else:
            # 切换目标点为起点/终点
            self.target = self.start if np.array_equal(self.target, self.end) else self.end
            self.direction = self._get_direction()  # 重新计算方向
        # 边界裁剪（不影响方向）
        self.pos = np.clip(self.pos, self.radius, 20 - self.radius)

class MultiAgentEnv(gym.Env):
    def __init__(self):
        super(MultiAgentEnv, self).__init__()


        # 智能体列表（支持动态扩展，每个元素为字典）
        self.agents = [
            {
                "id": 0,
                "pos": np.array([4.0, 4.0], dtype=np.float32),
                "speed": np.zeros(2, dtype=np.float32),
                "arrived": False,
                "total_reward": 0.0,
                "prev_distance_to_target": None,
                "arrived_by_target": False,
                "radius": 0.5, 
            }
        ]
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )


        
        # 统一管理所有障碍物（示例：2个椭圆 + 2个来回运动）
        self.obstacles = [
            # 椭圆障碍物1（中心在(135,135)）
            # Obstacle('ellipse', start_pos=[12, 12], center=[17, 17], a=2, b=0.5, angular_speed=0.03, radius=0.5),
            # Obstacle('ellipse', start_pos=[12.5, 15], center=[17, 17], a=2, b=0.5, angular_speed=0.0, radius=0.5),
            Obstacle('back_and_forth', start_pos=[12.5, 15], target=[11, 5], speed=0.0, radius=0.5),  # 新增行 

            # 椭圆障碍物2（中心在(80,80)）
            # Obstacle('ellipse', start_pos=[10, 10], center=[11, 11], a=2, b=0.8, angular_speed=0.0, theta=270, radius=0.5),
            Obstacle('back_and_forth', start_pos=[10, 10], target=[11, 5], speed=0.0, radius=0.5),  # 新增行 

            
            # 椭圆障碍物2（中心在(40,80)）
            Obstacle('ellipse', start_pos=[6, 7], center=[8, 14], a=2.5, b=1, angular_speed=0.03, theta=180, radius=0.5),


            # 来回运动障碍物1：在(90,50)和(50,90)之间
            Obstacle('back_and_forth', start_pos=[4, 9], target=[8, 5], speed=0.05, radius=0.5),
            # 来回运动障碍物2：在(150,50)和(190,90)之间
            Obstacle('back_and_forth', start_pos=[14, 6], target=[10, 6], speed=0.05, radius=0.5),
            # 新增：来回运动障碍物3（示例）
            Obstacle('back_and_forth', start_pos=[8, 2], target=[11, 5], speed=0.0, radius=0.5),  # 新增行 


        ]

        # 观测空间维度：智能体(4) + 其他智能体(4) + 障碍物(4*n) + 目标(2) 
        num_agents = len(self.agents)
        num_obstacles = len(self.obstacles)
        self.observation_space = spaces.Box(
            low=-1000,
            high=1000,
            shape=(4*num_agents + 4*num_obstacles + 2,),  # 4*智能体(4) + 障碍物(4n) + 目标(2)
            dtype=np.float32
        )

        # 保存上一时刻的障碍物位置（用于计算速度）
        self.prev_obstacle_positions = [obs.pos.copy() for obs in self.obstacles]

        

        self.agent1_speed = np.array([0.0, 0.0])  # 智能体 1 的速度
        self.agent2_speed = np.array([0.0, 0.0])  # 智能体 2 的速度
        self.target_pos = np.array([16.0, 16.0])  # 目标位置
        self.map_size = 20
        self.time_limit = 500
        self.current_time = 0

        # 智能体速度上限
        self.agent_speed_limit = 0.07

        # 奖励函数中的惩罚参数
        self.step_penalty = 0.1  # 每一步的小惩罚



###########每一轮训练才重置一次
    def reset(self):

        self.current_time = 0

        # 新增：重置终止原因标志
        self.arrived_by_target = [False] * len(self.agents)  # 动态长度

        self.agents = [
            {
                "id": 0,
                "pos": np.array([4.0, 4.0], dtype=np.float32),
                "speed": np.zeros(2, dtype=np.float32),
                "arrived": False,
                "total_reward": 0.0,
                "prev_distance_to_target": None,
                "arrived_by_target": False,
                "radius": 0.5, 
            }
        ]
        
        self.obstacles = [
            # 椭圆障碍物1（中心在(135,135)）
            # Obstacle('ellipse', start_pos=[12, 12], center=[17, 17], a=2, b=0.5, angular_speed=0.03, radius=0.5),
            # Obstacle('ellipse', start_pos=[12.5, 15], center=[17, 17], a=2, b=0.5, angular_speed=0.0, radius=0.5),
            Obstacle('back_and_forth', start_pos=[12.5, 15], target=[11, 5], speed=0.0, radius=0.5),  # 新增行 

            # 椭圆障碍物2（中心在(80,80)）
            # Obstacle('ellipse', start_pos=[10, 10], center=[11, 11], a=2, b=0.8, angular_speed=0.0, theta=270, radius=0.5),
            Obstacle('back_and_forth', start_pos=[10, 10], target=[11, 5], speed=0.0, radius=0.5),  # 新增行 

            
            # 椭圆障碍物2（中心在(40,80)）
            Obstacle('ellipse', start_pos=[6, 7], center=[8, 14], a=2.5, b=1, angular_speed=0.03, theta=180, radius=0.5),


            # 来回运动障碍物1：在(90,50)和(50,90)之间
            Obstacle('back_and_forth', start_pos=[4, 9], target=[8, 5], speed=0.05, radius=0.5),
            # 来回运动障碍物2：在(150,50)和(190,90)之间
            Obstacle('back_and_forth', start_pos=[14, 6], target=[10, 6], speed=0.05, radius=0.5),
            # 新增：来回运动障碍物3（示例）
            Obstacle('back_and_forth', start_pos=[8, 2], target=[11, 5], speed=0.0, radius=0.5),  # 新增行 

        ]

        # 初始化上一位置（用于速度计算）
        self.prev_obstacle_positions = [obs.pos.copy() for obs in self.obstacles]

        return self._get_obs()


    def calculate_obstacle_penalty_for_agent(self, pos,agent_radius):
        """计算单个智能体的障碍物惩罚"""
        obstacle_penalty = 0.0
        min_obs_dist = np.inf
        for obs in self.obstacles:
            dist = np.linalg.norm(pos - obs.pos)
            if dist < min_obs_dist:
                min_obs_dist = dist
            if dist < 1.8:
                obstacle_penalty -= self.calculate_obstacle_penalty(dist)
        return obstacle_penalty
    
    def calculate_boundary_penalty(self, pos,agent_radius):
        """计算边界惩罚"""
        distance_to_boundary = min(pos[0], self.map_size - pos[0], pos[1], self.map_size - pos[1])
        if distance_to_boundary < agent_radius * 6:
            return -(agent_radius * 6 - distance_to_boundary) * 15.0
        return 0.0

    def calculate_agent_penalty(self, agent_id, pos, new_dones):
        """计算智能体间惩罚"""
        penalty = 0.0
        for j in range(len(self.agents)):
            if agent_id != j and not new_dones[agent_id] and not self.agents[j]["arrived_by_target"]:
                distance = np.linalg.norm(pos - self.agents[j]["pos"])
                if distance < 1.6:
                    penalty -= (1.6 - distance) * 200.0
        return penalty

    # 惩罚智能体接近障碍物(内圈和外圈)============参数化==========     ###   既然这里给了固定数值2.2,那么必须保证进来这个函数之前两者距离得小于2.2,否则就是异常惩罚1200
    def calculate_obstacle_penalty(self, distance):             ###  同样的这里最小距离是0.3,那么必须保证最小距离大于0.3
        # 定义外圈和内圈的距离范围和惩罚值
        outer_min_distance = 1.31
        outer_max_distance = 1.8
        outer_max_penalty = 700  # 外圈最大惩罚

        inner_min_distance = 0.3
        inner_max_distance = 1.3
        inner_min_penalty = 800  # 内圈最小惩罚
        inner_max_penalty = 1300  # 内圈最大惩罚

        if outer_min_distance <= distance <= outer_max_distance:
            # 外圈惩罚：线性递增
            return outer_max_penalty / (outer_min_distance - outer_max_distance) * (distance - outer_max_distance)
        elif inner_min_distance <= distance < inner_max_distance:
            # 内圈惩罚：线性递增
            return (inner_max_penalty - inner_min_penalty) / (inner_min_distance - inner_max_distance) * (distance - inner_max_distance) + inner_min_penalty
        else:  # distance < inner_min_distance
            # 距离小于内圈最小距离时的惩罚
            return inner_max_penalty


    def calculate_linear_reward(self, distance):
        min_distance = 0.1
        max_distance = 12
        max_reward = 2
        min_reward = 1
        return max_reward - (distance - min_distance) * (max_reward - min_reward) / (max_distance - min_distance)

    def calculate_relative_reward(self, distance, prev_distance):
        if distance < prev_distance:
            return (prev_distance - distance) * 200
        else:
            return -2 * self.calculate_linear_reward(distance) - (distance - prev_distance) * 500 * 3

        

    def step(self, actions, dones):
        num_agents = len(self.agents)
        rewards = [0.0 for _ in range(num_agents)]
        new_dones = [False for _ in range(num_agents)]
        target_positions = [None for _ in range(num_agents)]
        reward_details = []  # 新增：存储每个智能体的奖励细节

        # 0. 更新障碍物位置
        current_time = self.current_time
        for obs in self.obstacles:
            obs.update_position(current_time)
        
        # 保存当前障碍物位置用于下一时刻计算速度
        current_obstacle_positions = [obs.pos.copy() for obs in self.obstacles]
        
        # 1. 计算临时位置(智能体)
        for i in range(num_agents):
            if dones[i]:
                continue

            agent_radius = self.agents[i]["radius"]   # 获取当前智能体半径

            # 如果已经到达目标点，直接标记为终止
            if np.linalg.norm(self.agents[i]["pos"] - self.target_pos) < agent_radius * 0.2:
                new_dones[i] = True
                self.agents[i]["arrived_by_target"] = True
                continue

            linear_vel = actions[i][:2]
            direction = linear_vel / (np.linalg.norm(linear_vel) + 1e-8)
            temp_pos = self.agents[i]["pos"] + direction * np.linalg.norm(linear_vel) * self.agent_speed_limit
            temp_pos = np.clip(temp_pos, agent_radius, self.map_size - agent_radius)
            

            #######*********************************************************************#######
            # 检测目标点强制移动 (若出现智能体在目标点附近突然直线移动不受控制的滑行的状况,其实代表智能体已经到达目标点完成任务了
            # 只需要在测试该模型时把强制移动的半径扩大一些即可,也就是这个 self.agent_radius * 0.6,这是因为动画保存时有一个逻辑是让智能体保持最后一帧的动作,那里有点小问题,不想改了
            #######*********************************************************************#######
            
            self.agents[i]["pos"] = temp_pos

            # 检查是否到达目标点
            dist_to_target = np.linalg.norm(temp_pos - self.target_pos)
            if dist_to_target < agent_radius * 0.6:
                target_positions[i] = self.target_pos
                self.agents[i]["pos"] = self.target_pos
                
            # 立即检查是否触发终止条件
            if dist_to_target < agent_radius * 0.2:
                new_dones[i] = True
                self.agents[i]["arrived_by_target"] = True


        # 2. 计算终止状态（边界/碰撞/时间）
        for i in range(num_agents):
            if new_dones[i]:  # 如果已经终止，跳过检查
                continue        ######    再说一遍,满足if会只执行上面的,不满足会只执行下面的
            pos = self.agents[i]["pos"]
            agent_radius = self.agents[i]["radius"]   # 获取当前智能体半径

            # 边界终止
            if not new_dones[i]:
                boundary_check = any([
                    pos[0] < agent_radius * 1.2,
                    pos[0] > self.map_size - agent_radius * 1.2,
                    pos[1] < agent_radius * 1.2,
                    pos[1] > self.map_size - agent_radius * 1.2
                ])
                new_dones[i] = boundary_check or dones[i]
            # 障碍物碰撞终止
            for obs in self.obstacles:
                if np.linalg.norm(pos - obs.pos) < (agent_radius + obs.radius) * 1.2 and not new_dones[i]:    ####    不同的大小就该有不同的终止半径,因为惩罚也是
                    new_dones[i] = True

        # 3. 时间限制终止（统一处理）
        if self.current_time >= self.time_limit:
            for i in range(num_agents):
                if not new_dones[i]:
                    new_dones[i] = True

        # 3 计算奖励并记录日志
        for i in range(num_agents):
            if new_dones[i]:
                # 若已终止，奖励细节设为0
                reward_details.append({
                    "agent_id": i,
                    "pos": self.agents[i]["pos"],
                    "distance_to_target": np.linalg.norm(self.agents[i]["pos"] - self.target_pos),
                    "target_reward": 0.0,
                    "obstacle_penalty": 0.0,
                    "boundary_penalty": 0.0,
                    "agent_penalty": 0.0,
                    "arrival_reward": 0.0,
                    "total_reward": 0.0
                })
                continue  # 关键：跳过本轮循环的剩余代码

            agent_radius = self.agents[i]["radius"]   # 获取当前智能体半径
            # 只有未终止的智能体才能执行到这里
            pos = self.agents[i]["pos"]
            distance_to_target = np.linalg.norm(pos - self.target_pos)
            prev_distance = self.agents[i]["prev_distance_to_target"]

            # **关键修改**：仅在未终止时更新 prev_distance_to_target
            self.agents[i]["prev_distance_to_target"] = distance_to_target


            # 计算目标距离奖励（确保线性奖励始终计算，无论 prev_distance 是否为 None）
            linear_reward = self.calculate_linear_reward(distance_to_target)  # 新增辅助函数
            relative_reward = 0.0
            if prev_distance is not None:
                relative_reward = self.calculate_relative_reward(distance_to_target, prev_distance)
            target_reward = linear_reward + relative_reward

            # 计算各奖励分量
            obstacle_penalty = self.calculate_obstacle_penalty_for_agent(pos,agent_radius)  # 障碍物惩罚函数
            boundary_penalty = self.calculate_boundary_penalty(pos,agent_radius)   ##  边界惩罚函数
            agent_penalty = self.calculate_agent_penalty(i, pos, new_dones)        ##  智能体间的距离惩罚函数
            if target_positions[i] is not None and not self.agents[i]["arrived"]:
                arrival_reward = 200
                self.agents[i]["arrived"] = True   ###  ******  此处很关键,相当于清零条件,上面两个if第一个一直不满足,第二个一直满足,而当智能体被强制移动时第一个会满足

#也就是同时满足,会给20000的奖励,然后将第二个条件清零变成不满足,那么当下一步时第一个就是一直满足,第二个一直不满足,一直如此直到下一轮训练(一次训练只加一次目标点奖励),完美
     
            else:
                arrival_reward = 0.0
            # 累加总奖励
            total_reward = target_reward + obstacle_penalty + boundary_penalty + agent_penalty + arrival_reward
            rewards[i] = total_reward

            # 保存奖励细节（与主函数日志键名匹配）
            reward_details.append({
                "agent_id": i,
                "pos": pos,
                "distance_to_target": distance_to_target,
                "target_reward": target_reward,
                "obstacle_penalty": obstacle_penalty,
                "boundary_penalty": boundary_penalty,
                "agent_penalty": agent_penalty,
                "arrival_reward": arrival_reward,
                "total_reward": total_reward
            })

        # 5. 更新总奖励（带折扣）
        gamma = 0.9
        for i in range(num_agents):
            if not new_dones[i]:
                self.agents[i]["total_reward"] = self.agents[i]["total_reward"] * gamma + rewards[i]
            else:
                self.agents[i]["total_reward"] += rewards[i]  # 终止时直接累加剩余奖励

        # 更新上一时刻障碍物位置
        self.prev_obstacle_positions = current_obstacle_positions


        self.current_time += 1
        if self.current_time > self.time_limit:
            self.current_time = self.time_limit  # 防止溢出

        return self._get_obs(), rewards, new_dones, reward_details
    

    def is_within_range(self, pos1, pos2, radius):
        """判断是否在观测范围内（默认半径倍数，可调整）"""
        distance = np.linalg.norm(pos1 - pos2)
        return distance <= radius * 10  # 10倍半径范围，与原始代码一致



    def _get_obs(self):
        num_agents = len(self.agents)
        num_obstacles = len(self.obstacles)
        obs = []

        for i, agent in enumerate(self.agents):
            agent_pos = agent["pos"]
            agent_obs = np.zeros(0, dtype=np.float32)  # 动态构建观测向量

            agent_radius = self.agents[i]["radius"]   # 获取当前智能体半径

            # 1. 自身状态（位置+速度）
            agent_obs = np.concatenate([agent_obs, agent["pos"], agent["speed"]])  # 4维

            # 2. 其他智能体状态（位置+速度，带观测范围裁剪）
            for j, other_agent in enumerate(self.agents):
                if i == j:
                    continue  # 跳过自身
                if self.is_within_range(agent_pos, other_agent["pos"], agent_radius):
                    # 在观测范围内，添加其他智能体信息
                    agent_obs = np.concatenate([agent_obs, other_agent["pos"], other_agent["speed"]])  # 每个智能体4维
                else:
                    # 不在范围内，用零填充
                    agent_obs = np.concatenate([agent_obs, np.zeros(4, dtype=np.float32)])

            # 3. 障碍物状态（位置+速度，带观测范围裁剪）
            for obs_obj in self.obstacles:
                if self.is_within_range(agent_pos, obs_obj.pos, agent_radius):
                    # 在观测范围内，添加障碍物信息
                    obs_velocity = obs_obj.pos - self.prev_obstacle_positions[self.obstacles.index(obs_obj)]
                    agent_obs = np.concatenate([agent_obs, obs_obj.pos, obs_velocity])  # 每个障碍物4维
                else:
                    # 不在范围内，用零填充
                    agent_obs = np.concatenate([agent_obs, np.zeros(4, dtype=np.float32)])

            # 4. 目标位置（固定2维）
            agent_obs = np.concatenate([agent_obs, self.target_pos])

            # 确保观测维度正确（兼容原有逻辑）
            expected_dim = 4 + 4*(num_agents-1) + 4*num_obstacles + 2
            if agent_obs.shape[0] != expected_dim:
                # 补零处理（应对动态数量差异）
                agent_obs = np.concatenate([agent_obs, np.zeros(expected_dim - agent_obs.shape[0])])

            obs.append(agent_obs)

        return obs


