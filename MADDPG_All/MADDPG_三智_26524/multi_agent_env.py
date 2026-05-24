

import gym
from gym import spaces
import numpy as np
from scipy.optimize import fsolve
import random

class Obstacle:
    def __init__(self, obstacle_type, start_pos, **kwargs):
        """
        障碍物类
        :param obstacle_type: 'linear'（直线运动）或 'curve'（曲线运动）或 'ellipse'（椭圆运动）或 'back_and_forth'（两点来回运动）
        :param start_pos: 初始位置坐标
        :param kwargs: 其他参数（速度、半径、椭圆参数等）
        """
        self.type = obstacle_type
        self.radius = np.float32(kwargs.get('radius', 0.5))
        self.pos = np.array(start_pos, dtype=np.float32)
        self.speed = np.float32(kwargs.get('speed', 0.05))
        self.map_size = np.float32(kwargs.get('map_size', 20))
        
        # 椭圆运动参数
        if self.type == 'ellipse':
            self.center = np.array(kwargs.get('center', [13.50, 13.50]), dtype=np.float32)
            self.a = np.float32(kwargs.get('a', 2.0))
            self.b = np.float32(kwargs.get('b', 0.5))
            self.angular_speed = np.float32(kwargs.get('angular_speed', 0.03))
            self.theta = np.deg2rad(kwargs.get('theta', 135))  # 椭圆的旋转角度（弧度）
            self.phi = np.pi  # 初始相位为π，对应右侧顶点
            self.pos = self._calculate_ellipse_position(self.phi)  # 计算初始位置
            self.pos = np.clip(self.pos, self.radius, self.map_size - self.radius)  # 边界裁剪
        
        # 来回运动参数
        elif self.type == 'back_and_forth':        
            self.start = np.array(start_pos, dtype=np.float32)
            self.end = np.array(kwargs.get('target', start_pos), dtype=np.float32)
            self.target = self.end  # 初始目标为终点
            self.direction = self._get_direction()
        
        # 随机运动参数
        else:
            self.direction = np.random.uniform(0, 2*np.pi)  # 初始随机方向（弧度）
            self.curvature = 0.0
            self.change_counter = 0
            self.change_interval = random.randint(10, 20)  # 随机变化间隔
        
        # 边界裁剪初始化
        self.pos = np.clip(self.pos, self.radius, self.map_size - self.radius)

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

    def update_position(self, target_reached, target_pos, all_obstacles, t):
        """更新障碍物位置，并处理避让逻辑"""
        # 1. 更新位置
        self._update_movement(t)
        
        # 2. 处理边界避让
        self._handle_boundary_collision()
        
        # 3. 处理障碍物间避让
        self._handle_obstacle_collision(all_obstacles)
        
        # 4. 处理目标点避让
        if target_reached:
            self._handle_target_collision(target_pos)
        
    
    def _update_movement(self, t):
        """根据运动类型更新位置"""
        if self.type == 'linear' or self.type == 'curve':
            # 随机运动障碍物
            self._update_random_movement()
        elif self.type == 'ellipse':
            # 椭圆运动
            self._update_ellipse_position(t)
        elif self.type == 'back_and_forth':
            # 来回直线运动
            self._update_back_and_forth_position()
    
    def _update_random_movement(self):
        """更新随机运动障碍物位置"""
        # 减少计数器，检查是否需要改变运动参数
        self.change_counter -= 1
        if self.change_counter <= 0:
            self.change_counter = random.randint(10, 20)  # 重置计数器
            
            if self.type == 'linear':
                # 直线运动：随机改变方向 (0-90度)
                angle_change = np.random.uniform(0, np.pi/2)
                self.direction = (self.direction + angle_change) % (2*np.pi)
            elif self.type == 'curve':
                # 曲线运动：随机改变曲率 (-0.1 到 0.1)
                self.curvature = np.random.uniform(-0.1, 0.1)
        
        # 应用运动
        if self.type == 'linear':
            # 直线运动
            dx = np.cos(self.direction) * self.speed
            dy = np.sin(self.direction) * self.speed
            self.pos += np.array([dx, dy], dtype=np.float32)
        elif self.type == 'curve':
            # 曲线运动：更新方向并移动
            self.direction += self.curvature
            dx = np.cos(self.direction) * self.speed
            dy = np.sin(self.direction) * self.speed
            self.pos += np.array([dx, dy], dtype=np.float32)
        
        # 边界裁剪
        self.pos = np.clip(self.pos, self.radius, self.map_size - self.radius)
    
    def _update_ellipse_position(self, t):
        """椭圆运动更新逻辑"""
        phi = self.angular_speed * t + np.pi  # 初始相位为 π（180°），对应右侧顶点且参数角随时间变化
        self.pos = self._calculate_ellipse_position(phi)
        self.pos = np.clip(self.pos, self.radius, self.map_size - self.radius)  # 边界裁剪

    def _update_back_and_forth_position(self):
        """来回直线运动更新逻辑"""
        distance = np.linalg.norm(self.pos - self.target)
        
        if distance > self.radius + 1e-6:
            self.pos += self.direction * self.speed
            self.pos = np.clip(self.pos, self.radius, self.map_size - self.radius)
        else:
            # 切换目标点为起点/终点
            self.target = self.start if np.array_equal(self.target, self.end) else self.end
            self.direction = self._get_direction()  # 重新计算方向
    
    def _handle_boundary_collision(self):
        """处理边界避让"""
            
        # 检查与边界的距离
        left_dist = self.pos[0] - self.radius
        right_dist = self.map_size - self.radius - self.pos[0]
        bottom_dist = self.pos[1] - self.radius
        top_dist = self.map_size - self.radius - self.pos[1]
        
        # 如果接近边界，改变方向
        if min(left_dist, right_dist, bottom_dist, top_dist) < 0.3:
            # 确定最近的边界
            if left_dist == min(left_dist, right_dist, bottom_dist, top_dist):
                self.direction = np.pi - self.direction  # 左右翻转
            elif right_dist == min(left_dist, right_dist, bottom_dist, top_dist):
                self.direction = np.pi - self.direction  # 左右翻转
            elif bottom_dist == min(left_dist, right_dist, bottom_dist, top_dist):
                self.direction = -self.direction  # 上下翻转
            elif top_dist == min(left_dist, right_dist, bottom_dist, top_dist):
                self.direction = -self.direction  # 上下翻转
            
            # 确保方向在 [0, 2π) 范围内
            self.direction = self.direction % (2*np.pi)
    
    def _handle_obstacle_collision(self, all_obstacles):
        """处理与其他障碍物的避让"""
        for other in all_obstacles:
            if other is not self:  # 不与自己比较
                dist = np.linalg.norm(self.pos - other.pos)
                if dist < self.radius + other.radius + 0.3:
                    # 椭圆障碍物不改变方向
                    if self.type == 'ellipse':
                        continue
                        
                    # 计算反弹方向（远离其他障碍物）
                    direction_to_other = other.pos - self.pos
                    angle_to_other = np.arctan2(direction_to_other[1], direction_to_other[0])
                    self.direction = (angle_to_other + np.pi) % (2*np.pi)  # 反转方向
    
    def _handle_target_collision(self, target_pos):
        """处理与目标点的避让"""
            
        dist_to_target = np.linalg.norm(self.pos - target_pos)
        if dist_to_target < 1.2:
            # 计算反弹方向（远离目标点）
            direction_to_target = target_pos - self.pos
            angle_to_target = np.arctan2(direction_to_target[1], direction_to_target[0])
            self.direction = (angle_to_target + np.pi) % (2*np.pi)  # 反转方向

class MultiAgentEnv(gym.Env):
    def __init__(self):
        super(MultiAgentEnv, self).__init__()

        # 智能体列表
        self.agents = [
            {
                "id": 0,
                "pos": np.array([2.0, 6.0], dtype=np.float32),
                "speed": np.zeros(2, dtype=np.float32),
                "arrived": False,
                "total_reward": 0.0,
                "prev_distance_to_target": None,
                "arrived_by_target": False,
                "radius": 0.5, 
            },
            {
                "id": 1,
                "pos": np.array([2.0, 3.0], dtype=np.float32),
                "speed": np.zeros(2, dtype=np.float32),
                "arrived": False,
                "total_reward": 0.0,
                "prev_distance_to_target": None,
                "arrived_by_target": False,
                "radius": 0.5, 
            },
            {
                "id": 2,
                "pos": np.array([5.0, 2.0], dtype=np.float32),
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
        
        # 统一管理所有障碍物（包含四种类型）
        self.obstacles = [
            # 静态障碍物
            # Obstacle('linear', start_pos=[9, 10], speed=0.0, radius=0.5),
            Obstacle('linear', start_pos=[6, 13], speed=0.0, radius=0.5),
            Obstacle('linear', start_pos=[12, 5], speed=0.0, radius=0.5),

            Obstacle('linear', start_pos=[14, 17], speed=0.0, radius=0.5),
            Obstacle('curve', start_pos=[17, 14], speed=0.0, radius=0.5),

            # 椭圆运动障碍物
            Obstacle('ellipse', start_pos=[11, 7], center=[10, 15], a=2, b=0.5, angular_speed=0.03, theta=240,radius=0.5),
            Obstacle('ellipse', start_pos=[14, 12], center=[15, 8], a=2, b=0.8, angular_speed=0.03, theta=90, radius=0.5),
            # 来回运动障碍物
            Obstacle('back_and_forth', start_pos=[6, 11], target=[11, 6], speed=0.05, radius=0.5),
        ]

        # 观测空间维度
        num_agents = len(self.agents)
        num_obstacles = len(self.obstacles)
        self.observation_space = spaces.Box(
            low=-1000,
            high=1000,
            shape=(4*num_agents + 32 + 2,),
            dtype=np.float32
        )

        # 保存上一时刻的障碍物位置
        self.prev_obstacle_positions = [obs.pos.copy() for obs in self.obstacles]
        
        # 新增：目标点到达标志
        self.target_reached = False
        
        self.agent1_speed = np.array([0.0, 0.0])
        self.agent2_speed = np.array([0.0, 0.0])
        self.target_pos = np.array([18.0, 18.0])
        self.map_size = 20
        self.time_limit = 500
        self.current_time = 0
        self.agent_speed_limit = 0.07
        self.step_penalty = 0.1

    def reset(self):
        self.current_time = 0
        self.arrived_by_target = [False] * len(self.agents)
        self.target_reached = False  # 重置目标点到达标志

        self.agents = [
            {
                "id": 0,
                "pos": np.array([2.0, 6.0], dtype=np.float32),
                "speed": np.zeros(2, dtype=np.float32),
                "arrived": False,
                "total_reward": 0.0,
                "prev_distance_to_target": None,
                "arrived_by_target": False,
                "radius": 0.5, 
            },
            {
                "id": 1,
                "pos": np.array([2.0, 3.0], dtype=np.float32),
                "speed": np.zeros(2, dtype=np.float32),
                "arrived": False,
                "total_reward": 0.0,
                "prev_distance_to_target": None,
                "arrived_by_target": False,
                "radius": 0.5, 
            },
            {
                "id": 2,
                "pos": np.array([5.0, 2.0], dtype=np.float32),
                "speed": np.zeros(2, dtype=np.float32),
                "arrived": False,
                "total_reward": 0.0,
                "prev_distance_to_target": None,
                "arrived_by_target": False,
                "radius": 0.5, 
            }
        ]
        
        # 重置障碍物位置
        self.obstacles = [
            # 静态障碍物
            # Obstacle('linear', start_pos=[9, 10], speed=0.0, radius=0.5),
            Obstacle('linear', start_pos=[6, 13], speed=0.0, radius=0.5),
            Obstacle('linear', start_pos=[12, 5], speed=0.0, radius=0.5),

            Obstacle('linear', start_pos=[14, 17], speed=0.0, radius=0.5),
            Obstacle('curve', start_pos=[17, 14], speed=0.0, radius=0.5),

            # 椭圆运动障碍物
            Obstacle('ellipse', start_pos=[11, 7], center=[10, 15], a=2, b=0.5, angular_speed=0.03, theta=240,radius=0.5),
            Obstacle('ellipse', start_pos=[14, 12], center=[15, 8], a=2, b=0.8, angular_speed=0.03, theta=90, radius=0.5),
            # 来回运动障碍物
            Obstacle('back_and_forth', start_pos=[6, 11], target=[11, 6], speed=0.05, radius=0.5),
        ]

        self.prev_obstacle_positions = [obs.pos.copy() for obs in self.obstacles]
        return self._get_obs()

    def calculate_obstacle_penalty_for_agent(self, pos, agent_radius):
        """计算单个智能体的障碍物惩罚"""
        obstacle_penalty = 0.0
        min_obs_dist = np.inf
        for obs in self.obstacles:
            dist = np.linalg.norm(pos - obs.pos)
            if dist < min_obs_dist:
                min_obs_dist = dist
            if dist < 1.7:
                obstacle_penalty -= self.calculate_obstacle_penalty(dist)
        return obstacle_penalty
    
    def calculate_boundary_penalty(self, pos, agent_radius):
        """计算边界惩罚"""
        distance_to_boundary = min(pos[0], self.map_size - pos[0], pos[1], self.map_size - pos[1])
        if distance_to_boundary < agent_radius * 3:
            return -(agent_radius * 3 - distance_to_boundary) * 300.0
        return 0.0

    def calculate_agent_penalty(self, agent_id, pos, new_dones):
        """计算智能体间惩罚"""
        penalty = 0.0
        for j in range(len(self.agents)):
            if agent_id != j and not new_dones[agent_id] and not self.agents[j]["arrived_by_target"]:
                distance = np.linalg.norm(pos - self.agents[j]["pos"])
                if distance < 1.6:
                    penalty -= (1.6 - distance) * 500.0
        return penalty

    def calculate_obstacle_penalty(self, distance):
        # 定义外圈和内圈的距离范围和惩罚值
        outer_min_distance = 1.31
        outer_max_distance = 1.7
        outer_max_penalty = 340

        inner_min_distance = 0.3
        inner_max_distance = 1.3
        inner_min_penalty = 500
        inner_max_penalty = 800

        if outer_min_distance <= distance <= outer_max_distance:
            return outer_max_penalty / (outer_min_distance - outer_max_distance) * (distance - outer_max_distance)
        elif inner_min_distance <= distance < inner_max_distance:
            return (inner_max_penalty - inner_min_penalty) / (inner_min_distance - inner_max_distance) * (distance - inner_max_distance) + inner_min_penalty
        else:
            return inner_max_penalty

    def calculate_linear_reward(self, distance):
        min_distance = 0.1
        max_distance = 12
        max_reward = 2
        min_reward = 1
        return max_reward - (distance - min_distance) * (max_reward - min_reward) / (max_distance - min_distance)

    def calculate_relative_reward(self, distance, prev_distance):
        if distance < prev_distance:
            return (prev_distance - distance) * 300
        else:
            return -2 * self.calculate_linear_reward(distance) - (distance - prev_distance) * 600 * 3

    def step(self, actions, dones):
        num_agents = len(self.agents)
        rewards = [0.0 for _ in range(num_agents)]
        new_dones = [False for _ in range(num_agents)]
        target_positions = [None for _ in range(num_agents)]
        reward_details = []

        # 0. 更新障碍物位置（使用新的运动模式）
        for obs in self.obstacles:
            # 传递目标点到达状态和目标点位置，以及当前时间步
            obs.update_position(self.target_reached, self.target_pos, self.obstacles, self.current_time)
        
        # 保存当前障碍物位置用于下一时刻计算速度
        current_obstacle_positions = [obs.pos.copy() for obs in self.obstacles]
        
        # 1. 计算临时位置(智能体)
        for i in range(num_agents):
            if dones[i]:
                continue

            agent_radius = self.agents[i]["radius"]

            # 如果已经到达目标点，直接标记为终止
            if np.linalg.norm(self.agents[i]["pos"] - self.target_pos) < agent_radius * 0.2:
                new_dones[i] = True
                self.agents[i]["arrived_by_target"] = True
                self.target_reached = True  # 标记目标点已被到达
                continue

            linear_vel = actions[i][:2]
            direction = linear_vel / (np.linalg.norm(linear_vel) + 1e-8)
            temp_pos = self.agents[i]["pos"] + direction * np.linalg.norm(linear_vel) * self.agent_speed_limit
            temp_pos = np.clip(temp_pos, agent_radius, self.map_size - agent_radius)
            
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
                self.target_reached = True  # 标记目标点已被到达

        # 2. 计算终止状态（边界/碰撞/时间）
        for i in range(num_agents):
            if new_dones[i]:
                continue
            pos = self.agents[i]["pos"]
            agent_radius = self.agents[i]["radius"]

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
                if np.linalg.norm(pos - obs.pos) < (agent_radius + obs.radius) * 1.2 and not new_dones[i]:
                    new_dones[i] = True

        # 3. 时间限制终止
        if self.current_time >= self.time_limit:
            for i in range(num_agents):
                if not new_dones[i]:
                    new_dones[i] = True

        # 3 计算奖励并记录日志
        for i in range(num_agents):
            if new_dones[i]:
                reward_details.append({
                    "agent_id": i,
                    "pos": self.agents[i]["pos"],
                    "distance_to_target": np.linalg.norm(self.agents[i]["pos"] - self.target_pos),
                    "target_reward": 0.0,
                    "obstacle_penalty": 0.0,
                    "boundary_penalty": 0.0,
                    "agent_penalty": 0.0,
                    "arrival_reward": 0.0,
                    "speed_penalty": 0.0,
                    "total_reward": 0.0
                })
                continue

            agent_radius = self.agents[i]["radius"]
            pos = self.agents[i]["pos"]
            distance_to_target = np.linalg.norm(pos - self.target_pos)
            prev_distance = self.agents[i]["prev_distance_to_target"]

            self.agents[i]["prev_distance_to_target"] = distance_to_target

            linear_reward = self.calculate_linear_reward(distance_to_target)
            relative_reward = 0.0
            if prev_distance is not None:
                relative_reward = self.calculate_relative_reward(distance_to_target, prev_distance)
            target_reward = linear_reward + relative_reward

            obstacle_penalty = self.calculate_obstacle_penalty_for_agent(pos, agent_radius)
            boundary_penalty = self.calculate_boundary_penalty(pos, agent_radius)
            # agent_penalty = self.calculate_agent_penalty(i, pos, new_dones)
            agent_penalty = 0


            if target_positions[i] is not None and not self.agents[i]["arrived"]:
                arrival_reward = 800
                self.agents[i]["arrived"] = True
            else:
                arrival_reward = 0.0
                

            # ====== 新增：动态速度惩罚 ======
            movement_speed = np.linalg.norm(actions[i][:2]) * self.agent_speed_limit
            low_speed_threshold = self.agent_speed_limit * 0.3
            high_speed_threshold = self.agent_speed_limit * 0.7
            speed_penalty = 0.0
            
            if movement_speed < low_speed_threshold:
                speed_penalty = (low_speed_threshold - movement_speed) * 8000
            # elif movement_speed > high_speed_threshold:
            #     speed_penalty = (movement_speed - high_speed_threshold) * 8000
            else:
                speed_penalty = 0.1
            # ====== 结束新增 ======
            
            total_reward = target_reward + obstacle_penalty + boundary_penalty + agent_penalty + arrival_reward - speed_penalty

            rewards[i] = total_reward

            reward_details.append({
                "agent_id": i,
                "pos": pos,
                "distance_to_target": distance_to_target,
                "target_reward": target_reward,
                "obstacle_penalty": obstacle_penalty,
                "boundary_penalty": boundary_penalty,
                "agent_penalty": agent_penalty,
                "arrival_reward": arrival_reward,
                "total_reward": total_reward,
                "speed_penalty": speed_penalty
            })

        # 5. 更新总奖励（带折扣）
        gamma = 0.95
        for i in range(num_agents):
            if not new_dones[i]:
                self.agents[i]["total_reward"] = self.agents[i]["total_reward"] * gamma + rewards[i]
            else:
                self.agents[i]["total_reward"] += rewards[i]

        # 更新上一时刻障碍物位置
        self.prev_obstacle_positions = current_obstacle_positions

        self.current_time += 1
        if self.current_time > self.time_limit:
            self.current_time = self.time_limit

        # 在返回之前添加完整环境状态的记录
        full_state = {
            "agent_positions": [agent["pos"].copy() for agent in self.agents],
            "obstacle_positions": [obs.pos.copy() for obs in self.obstacles],
            "agent_speeds": [agent["speed"].copy() for agent in self.agents],
            "agent_arrived": [agent["arrived"] for agent in self.agents],
            "agent_arrived_by_target": [agent["arrived_by_target"] for agent in self.agents],
            "agent_prev_distances": [agent["prev_distance_to_target"] for agent in self.agents],
            "current_time": self.current_time,
            "target_reached": self.target_reached
        }
        
        return self._get_obs(), rewards, new_dones, reward_details, full_state  # 添加 full_state


    
    def is_within_range(self, pos1, pos2, radius):
        distance = np.linalg.norm(pos1 - pos2)
        return distance <= radius * 10

    def _get_obs(self):
        num_agents = len(self.agents)
        num_obstacles = len(self.obstacles)
        obs = []

        for i, agent in enumerate(self.agents):
            agent_pos = agent["pos"]
            agent_obs = np.zeros(0, dtype=np.float32)
            agent_radius = self.agents[i]["radius"]

            # 1. 自身状态
            agent_obs = np.concatenate([agent_obs, agent["pos"], agent["speed"]])

            # 2. 其他智能体状态
            for j, other_agent in enumerate(self.agents):
                if i == j:
                    continue
                if self.is_within_range(agent_pos, other_agent["pos"], agent_radius):
                    agent_obs = np.concatenate([agent_obs, other_agent["pos"], other_agent["speed"]])
                else:
                    agent_obs = np.concatenate([agent_obs, np.zeros(4, dtype=np.float32)])

            # 3. 障碍物状态
            for obs_obj in self.obstacles:
                if self.is_within_range(agent_pos, obs_obj.pos, agent_radius):
                    obs_velocity = obs_obj.pos - self.prev_obstacle_positions[self.obstacles.index(obs_obj)]
                    agent_obs = np.concatenate([agent_obs, obs_obj.pos, obs_velocity])
                else:
                    agent_obs = np.concatenate([agent_obs, np.zeros(4, dtype=np.float32)])

            # 4. 目标位置
            agent_obs = np.concatenate([agent_obs, self.target_pos])

            # 确保观测维度正确
            expected_dim = 4 + 4*(num_agents-1) + 32 + 2
            if agent_obs.shape[0] != expected_dim:
                agent_obs = np.concatenate([agent_obs, np.zeros(expected_dim - agent_obs.shape[0])])

            obs.append(agent_obs)

        return obs




