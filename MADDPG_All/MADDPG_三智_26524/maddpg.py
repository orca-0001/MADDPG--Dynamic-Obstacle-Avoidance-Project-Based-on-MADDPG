

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class SumTree:
    """SumTree数据结构用于高效优先级采样"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
            
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, p)
        
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.max_priority = 1.0
        self.epsilon = 1e-6  # 避免优先级为0

    def beta(self):
        """重要性采样权重系数随时间增加"""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def add(self, experience):
        """添加经验到缓冲区"""
        priority = self.max_priority
        self.tree.add(priority, experience)

    def sample(self, batch_size):
        """采样一批经验"""
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size
        beta = self.beta()

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        # 计算重要性采样权重
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()  # 归一化
        
        return batch, idxs, is_weights

    def update_priorities(self, idxs, priorities):
        """更新采样经验的优先级"""
        priorities = (priorities + self.epsilon) ** self.alpha
        for idx, priority in zip(idxs, priorities):
            self.tree.update(idx, priority)
            if priority > self.max_priority:
                self.max_priority = priority
                
        self.frame += 1

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),  
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),    # 保留层归一化
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        return self.net(x)

class MADDPG:
    def __init__(self, state_dim, action_dim, num_agents, device, epsilon=0.8):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.device = device
        self.epsilon = epsilon  # 初始化 epsilon

        # 初始化Actor和Critic网络
        self.actors = [Actor(state_dim, action_dim).to(device) for _ in range(num_agents)]
        self.critics = [Critic(state_dim * num_agents, action_dim * num_agents).to(device) for _ in range(num_agents)]
        self.target_actors = [Actor(state_dim, action_dim).to(device) for _ in range(num_agents)]
        self.target_critics = [Critic(state_dim * num_agents, action_dim * num_agents).to(device) for _ in range(num_agents)]

        # 存储每个智能体的损失值
        self.critic_losses = [[] for _ in range(num_agents)]
        self.actor_losses = [[] for _ in range(num_agents)] 

        # 初始化优化器
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=1e-4) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=1e-3) for critic in self.critics]

        # 优先经验回放缓冲区（每个智能体独立）
        self.replay_buffers = [
            PrioritizedReplayBuffer(capacity=6000000) 
            for _ in range(num_agents)
        ]

        # 软更新参数
        self.tau = 0.05

        # 记录每个智能体进入终止状态的次数
        self.agent_termination_count = [0] * num_agents

    def act(self, states, dones, noise=0.5):
        actions = []
        for i, state in enumerate(states):

            if dones[i]:  # 若智能体已终止，直接返回零动作
                action = np.zeros(self.action_dim)
            else:  # 未终止时正常生成动作（探索或策略）
                if np.random.rand() < self.epsilon:  # 探索动作
                    linear_vel = np.random.uniform(-1, 1, size=2)
                    action = linear_vel.flatten()
                else:  # 策略动作 + 噪声
                    state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
                    action = self.actors[i](state_tensor).detach().cpu().numpy()
                    action += noise * np.random.normal(size=self.action_dim)
                    action = np.clip(action, -1, 1).flatten()
            # 确保动作在终止状态下为零
            actions.append(action)

        self.epsilon = max(0.05, self.epsilon - 0.0005)
        return actions

    def update(self, agent_id):
        # 从优先经验回放缓冲区采样
        batch, idxs, is_weights = self.replay_buffers[agent_id].sample(128)
        
        # 解包批处理数据
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = zip(*batch)
        
        # 转换为PyTorch张量
        states_tensor = torch.tensor(np.array(states_batch), dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(np.array(actions_batch), dtype=torch.float32).to(self.device)
        rewards_tensor = torch.tensor(np.array(rewards_batch), dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(np.array(next_states_batch), dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(np.array(dones_batch), dtype=torch.float32).to(self.device)
        is_weights_tensor = torch.tensor(is_weights, dtype=torch.float32).to(self.device).unsqueeze(1)

        batch_size = states_tensor.shape[0]

        with torch.no_grad():
            # 生成目标动作
            next_actions = []
            for i in range(self.num_agents):
                next_agent_actions = self.target_actors[i](next_states_tensor[:, i])
                next_actions.append(next_agent_actions)
            next_actions = torch.cat(next_actions, dim=1)

            # 计算目标 Q 值
            all_next_states = next_states_tensor.view(batch_size, -1)
            target_q = self.target_critics[agent_id](all_next_states, next_actions)
            target_q = rewards_tensor[:, agent_id].unsqueeze(1) + 0.99 * target_q * (1 - dones_tensor[:, agent_id].unsqueeze(1))

        # 计算当前 Q 值
        current_actions = actions_tensor.view(batch_size, -1)   # 所有智能体的动作拼接
        all_states = states_tensor.view(batch_size, -1)    # 所有智能体的状态拼接
        current_q = self.critics[agent_id](all_states, current_actions)

        # 计算Critic损失（使用重要性采样权重）
        td_errors = current_q - target_q.detach()
        critic_loss = (is_weights_tensor * td_errors.pow(2)).mean()
        
        # 更新Critic
        self.critic_optimizers[agent_id].zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), 0.5)
        self.critic_optimizers[agent_id].step()

        # 更新Actor
        new_actions = []
        for i in range(self.num_agents):
            if i == agent_id:
                new_action = self.actors[i](states_tensor[:, i])
            else:
                new_action = self.actors[i](states_tensor[:, i]).detach()
            new_actions.append(new_action)
        new_actions = torch.cat(new_actions, dim=1)

        actor_loss = -self.critics[agent_id](all_states, new_actions).mean()
        self.actor_optimizers[agent_id].zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), 0.5)
        self.actor_optimizers[agent_id].step()

        # 记录损失值
        self.critic_losses[agent_id].append(critic_loss.item())
        self.actor_losses[agent_id].append(actor_loss.item())

        # 更新优先级（使用TD误差）
        with torch.no_grad():
            new_priorities = torch.abs(td_errors).squeeze().cpu().numpy()
        self.replay_buffers[agent_id].update_priorities(idxs, new_priorities)

        # 软更新目标网络
        self._soft_update(self.actors[agent_id], self.target_actors[agent_id])
        self._soft_update(self.critics[agent_id], self.target_critics[agent_id])

    def _soft_update(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics]
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(checkpoint['actors'][i])
            self.critics[i].load_state_dict(checkpoint['critics'][i])

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon




