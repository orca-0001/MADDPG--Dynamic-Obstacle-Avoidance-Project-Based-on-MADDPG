

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque


#############这个更稳定,上限更高,但是更慢(可用)
    

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
    


# 改进后的Critic（集成其他智能体信息）
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


        self.critic_losses = [[] for _ in range(num_agents)]  # 定义每个智能体的Critic损失列表,后续会把损失值添加到这个列表里面
        self.actor_losses = [[] for _ in range(num_agents)]   # 每个智能体的Actor损失

        # 初始化优化器
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=1e-4) for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=1e-3) for critic in self.critics]

        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=3000000)

        # 软更新参数
        self.tau = 0.05

        # 新增：记录每个智能体进入终止状态的次数
        self.agent_termination_count = [0] * num_agents


###################################这里的noise=0.5说的不是令noise初始值为零然后一步步衰减,而是设置默认值,在没有显式传递noise的值时就会使用0.5,
#############要改进噪声初始值在这里        new_noise = max(0.1, 1.0 - episode / 10000),此处的1.0就是初始值



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

        self.epsilon = max(0.05, self.epsilon - 0.0003)
        return actions


    def update(self, experiences, agent_id):
        states, actions, rewards, next_states, dones = experiences

        ##### 将元组转换为列表 这样才可以修改actions的类型,要变成数组才能确保统一,而元组类型的数据不能修改其类型,所以先转换成列表,再修改成数组
        actions = list(actions)
        # 确保所有动作是 numpy.ndarray
        for i in range(len(actions)):
            if not isinstance(actions[i], np.ndarray):
                actions[i] = np.array(actions[i]).flatten()
        
        # 转换为统一的二维数组
        actions = np.array(actions)
        states = np.array(states)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)



        # 转换为 PyTorch 张量
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)

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




        # 更新 Critic
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_optimizers[agent_id].zero_grad()
        critic_loss.backward()

        nn.utils.clip_grad_norm_(self.critics[agent_id].parameters(), 0.5)  # 添加梯度裁剪  

        self.critic_optimizers[agent_id].step()



        # 更新 Actor
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

        nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), 0.5)# 添加梯度裁剪

        self.actor_optimizers[agent_id].step()



        # 更新 Critic(前面已经计算过了,只用保存即可)
        #critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_losses[agent_id].append(critic_loss.item())  # 记录Critic损失

        # 更新 Actor(前面已经计算过了,只用保存即可)
        #actor_loss = -self.critics[agent_id](all_states, new_actions).mean()
        self.actor_losses[agent_id].append(actor_loss.item())    # 记录Actor损失

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




