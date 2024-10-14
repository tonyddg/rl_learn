import torch
from torch import nn
from torch import optim

import numpy as np

from dataclasses import dataclass
from .Utility import *

# 环境 Cart Pole
# Acion: 离散值 {0, 1}
# State: np 数组 (4,)
ACTION_NUM = 2
STATE_DIM = 4

@dataclass
class HyperParam:
    ## 基础
    # 学习率
    alpha: float = 2e-3
    # 回报折扣率
    gamma: float = 0.98
    
    ## epsilon-greedy
    # 随机决策概率
    # epsilon: float = 0.01
    epsilon_start: float = 0.90
    epsilon_final: float = 0.01
    episode_decay: int = 50
    
    ## Double DQN
    # Target Network 更新中, DQN 网络权重
    tau: float = 0.99
    # Target Network 更新频率 (单位为 update 方法调用次数, 即 batch)
    tn_update_preiod: int = 10

    ## 网络
    # 隐藏层大小
    hidden_dim: int = 128
    # 状态参数数
    state_dim: int = STATE_DIM
    # 可执行操作数
    action_num: int = ACTION_NUM

    ## 经验回放
    # 批次大小
    batch_size: int = 64
    # 回放队列容量
    reply_size: int = 10000
    # 开始训练所需回放
    minimal_size: int = 500

class QNet(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int = STATE_DIM, action_dim: int = ACTION_NUM) -> None:
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, X):
        return self.dense(X)

class BaseDQN:

    # 初始化部分

    def __init__(self, hparams: HyperParam) -> None:
        
        self.hparams = hparams

        self.train_batch_count = 0
        self.epsilon = self.hparams.epsilon_start
        self.reply_queue = ReplyQueue(self.hparams.reply_size)

        self.q_network = QNet(
            self.hparams.hidden_dim,
            self.hparams.state_dim,
            self.hparams.action_num
        )

        # Target Network 保持在评估模式
        self.target_network = QNet(
            self.hparams.hidden_dim,
            self.hparams.state_dim,
            self.hparams.action_num
        )
        self.target_network.eval()

        self.optimizer = optim.Adam( # type: ignore
            self.q_network.parameters(),
            self.hparams.alpha
        )
        # 使用均值 MSE 作为损失函数
        self.loss_fn = nn.MSELoss(reduction = "mean")

    # 决策部分

    def _take_random_action(self, state: torch.Tensor) -> torch.Tensor:
        # 取 0 到 action_num 的随机数
        return torch.randint(0, self.hparams.action_num, (state.size()[0], 1))

    def _take_valuable_action(self, state: torch.Tensor) -> torch.Tensor:
        # 取第一维的 Argmax (非 Batch), 但保持维度 (维度长度变为 1)
        res1 = self.q_network(state)
        res2 = torch.argmax(res1, 1, True)
        return res2
        # return torch.argmax(self.q_network(state), 1, True)

    def take_action_vectorize(self, state: torch.Tensor) -> torch.Tensor:
        '''
        向量化 epsilon-greedy 决策  
        传入张量, 同时传出张量
        '''
        with torch.no_grad():
            if random.random() > self.epsilon:
                return self._take_valuable_action(state)
            else:
                return self._take_random_action(state)

    def take_action_single(self, state: np.ndarray) -> np.ndarray:
        '''
        单次 epsilon-greedy 决策  
        传入 Numpy 数组, 同时传出 Numpy 数组
        '''
        state_tensor = torch.tensor(state).view(1, -1)
        action_tensor = self.take_action_vectorize(state_tensor)
        return action_tensor[0].numpy()

    # 训练部分

    def _get_td_target(self, transition: Transition) -> torch.Tensor:
        '''
        获取 TD Target, 不会计算梯度
        '''
        with torch.no_grad():
            # 使用 DQN 预测动作
            valuable_action = self._take_valuable_action(transition.next_state)
            batch_size = valuable_action.size()[0]

            # 使用 Target Network 预测价值
            target_output = self.target_network(transition.next_state)
            
            # 取 Target Network 中 DQN 的预测动作作为 Q* 的预测
            mix_predict = torch.zeros((batch_size, 1))
            for i in range(batch_size):
                mix_predict[i, 0] = target_output[i, valuable_action[i]]

            return transition.reward + self.hparams.gamma * mix_predict * (1 - transition.done)

    def _get_predict(self, transition: Transition) -> torch.Tensor:
        '''
        获取模型预测
        '''
        # 类似 _take_valuable_action, torch 的 max 返回值为二元组 (最大值, 最大值索引)
        return torch.max(self.q_network(transition.state), 1, True)[0]

    def _update_target_network(self):
        '''
        更新 Target Network 参数
        '''
        q_params = self.q_network.state_dict()

        # debug, 使用直接替换法
        # t_params = self.target_network.state_dict()

        # for key in q_params.keys():
        #     t_params[key] *= 1 - self.hparams.tau
        #     t_params[key] += q_params[key] * self.hparams.tau
        
        # self.target_network.load_state_dict(t_params)
        self.target_network.load_state_dict(q_params)

    def _batch_update(self, transition: Transition) -> float:
        '''
        进行一个批次的训练
        '''

        self.q_network.train()

        self.train_batch_count += 1

        # 计算 TD 目标与模型预测
        td_target = self._get_td_target(transition)
        predict = self._get_predict(transition)

        # 梯度下降更新模型
        loss = self.loss_fn(predict, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新 Target Network
        if self.train_batch_count % self.hparams.tn_update_preiod == 0:
            self._update_target_network()

        self.q_network.eval()

        return loss.item()

    def update(self, transition: Transition):
        '''
        基于经验回放更新模型

        * 成功更新时返回 loss
        * 失败时返回 False
        '''

        self.reply_queue.append(transition)

        if self.reply_queue.size() < self.hparams.minimal_size:
            return False
        else:
            batch_transition = self.reply_queue.sample(self.hparams.batch_size)
            return self._batch_update(batch_transition)

    # 片段结束后更新

    def  _update_epsilon(self, episode: int):
        '''
        epsilon-greedy 决策中, 使用指数规律更新 epsilon
        '''
        self.epsilon = \
            self.hparams.epsilon_final + (self.hparams.epsilon_start - self.hparams.epsilon_final) \
            * np.exp(episode / self.hparams.episode_decay)
        self.epsilon = max(self.hparams.epsilon_final, self.hparams.epsilon_start - episode / self.hparams.episode_decay)

        # self.epsilon = self.hparams.epsilon_final # debug 使用固定 epsilon

    def update_episode(self, episode: int):
        self._update_epsilon(episode)
