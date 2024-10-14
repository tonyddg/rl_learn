from collections import deque
from dataclasses import dataclass
import random

import numpy as np

import torch

from typing import SupportsFloat

@dataclass
class Transition:
    '''
    Transition  
    应保证元素的第一维为批次
    '''
    state: torch.Tensor 
    action: torch.Tensor
    next_state: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor

    def __eq__(self, obj):
        if not isinstance(obj, Transition):
            raise Exception("Can not compare with other type")
        for key in self.__dict__.keys():
            if not (self.__dict__[key] == obj.__dict__[key]).all():
                return False
        return True

def random_transition(state_dim: int, action_dim: int, batch_size: int = 1):
    '''
    生成随机 Transition
    '''
    return Transition(
        torch.rand((batch_size, state_dim)),
        torch.rand((batch_size, action_dim)),
        torch.rand((batch_size, state_dim)),
        torch.rand((batch_size, 1)),
        torch.rand((batch_size, 1))
    )

def make_transition_from_numpy(state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: SupportsFloat | np.ndarray, done: SupportsFloat | np.ndarray):
    '''
    通过 Numpy 数组创建 Transition, 要求 state, action, next_state, reward (多 batch) 都必须是转移所有权的 Numpy 数组  
    
    根据 reward 判断是否传入多 Batch
    * reward 为单个数字 (float 或 numpy) 时为非 batch, 将自动升维
    * reward 为多元素 numpy 数组或具有维度的单元素时, 不会自动升维
    '''
    if not isinstance(reward, np.ndarray) or reward.shape == () :
        return Transition(
            torch.from_numpy(state).view(1, -1),
            torch.from_numpy(action).view(1, -1),
            torch.from_numpy(next_state).view(1, -1),
            torch.tensor(reward).view(1, -1),
            torch.tensor(done).view(1, -1)
        )
    else:
        return Transition(
            torch.from_numpy(state),
            torch.from_numpy(action),
            torch.from_numpy(next_state),
            torch.from_numpy(reward),
            torch.tensor(done).view(1, -1)
        )

def pack_transition_batch(pack: list[Transition]) -> Transition:
    '''
    将多条单个的 Transition 合并为一个批次
    '''
    batch_size = len(pack)
    
    pack_state = torch.zeros((batch_size,) + pack[0].state.shape[1:])
    pack_action = torch.zeros((batch_size,) + pack[0].action.shape[1:])
    pack_next_state = torch.zeros((batch_size,) + pack[0].next_state.shape[1:])
    pack_reward = torch.zeros((batch_size,) + pack[0].reward.shape[1:])
    pack_done = torch.zeros((batch_size,) + pack[0].reward.shape[1:])

    for i, iter in enumerate(pack):
        pack_state[i] = iter.state[0]
        pack_action[i] = iter.action[0]
        pack_next_state[i] = iter.next_state[0]
        pack_reward[i] = iter.reward[0]
        pack_done[i] = iter.done[0]

    return Transition(pack_state, pack_action, pack_next_state, pack_reward, pack_done)

class ReplyQueue:
    def __init__(self, capacity: int) -> None:
        '''
        经验队列
        '''
        # 当 deque 已满时插入元素, 另一端元素将自动弹出
        self.buffer: deque[Transition] = deque(maxlen = capacity)

    def size(self):
        return len(self.buffer)

    def append(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        '''
        均匀采样, 当经验回放中的 Transition 不足时将抛出异常
        '''
        if batch_size > self.size():
            raise Exception("Not enough reply")
        return pack_transition_batch(random.sample(self.buffer, batch_size))
