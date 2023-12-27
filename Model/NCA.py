"""
VNCA implementation

"""

import random

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint


class NCA(nn.Module):

    def __init__(self, update_net: nn.Module, min_steps, max_steps, 
                 p_update:float=0.5, with_alive_mask:bool=False):
        super().__init__()
        """
        接收 update_net（一个神经网络模块），用于更新细胞的状态。
        min_steps 和 max_steps 定义了更新细胞状态的最小和最大步数，这意味着在每次前向传播中，细胞的状态将至少更新 min_steps 次，最多 max_steps 次。
        p_update 是一个概率值，决定了在每一步中细胞状态更新的可能性。


        """
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.update_net = update_net
        self.p_update = p_update

        self.with_alive_mask = with_alive_mask

    def step(self, state, rand_update_mask):
        """
        这是 NCA 的核心函数，用于执行一步状态更新。
        state 是当前的细胞状态。
        rand_update_mask 是一个随机生成的遮罩，根据 p_update 的值确定是否更新特定的细胞。
        update_net 生成一个更新量，这个量乘以 rand_update_mask 后加到原始状态上，得到新的状态。
        """
        if self.with_alive_mask:
            pre_alive_mask = self.alive_mask(state)

        update = self.update_net(state)
        state = (state + update * rand_update_mask)

        if self.with_alive_mask:
            post_alive_mask = self.alive_mask(state) # post_alive_mask = self.alive_mask(state)
            state = state * (post_alive_mask * pre_alive_mask) #state * (post_alive_mask * pre_alive_mask)
            
        else:
            state = state  

        return state


    def alive_mask(self, state):
        x = F.max_pool2d(state[:, 0:1], kernel_size=(3, 3), stride=1, padding=1)
        hard = (torch.sigmoid(x - 6.0) > 0.1).to(torch.float32)
        soft = x  # t.sigmoid(x - 6.0 - t.logit(t.tensor(0.1)))
        out = hard + soft - soft.detach()

        return out


    def forward(self, state):
        """
        这是 NCA 的前向传播函数。
        它首先创建一个包含初始状态的列表 states。
        然后，它在 min_steps 和 max_steps 之间随机选择一个步数，并在这些步数内循环更新状态。
        在每一步中，它生成一个新的 rand_update_mask，并调用 step 函数更新状态。
        更新后的状态被添加到 states 列表中。
        """
        states = [state]

        for j in range(random.randint(self.min_steps, self.max_steps)):
            rand_update_mask = (torch.rand((state.shape[0], 1, state.shape[2], state.shape[3]), device=self.device) < self.p_update).to(torch.float32)
            state = checkpoint(self.step, state, rand_update_mask)
            states.append(state)

        return states