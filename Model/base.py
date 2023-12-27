"""

"""

import torch 
import torch.nn as nn


class CAModel(nn.Module):

    def train_batch(self) -> float:
        raise NotImplemented()

    def eval_batch(self) -> float:
        raise NotImplemented()

    def save(self, fn):
        torch.save({
            'batch_idx': self.batch_idx,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, fn)

    def load(self, fn):
        checkpoint = torch.load(fn, map_location=torch.device(self.device))
        self.batch_idx = checkpoint["batch_idx"]
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])



class Residual(nn.Module):
    """
    Add residual connection to a module.
    
    """
    def __init__(self, *args: nn.Module):
        super().__init__()
        self.delegate = nn.Sequential(*args)

    def forward(self, inputs):
        return self.delegate(inputs) + inputs
    

class SobelConv(nn.Module):
    """
    SobelConv 类，继承自 PyTorch 的 nn.Module。这个类实现了一个自定义的卷积层，使用 Sobel 滤波器来计算图像的梯度
    """
    def __init__(self, channels_in, device):
        super().__init__()
        """
        paras:
            channels_in：输入通道的数量。
            device：指定将张量分配到的设备（如 CPU 或 GPU）。

        在初始化函数中，定义了两个 Sobel 滤波器：sobel_x 和 sobel_y。
        sobel_x 用于检测水平方向（x轴）上的边缘，sobel_y 用于检测垂直方向（y轴）上的边缘。
        这些滤波器被初始化为标准的 Sobel 滤波器权重，并根据输入通道数量进行扩展。权重被除以 8。
        """
        self.channels_in = channels_in
        nn.Parameter()
        self.sobel_x = torch.tensor([[-1.0, 0.0, +1.0],
                                 [-2.0, 0.0, +2.0],
                                 [-1.0, 0.0, +1.0]], device=device).unsqueeze(0).unsqueeze(0).expand((channels_in, 1, 3, 3)) / 8.0  # (out, in, h, w)
        self.sobel_y = self.sobel_x.permute(0, 1, 3, 2)

    def forward(self, state):
        """
        paras:
            state：输入的图像或特征图。
        该函数首先使用 self.sobel_x 和 self.sobel_y 分别计算输入图像的水平和垂直梯度。
        使用 torch.conv2d 函数执行卷积，groups=self.channels_in 表示每个输入通道分别与对应的 Sobel 滤波器卷积，padding=1 用于保持输出尺寸与输入尺寸一致。
        最后，将原始输入 state 与计算得到的水平和垂直梯度在通道维度上拼接（torch.cat），生成包含原始特征及其梯度信息的输出
        """
        # Convolve sobel filters with states
        # in x, y and channel dimension.
        grad_x = torch.conv2d(state, self.sobel_x, groups=self.channels_in, padding=1)
        grad_y = torch.conv2d(state, self.sobel_y, groups=self.channels_in, padding=1)
        # Concatenate the cell’s state channels,
        # the gradients of channels in x and
        # the gradient of channels in y.
        return torch.cat([state, grad_x, grad_y], dim=1)