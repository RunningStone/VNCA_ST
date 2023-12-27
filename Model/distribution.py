"""
distribution related 

copy from https://github.com/rasmusbergpalm/vnca/blob/main/modules/dml.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Distribution


class DiscretizedMixtureLogitsDistribution(Distribution):
    """
    DiscretizedMixtureLogitsDistribution 类提供了一种处理离散混合逻辑斯特分布的方法。
    它可以用于像素级的图像生成或其他类似的任务，其中需要从复杂的、由多个组成部分混合的分布中采样或计算概率
    """
    def __init__(self, nr_mix, logits,sample_iters:int=100):
        super().__init__()
        """
        接收两个参数：
            nr_mix（混合成分的数量）
            logits（概率对数值）
        """
        self.logits = logits #存储了分布的对数概率。
        self.nr_mix = nr_mix #表示分布中混合成分的数量。
        self._batch_shape = logits.shape #存储了 logits 的形状，这在处理批次数据时非常有用。
        self.sample_iters = sample_iters

    def log_prob(self, value):
        """
        计算给定值的对数概率。
        它调用 discretized_mix_logistic_loss 函数来计算对数概率，使用离散混合逻辑斯特回归模型。
        输入值 value 被标准化为 [-1, 1] 的范围内(这是常见的做法，尤其是在处理图像数据时)。
        返回的对数概率增加了一个维度，以符合预期的 bchw（批次、通道、高度、宽度）格式。
        """

        return - discretized_mix_logistic_loss(value * 2 - 1, self.logits).unsqueeze(1)  # add channel dim for compatibility with loss functions expecting bchw

    def sample(self):
        """
        用于从分布中采样。主要是调用 sample_from_discretized_mix_logistic 函数，该函数根据 logits 和 nr_mix 生成样本。
        
        生成的样本被重新缩放到 [0, 1] 的范围
        """
        return (sample_from_discretized_mix_logistic(self.logits, self.nr_mix) + 1) / 2

    @property
    def mean(self):
        """
        计算分布均值的属性。
        它通过多次采样（在这里默认是100次）并计算这些样本的平均值来估计均值。
        这种方法可能不是最高效的，尤其是在大型数据集或复杂分布的情况下，但它提供了一种近似均值的实用方法
        """
        return t.stack([self.sample() for _ in range(self.sample_iters)]).mean(dim=0)


class DiscretizedMixtureLogits():

    def __init__(self, nr_mix, **kwargs):
        self.nr_mix = nr_mix

    def __call__(self, logits):
        return DiscretizedMixtureLogitsDistribution(self.nr_mix, logits)


# copied from: https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py


def log_sum_exp(x):
    """ 函数提供了一个在数值上稳定的方法来计算 log-sum-exp 操作，这是在处理概率和对数概率时常用的技巧
    
    """
    # TF ordering
    axis = len(x.size()) - 1 #确定了要在哪个轴上执行操作。对于一维数组，它将是 0；对于二维数组，它将是 1，以此类推。这是为了确保操作可以应用于多维数据
    m, _ = torch.max(x, dim=axis) #计算最大值（Max）以实现数值稳定性
    m2, _ = torch.max(x, dim=axis, keepdim=True) #同样计算最大值，但保持了原始输入的维度。这是为了使后续操作在维度上保持一致

    #原始值减去其最大值，然后对这些值应用 exp。接着，对结果求和（沿着指定轴），最后对这个总和取对数。这整个操作是 log-sum-exp 的核心。
    #最后，将步骤 2 中计算出的最大值 m 添加回去
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis)) 

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def discretized_mix_logistic_loss(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval 
    用于混合离散逻辑斯蒂分布的对数似然损失计算

    """
    # Pytorch ordering
    # 1. 将输入的维度从 bchw 转换为 bhwc
    x = x.permute(0, 2, 3, 1) # 将输入的维度从 bchw 转换为 bhwc
    l = l.permute(0, 2, 3, 1) # 将输入的维度从 bchw 转换为 bhwc
    #两个张量都被置换维度（permute），以符合 PyTorch 中的数据排列顺序
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    # 2.用于解包离散逻辑斯蒂分布的参数
    # 从 l 中提取混合逻辑斯蒂参数。nr_mix 是混合组件的数量，logit_probs 是混合权重的对数概率。
    nr_mix = int(ls[-1] / 10) #
    logit_probs = l[:, :, :, :nr_mix] 
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])  # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix] # means 是均值
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.) # 限制 log_scales 的最小值为 -7

    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) # tanh 函数用于确保系数在 [-1, 1] 范围内

    # here and below: getting the means and adjusting them based on preceding sub-pixels
    # 3. 获取均值并根据前面的子像素进行调整
    # 
    x = x.contiguous()
    x = x.unsqueeze(-1) + torch.zeros(xs + [nr_mix], device=x.device)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
          * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
          coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)

    # 4. 计算离散逻辑斯蒂分布的对数概率 CDF 和 CDF_min
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling) 
    # 0 的边缘情况的对数概率（缩放前）
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    # 255 的边缘情况的对数概率（缩放前）
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    # 在 bin 的中心的对数概率，用于极端情况（实际上没有在我们的代码中使用）
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)
    # 5. 选择正确的输出：左边缘情况、右边缘情况、正常情况、极低概率情况（实际上不会发生在我们身上）

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
            log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return - log_sum_exp(log_probs)


def discretized_mix_logistic_loss_1d(x, l):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 3)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])  # 2 for mean, scale
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).cuda(), requires_grad=False)

    # means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (
            log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return -torch.sum(log_sum_exp(log_probs))


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda: one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return Variable(one_hot)


def sample_from_discretized_mix_logistic_1d(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [1]  # [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 2])  # for mean, scale

    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda: temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    u = torch.FloatTensor(means.size())
    if l.is_cuda: u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    out = x0.unsqueeze(1)
    return out


def sample_from_discretized_mix_logistic(l, nr_mix):
    # Pytorch ordering
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda: temp = temp.cuda()
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(l[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        l[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    coeffs = torch.sum(torch.tanh(
        l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.FloatTensor(means.size())
    if l.is_cuda: u = u.cuda()
    u.uniform_(1e-5, 1. - 1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    x1 = torch.clamp(torch.clamp(
        x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.), max=1.)
    x2 = torch.clamp(torch.clamp(
        x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.), max=1.)

    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out


if __name__ == '__main__':
    import torch as t

    b, c, h, w = 5, 3, 7, 9
    x = 2 * t.rand((b, c, h, w)) - 1
    logits = t.randn((b, 10, h, w))
    dist = DiscretizedMixtureLogitsDistribution(1, logits)
    print(dist.log_prob(x).shape)
    print(dist.sample())