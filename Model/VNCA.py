import random
from typing import Sequence, Tuple

import numpy as np
import torch 
import torch.utils.data
import tqdm
from shapeguard import ShapeGuard
from torch import optim
from torch.distributions import Normal, Distribution
from torch.utils.data import DataLoader, Dataset

from VNCA_ST import logger
from VNCA_ST.Trainer.dataset import IterableWrapper
from VNCA_ST.Trainer.loss import elbo, iwae
from VNCA_ST.Model.NCA import NCA
from VNCA_ST.Model.base import CAModel

from VNCA_ST.Trainer.reporter import get_writer

# torch.autograd.set_detect_anomaly(True)

class VNCA_paras:
    h: int
    w: int
    n_channels: int
    z_size: int
    encoder: torch.nn.Module
    update_net: torch.nn.Module
    train_data: Dataset
    val_data: Dataset
    test_data: Dataset
    states_to_dist: callable
    batch_size: int
    dmg_size: int
    p_update: float
    min_steps: int
    max_steps: int

    # reporter
    writer:callable = None
    reporter_type:str = "tensorboard"
    exp_name:str = "VNCAST_original"



class VNCA(CAModel):
    def __init__(self,
                paras:VNCA_paras,
                 ):
        super(CAModel, self).__init__()
        self.h = paras.h
        self.w = paras.w
        self.n_channels = paras.n_channels
        self.state_to_dist = paras.states_to_dist
        self.z_size = paras.z_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pool = []
        self.pool_size = 1024
        self.n_damage = paras.batch_size // 4
        self.dmg_size = paras.dmg_size

        self.encoder = paras.encoder
        self.nca = NCA(paras.update_net, paras.min_steps, paras.max_steps, paras.p_update)
        self.p_z = Normal(torch.zeros(self.z_size, device=self.device), 
                          torch.ones(self.z_size, device=self.device))

        self.test_set = paras.test_data
        self.train_loader = iter(DataLoader(IterableWrapper(paras.train_data), 
                                            batch_size=paras.batch_size, pin_memory=True))
        self.val_loader = iter(DataLoader(IterableWrapper(paras.val_data), 
                                          batch_size=paras.batch_size, pin_memory=True))

        self.writer = paras.writer if paras.writer is not None \
                                    else get_writer(paras.reporter_type,
                                                    self.state_to_dist,
                                                    paras.exp_name)

        logger.info(self)
        total = sum(p.numel() for p in self.parameters())
        for n, p in self.named_parameters():
            logger.info(n, p.shape, p.numel(), "%.1f" % (p.numel() / total * 100))
        logger.info("Total: %d" % total)

        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.batch_idx = 0

    def train_batch(self):
        self.train(True)

        self.optimizer.zero_grad()
        x, y = next(self.train_loader)
        loss, z, p_x_given_z, recon_loss, kl_loss, states = self.forward(x, 1, elbo)
        loss.mean().backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0, error_if_nonfinite=True)

        self.optimizer.step()

        if self.batch_idx % 100 == 0:
            self.writer.report_train( states, loss, recon_loss, kl_loss)

        self.batch_idx += 1
        return loss.mean().item()

    def eval_batch(self):
        self.train(False)
        with torch.no_grad():
            x, y = next(self.val_loader)
            loss, z, p_x_given_z, recon_loss, kl_loss, states = self.forward(x, 1, iwae)
            self.writer.report_test(states, loss, recon_loss, kl_loss)
        return loss.mean().item()

    def test(self, n_iw_samples):
        self.train(False)
        with torch.no_grad():
            total_loss = 0.0
            for x, y in tqdm.tqdm(self.test_set):
                loss, z, p_x_given_z, recon_loss, kl_loss, states = self.forward(x, n_iw_samples, iwae)
                total_loss += loss.mean().item()

        logger.info(total_loss / len(self.test_set))

    def encode(self, x) -> Distribution:  # q(z|x)
        x.sg("Bchw")
        q = self.encoder(x).sg("BZ")
        loc = q[:, :self.z_size].sg("Bz")
        logsigma = q[:, self.z_size:].sg("Bz")
        return Normal(loc=loc, scale=torch.exp(logsigma))

    def decode(self, z: torch.Tensor) -> Tuple[Distribution, Sequence[torch.Tensor]]:  # p(x|z)
        z.sg("bzhw")
        return self.nca(z)

    def damage(self, states):
        states.sg("*zhw")
        mask = torch.ones_like(states)
        for i in range(states.shape[0]):
            h1 = random.randint(0, states.shape[2] - self.dmg_size)
            w1 = random.randint(0, states.shape[3] - self.dmg_size)
            mask[i, :, h1:h1 + self.dmg_size, w1:w1 + self.dmg_size] = 0.0
        return states * mask

    def forward(self, x, n_samples, loss_fn):
        ShapeGuard.reset()
        x.sg("Bchw")
        x = x.to(self.device)

        # Pool samples
        bs = x.shape[0]
        n_pool_samples = bs // 2
        pool_states = None
        if self.training and 0 < n_pool_samples < len(self.pool):
            # pop n_pool_samples worst in the pool
            pool_samples = self.pool[:n_pool_samples]
            self.pool = self.pool[n_pool_samples:]

            pool_x, pool_states, _ = zip(*pool_samples)
            pool_x = torch.stack(pool_x).to(self.device)
            pool_states = torch.stack(pool_states).to(self.device)
            pool_states[:self.n_damage] = self.damage(pool_states[:self.n_damage])
            x[-n_pool_samples:] = pool_x

        q_z_given_x = self.encode(x).sg("Bz")
        z = q_z_given_x.rsample((n_samples,)).permute((1, 0, 2)).sg("Bnz")

        seeds = (z.reshape((-1, self.z_size))  # stuff samples into batch dimension
                 .unsqueeze(2)
                 .unsqueeze(3)
                 .expand(-1, -1, self.h, self.w).sg("bzhw"))

        if pool_states is not None:
            seeds = seeds.clone()
            seeds[-n_pool_samples:] = pool_states  # yes this is wrong and will mess up the gradientorch.

        states = self.decode(seeds)
        p_x_given_z = self.state_to_dist(states[-1])

        loss, recon_loss, kl_loss = loss_fn(x, p_x_given_z, q_z_given_x, self.p_z, z)

        if self.training:
            # Add states to pool
            def split(tensor: torch.Tensor):
                return [x for x in tensor]

            self.pool += list(zip(split(x.cpu()), split(states[-1].detach().cpu()), loss.tolist()))
            # Retain the worst
            # self.pool = sorted(self.pool, key=lambda x: x[-1], reverse=True)
            random.shuffle(self.pool)
            self.pool = self.pool[:self.pool_size]

        return loss, z, p_x_given_z, recon_loss, kl_loss, states
