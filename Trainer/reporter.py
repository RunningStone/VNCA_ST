"""
定义了Reporter类，用于记录训练过程中的信息
"""
import os
import torch
import random
import numpy as np
from shapeguard import ShapeGuard
from torch.distributions import Normal, Distribution


def get_writer(reporter_type:str,state_to_dist,exp_name):
    if reporter_type == "tensorboard":
        return TensorBoardReporter(state_to_dist,exp_name)
    else:
        raise ValueError(f"Unknown reporter type{reporter_type}")

########################################################################################################################
# base reporter
########################################################################################################################
class BaseReporter:
    def __init__(self,state_to_dist,exp_name) -> None:
        self.state_to_dist = state_to_dist
        self.exp_name = exp_name

    def set_batch_params(self, batch_idx, batch_size, device, h, w, n_channels,pool):
        self.batch_idx = batch_idx
        self.batch_size = batch_size
        self.device = device
        self.h = h
        self.w = w
        self.n_channels = n_channels
        self.pool = pool

    def set_callable(self,p_z:callable, encode:callable, decode:callable, damage:callable):
        self.p_z = p_z
        self.encode = encode
        self.decode = decode
        self.damage = damage

    def to_rgb(self, state):
        dist: Distribution = self.state_to_dist(state)
        return dist.sample(), dist.mean
    
    def report_train(self, recon_states, loss, recon_loss, kl_loss):
        self._report(self.train_writer, recon_states, loss, recon_loss, kl_loss)

    def report_test(self, recon_states, loss, recon_loss, kl_loss):
        self._report(self.test_writer, recon_states, loss, recon_loss, kl_loss)

    def _report(self,writer, recon_states, loss, recon_loss, kl_loss,):
        raise NotImplementedError
########################################################################################################################
# tensorboard reporter
########################################################################################################################
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



class TensorBoardReporter(BaseReporter):
    def __init__(self,state_to_dist,exp_name) -> None:
        super().__init__(state_to_dist,exp_name)

        # init tensorboard params
        self.revision = os.environ.get("REVISION") or "%s" % datetime.now()
        self.message = os.environ.get('MESSAGE')
        self.tensorboard_dir = os.environ.get('TENSORBOARD_DIR') or "/tmp/tensorboard"
        self.flush_secs = 10
        self.train_writer, self.test_writer = self.get_writers("vnca")

    def get_writers(self):
        train_writer = SummaryWriter(self.tensorboard_dir + '/%s/tensorboard/%s/train/%s' % (self.exp_name, self.revision, self.message), flush_secs=self.flush_secs)
        test_writer = SummaryWriter(self.tensorboard_dir + '/%s/tensorboard/%s/test/%s' % (self.exp_name, self.revision, self.message), flush_secs=self.flush_secs)
        return train_writer, test_writer

    def _report(self,writer: SummaryWriter, recon_states, loss, recon_loss, kl_loss,):
        writer.add_scalar('loss', loss.mean().item(), self.batch_idx)
        writer.add_scalar('bpd', loss.mean().item() / (np.log(2) * self.n_channels * self.h * self.w), self.batch_idx)
        writer.add_scalar('pool_size', len(self.pool), self.batch_idx)

        if recon_loss is not None:
            writer.add_scalar('recon_loss', recon_loss.mean().item(), self.batch_idx)
        if kl_loss is not None:
            writer.add_scalar('kl_loss', kl_loss.mean().item(), self.batch_idx)

        ShapeGuard.reset()
        with torch.no_grad():
            # samples
            samples = self.p_z.sample((8,)).view(8, -1, 1, 1).expand(8, -1, self.h, self.w).to(self.device)
            states = self.decode(samples)
            samples, samples_means = self.to_rgb(states[-1])
            writer.add_images("samples/samples", samples, self.batch_idx)
            writer.add_images("samples/means", samples_means, self.batch_idx)

            def plot_growth(states, tag):
                growth_samples = []
                growth_means = []
                for state in states:
                    growth_sample, growth_mean = self.to_rgb(state[0:1])
                    growth_samples.append(growth_sample)
                    growth_means.append(growth_mean)

                growth_samples = torch.cat(growth_samples, dim=0).cpu().detach().numpy()  # (n_states, 3, h, w)
                growth_means = torch.cat(growth_means, dim=0).cpu().detach().numpy()  # (n_states, 3, h, w)
                writer.add_images(tag + "/samples", growth_samples, self.batch_idx)
                writer.add_images(tag + "/means", growth_means, self.batch_idx)

            plot_growth(states, "growth")

            # Damage
            state = states[-1]
            _, original_means = self.to_rgb(state)
            writer.add_images("dmg/1-pre", original_means, self.batch_idx)
            dmg = self.damage(state)
            _, dmg_means = self.to_rgb(dmg)
            writer.add_images("dmg/2-dmg", dmg_means, self.batch_idx)
            recovered = self.nca(dmg)
            _, recovered_means = self.to_rgb(recovered[-1])
            writer.add_images("dmg/3-post", recovered_means, self.batch_idx)

            plot_growth(recovered, "recovery")

            # Reconstructions
            recons_samples, recons_means = self.to_rgb(recon_states[-1].detach())
            writer.add_images("recons/samples", recons_samples, self.batch_idx)
            writer.add_images("recons/means", recons_means, self.batch_idx)

            # Pool
            if len(self.pool) > 0:
                pool_xs, pool_states, pool_losses = zip(*random.sample(self.pool, min(len(self.pool), 64)))
                pool_states = torch.stack(pool_states)  # 64, z, h, w
                pool_samples, pool_means = self.to_rgb(pool_states)
                writer.add_images("pool/samples", pool_samples, self.batch_idx)
                writer.add_images("pool/means", pool_means, self.batch_idx)

        writer.flush()


########################################################################################################################
# wandb reporter
########################################################################################################################
import wandb
import wandb
import numpy as np
import torch
import random
import os
from datetime import datetime

class WandbReporter(BaseReporter):
    def __init__(self, state_to_dist, exp_name):
        super().__init__(state_to_dist, exp_name)

        # 初始化 wandb
        #exp_name 按照 "_" 分割为 project 和 name
        self.project, self.name = exp_name.split("_")

        wandb.init(project=self.project, name=self.name)

    def _report(self, recon_states, loss, recon_loss, kl_loss,phase="train"):
        # 计算并记录标量指标
        metrics = {
            f'{phase}/loss': loss.mean().item(),
            f'{phase}/bpd': loss.mean().item() / (np.log(2) * self.n_channels * self.h * self.w),
            f'{phase}/pool_size': len(self.pool)
        }

        if recon_loss is not None:
            metrics[f'{phase}/recon_loss'] = recon_loss.mean().item()
        if kl_loss is not None:
            metrics[f'{phase}/kl_loss'] = kl_loss.mean().item()

        wandb.log(metrics, step=self.batch_idx)

        with torch.no_grad():
            # 处理并记录样本图像
            self._log_sample_images(recon_states[-1], "samples", phase)

            # 处理并记录生长图像
            self._log_growth_images(recon_states, "growth", phase)

            # 处理并记录损伤和恢复过程图像
            self._log_damage_and_recovery_images(recon_states[-1], "damage_recovery", phase)

            # 处理并记录重构图像
            self._log_reconstruction_images(recon_states[-1], "reconstructions", phase)

            # 处理并记录池中的图像
            self._log_pool_images("pool", phase)

    def _log_sample_images(self, states, tag, phase):
        samples, samples_means = self.to_rgb(states)
        full_tag = f"{phase}_{+tag}"
        wandb.log({f"{full_tag}/samples": [wandb.Image(img) for img in samples]}, step=self.batch_idx)
        wandb.log({f"{full_tag}/means": [wandb.Image(img) for img in samples_means]}, step=self.batch_idx)

    def _log_growth_images(self, states, tag,phase):
        # 假设实现 - 需要根据具体情况调整
        def plot_growth(states):
            growth_samples = []
            growth_means = []
            for state in states:
                growth_sample, growth_mean = self.to_rgb(state[0:1])
                growth_samples.append(growth_sample)
                growth_means.append(growth_mean)
            return growth_samples, growth_means

        growth_samples, growth_means = plot_growth(states)
        full_tag = f"{phase}_{+tag}"
        wandb.log({f"{full_tag}/samples": [wandb.Image(img) for img in growth_samples]}, step=self.batch_idx)
        wandb.log({f"{full_tag}/means": [wandb.Image(img) for img in growth_means]}, step=self.batch_idx)

    def _log_damage_and_recovery_images(self, state, tag,phase):

        # 记录损伤和恢复过程中的图像
        _, original_means = self.to_rgb(state)
        dmg = self.damage(state)
        _, dmg_means = self.to_rgb(dmg)
        recovered = self.nca(dmg)
        _, recovered_means = self.to_rgb(recovered[-1])
        full_tag = f"{phase}_{+tag}"
        wandb.log({f"{full_tag}/1-pre": [wandb.Image(img) for img in original_means]}, step=self.batch_idx)
        wandb.log({f"{full_tag}/2-dmg": [wandb.Image(img) for img in dmg_means]}, step=self.batch_idx)
        wandb.log({f"{full_tag}/3-post": [wandb.Image(img) for img in recovered_means]}, step=self.batch_idx)

    def _log_reconstruction_images(self, states, tag, phase):
        # 假设实现 - 需要根据具体情况调整
        recons, recons_means = self.to_rgb(states)
        full_tag = f"{phase}_{+tag}"
        wandb.log({f"{full_tag}/samples": [wandb.Image(img) for img in recons]}, step=self.batch_idx)
        wandb.log({f"{full_tag}/means": [wandb.Image(img) for img in recons_means]}, step=self.batch_idx)

    def _log_pool_images(self, tag, phase):
        # 假设实现 - 需要根据具体情况调整
        full_tag = f"{phase}_{+tag}"
        pool_imgs = [self.to_rgb(state)[0] for state in self.pool]
        wandb.log({f"{full_tag}/images": [wandb.Image(img) for img in pool_imgs]}, step=self.batch_idx)
