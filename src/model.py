from typing import Dict, Tuple, Union
from omegaconf import DictConfig

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from imageio import imwrite

import pytorch_lightning as pl
import torch

from encoder import Encoder
from decoder import Decoder


def reparameterize_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    noise = torch.randn_like(std)
    latent = mu + std * noise
    return latent


class Model(pl.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.cfg = cfg.model
        self.max_steps = cfg.trainer.max_steps
        self.val_check_interval = cfg.trainer.val_check_interval
        self.log_num_images = cfg.trainer.log_num_images
        self.gradient_clip_norm = cfg.trainer.gradient_clip_norm
        self.coef_normal = 0.25 * 0.5 / pow(self.cfg.loss.normal_scale, 2)
        self.coef_px = cfg.dataset.image_shape[0] * cfg.dataset.image_shape[1]
        self.num_slots_train = cfg.dataset.infer_slots.train
        self.num_slots_val = cfg.dataset.infer_slots.val
        self.num_slots_predict_0 = cfg.dataset.infer_slots.test
        self.num_slots_predict_1 = cfg.dataset.infer_slots.general
        self.num_views_min = cfg.dataset.num_views_min
        self.num_views_max = cfg.dataset.num_views_max
        self.max_shadow_ratio = self.cfg.max_shadow_ratio
        self.enc = Encoder(self.cfg, *cfg.dataset.image_shape)
        self.dec = Decoder(self.cfg, *cfg.dataset.image_shape)

    def compute_loss_coef(self, step: Union[int, None] = None) -> Dict[str, torch.Tensor]:
        if step is None:
            step = self.max_steps
        loss_coef = {'nll': 1, 'kld_view': 1, 'kld_bck': 1, 'kld_obj': 1, 'kld_pres': 1}
        for key, val in self.cfg.loss.coef.items():
            step_list = [1] + val['step'] + [self.max_steps]
            assert len(step_list) == len(val['value'])
            assert len(step_list) == len(val['linear']) + 1
            assert step_list == sorted(step_list)
            for idx in range(len(step_list) - 1):
                if step <= step_list[idx + 1]:
                    ratio = (step - step_list[idx]) / (step_list[idx + 1] - step_list[idx])
                    val_1 = val['value'][idx]
                    val_2 = val['value'][idx + 1]
                    if val['linear'][idx]:
                        loss_coef[key] = (1 - ratio) * val_1 + ratio * val_2
                    else:
                        loss_coef[key] = math.exp((1 - ratio) * math.log(val_1) + ratio * math.log(val_2))
                    assert math.isfinite(loss_coef[key])
                    break
            else:
                raise ValueError
        for name in ['kld', 'reg']:
            if f'sched_{name}' in loss_coef:
                coef = loss_coef[f'sched_{name}']
                for key in loss_coef:
                    if key.split('_')[0] == name:
                        loss_coef[key] *= coef
        loss_coef = {key: torch.tensor(val) for key, val in loss_coef.items()}
        return loss_coef

    def compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        loss_coef: Dict[str, torch.Tensor],
        eps: float = 1.0e-5,
    ) -> Dict[str, torch.Tensor]:
        def compute_kld_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            kld = 0.5 * (mu.square() + logvar.exp() - logvar - 1).flatten(start_dim=1).sum(1)
            return kld

        def compute_kld_pres(
            tau1: torch.Tensor,
            tau2: torch.Tensor,
            zeta: torch.Tensor,
            logits_zeta: torch.Tensor,
        ) -> torch.Tensor:
            tau1 = tau1.squeeze(-1)                # [B, S]
            tau2 = tau2.squeeze(-1)                # [B, S]
            zeta = zeta.squeeze(-1)                # [B, S]
            logits_zeta = logits_zeta.squeeze(-1)  # [B, S]
            coef_alpha = self.cfg.loss.pres_alpha / logits_zeta.shape[1]
            psi1 = torch.digamma(tau1)
            psi2 = torch.digamma(tau2)
            psi12 = torch.digamma(tau1 + tau2)
            kld_1 = torch.lgamma(tau1 + tau2) - torch.lgamma(tau1) - torch.lgamma(tau2) - math.log(coef_alpha)
            kld_2 = (tau1 - coef_alpha) * psi1 + (tau2 - 1) * psi2 - (tau1 + tau2 - coef_alpha - 1) * psi12
            log_zeta = torch.nn.functional.logsigmoid(logits_zeta)
            log1m_zeta = log_zeta - logits_zeta
            kld_3 = zeta * (log_zeta - psi1) + (1 - zeta) * (log1m_zeta - psi2) + psi12
            kld = (kld_1 + kld_2 + kld_3).flatten(start_dim=1).sum(1)
            return kld

        image = batch['image']
        apc_all = torch.cat([outputs['apc'], outputs['bck'][:, :, None]], dim=2)
        mask_all = torch.cat([outputs['mask_obj'], outputs['mask_bck'][:, :, None]], dim=2)
        sq_diff = (apc_all - image[:, :, None]).square()
        raw_pixel_ll = -self.coef_normal * sq_diff.sum(-1, keepdim=True)
        log_mask_all = torch.log(mask_all * (1 - 2 * eps) + eps)
        masked_pixel_ll = log_mask_all + raw_pixel_ll
        loss_nll_sm = -torch.logsumexp(masked_pixel_ll, dim=2).flatten(start_dim=1).sum(1)
        loss_nll_ws = self.coef_normal * (outputs['recon_aux'] - image).square().flatten(start_dim=1).sum(1)
        loss_nll_ws_imp = self.coef_normal * (outputs['recon_aux_imp'] - image).square().flatten(start_dim=1).sum(1)
        ratio_imp_sdw = loss_coef['ratio_imp_sdw']
        loss_nll_ws = ratio_imp_sdw * loss_nll_ws_imp + (1 - ratio_imp_sdw) * loss_nll_ws
        ratio_mixture = loss_coef['ratio_mixture']
        loss_nll = ratio_mixture * loss_nll_sm + (1 - ratio_mixture) * loss_nll_ws
        sdw_shp_hard = torch.gt(outputs['sdw_shp'], 0.1).to(outputs['sdw_shp'].dtype)
        sdw_ratio = (outputs['shp_imp'] * sdw_shp_hard).flatten(start_dim=-3).sum(-1) / \
            (outputs['shp_imp'].flatten(start_dim=-3).sum(-1) + eps)
        reg_sdw_ratio = torch.gt(sdw_ratio, self.max_shadow_ratio)[..., None, None, None] * \
            torch.abs(outputs['sdw_logits_shp'] + 3)
        reg_sdw_ratio = (1 - ratio_imp_sdw) * reg_sdw_ratio.flatten(start_dim=1).sum(1)
        losses = {
            'nll': loss_nll,
            'kld_view': compute_kld_normal(outputs['view_mu'], outputs['view_logvar']),
            'kld_bck': compute_kld_normal(outputs['bck_mu'], outputs['bck_logvar']),
            'kld_obj': compute_kld_normal(outputs['obj_mu'], outputs['obj_logvar']),
            'kld_pres': compute_kld_pres(outputs['tau1'], outputs['tau2'], outputs['zeta'], outputs['logits_zeta']),
            'reg_bck': self.coef_normal * (outputs['bck'] - batch['image']).square().flatten(start_dim=1).sum(1),
            'reg_pres': self.coef_px * (outputs['zeta'].detach() * outputs['logits_zeta']).flatten(start_dim=1).sum(1),
            'reg_shp': torch.abs(outputs['logits_shp_imp'] + 3).flatten(start_dim=1).sum(1),
            'reg_sdw': torch.abs(outputs['sdw_logits_shp'] + 3).flatten(start_dim=1).sum(1),
            'reg_sdw_ratio': reg_sdw_ratio,
        }
        losses = {key: val.mean() for key, val in losses.items()}
        losses['opt'] = torch.stack([loss_coef[key] * val for key, val in losses.items()]).sum()
        return losses

    @torch.no_grad()
    def compute_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        image = batch['image']
        segment = batch['segment']
        overlap = batch['overlap']
        segment_obj = segment[:, :, :-1].contiguous()

        # ARI
        segment_all_sel = segment if self.cfg.seg_overlap else segment * (1 - overlap)
        segment_obj_sel = segment_obj if self.cfg.seg_overlap else segment_obj * (1 - overlap)
        mask_all = torch.cat([outputs['mask_obj'], outputs['mask_bck'][:, :, None]], dim=2)
        mask_obj = outputs['mask_obj']

        def compute_ari_values(
            mask_true: torch.Tensor,
            mask_pred_soft: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            mask_true_s = mask_true.flatten(end_dim=1)
            mask_true_m = mask_true.transpose(1, 2).contiguous()
            mask_pred_index = torch.argmax(mask_pred_soft, dim=2, keepdim=True)
            mask_pred = torch.zeros_like(mask_pred_soft).scatter_(2, mask_pred_index, 1)
            mask_pred_s = mask_pred.flatten(end_dim=1)
            mask_pred_m = mask_pred.transpose(1, 2).contiguous()

            def compute_ari(mask_true: torch.Tensor, mask_pred: torch.Tensor) -> torch.Tensor:
                mask_true = mask_true.flatten(start_dim=2)
                mask_pred = mask_pred.flatten(start_dim=2)
                num_pixels = mask_true.flatten(start_dim=1).sum(1, keepdims=True)
                mat = torch.einsum('bin,bjn->bij', mask_true, mask_pred)
                sum_row = mat.sum(1)
                sum_col = mat.sum(2)

                def comb2(x: torch.Tensor) -> torch.Tensor:
                    x = x * (x - 1)
                    x = x.flatten(start_dim=1).sum(1)
                    return x

                comb_mat = comb2(mat)
                comb_row = comb2(sum_row)
                comb_col = comb2(sum_col)
                comb_num = comb2(num_pixels)
                comb_prod = (comb_row * comb_col) / comb_num
                comb_mean = 0.5 * (comb_row + comb_col)
                diff = comb_mean - comb_prod
                score = (comb_mat - comb_prod) / diff
                invalid = torch.logical_or(torch.eq(comb_num, 0), torch.eq(diff, 0))
                score.masked_fill_(invalid, 1)
                return score

            ari_all_s = compute_ari(mask_true_s, mask_pred_s).view(mask_true.shape[:2]).mean(1)
            ari_all_m = compute_ari(mask_true_m, mask_pred_m)
            return ari_all_s, ari_all_m

        ari_all_s, ari_all_m = compute_ari_values(segment_all_sel, mask_all)
        ari_obj_s, ari_obj_m = compute_ari_values(segment_obj_sel, mask_obj)

        # MSE
        mse = 0.25 * (outputs['recon'] - image).square().flatten(start_dim=1).mean(1)

        # Count
        count_true = segment_obj.transpose(1, 2).flatten(start_dim=2).max(-1).values.sum(-1)
        count_pred = outputs['pres'].sum(1)
        count_acc = torch.eq(count_true, count_pred).to(dtype=count_true.dtype)

        # Outputs
        metrics = {
            'ari_all_s': ari_all_s, 'ari_all_m': ari_all_m, 'ari_obj_s': ari_obj_s, 'ari_obj_m': ari_obj_m,
            'mse': mse, 'count': count_acc,
        }
        metrics = {key: val.mean() for key, val in metrics.items()}
        return metrics

    @torch.no_grad()
    @pl.utilities.rank_zero.rank_zero_only
    def log_image(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        split: str,
        global_step: int,
        dpi: int = 100,
    ) -> None:
        folder = f'image_logs_{split}'
        if not os.path.exists(folder):
            os.mkdir(folder)
        batch_size, num_views, num_slots, image_ht, image_wd = outputs['apc'].shape[:5]
        num_images = min(self.log_num_images, batch_size)
        data_sel = {
            **{key: batch[key][:num_images].data for key in ['image']},
            **{key: outputs[key][:num_images].data for key in [
                'recon', 'recon_aux', 'recon_aux_imp', 'mask_bck', 'bck', 'bck_imp', 'trs', 'pres',
                'apc', 'apc_aux', 'shp', 'shp_imp', 'sdw_apc', 'sdw_shp',
            ]},
        }
        for key in [
            'image', 'recon', 'recon_aux', 'recon_aux_imp', 'bck', 'bck_imp', 'apc', 'apc_aux', 'sdw_apc',
        ]:
            data_sel[key] = torch.clamp((data_sel[key] + 1) * 0.5, 0, 1)
        trs = data_sel['trs']
        coef_trs = torch.tensor([image_ht, image_wd], dtype=trs.dtype, device=trs.device)[None, None, None]
        data_sel['trs'] = (trs + 1) * 0.5 * coef_trs
        white = torch.ones([1, 1, 1, 1, 1, 3], dtype=trs.dtype, device=trs.device)
        green = torch.tensor([0, 1, 0], dtype=trs.dtype, device=trs.device)[None, None, None, None, None]
        data_sel['shp'] = data_sel['shp'] * white + (1 - data_sel['shp']) * data_sel['shp_imp'] * green
        data_sel['sdw_apc'] = data_sel['sdw_shp'] * data_sel['sdw_apc'] + \
            (1 - data_sel['sdw_shp']) * data_sel['bck_imp'][:, :, None]
        data_sel = {key: val.to(torch.float32).cpu().numpy() for key, val in data_sel.items()}

        def get_overview(idx_fig):
            rows, cols = 4 * num_views, num_slots + 2
            fig, axes = plt.subplots(rows, cols, figsize=(cols, (rows + 0.5 / num_views) * image_ht / image_wd), dpi=dpi)

            def plot_image(ax, image, xlabel=None, ylabel=None, color=None):
                plot = ax.imshow(image, interpolation='bilinear')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(xlabel, color='k' if color is None else color, fontfamily='monospace') if xlabel else None
                ax.set_ylabel(ylabel, color='k' if color is None else color, fontfamily='monospace') if ylabel else None
                ax.xaxis.set_label_position('top')
                return plot

            def convert_image(image):
                if image.shape[-1] == 1:
                    image = np.repeat(image, 3, axis=-1)
                return image

            for idx_view in range(num_views):
                xlabel = 'scene' if idx_view == 0 else None
                plot_image(axes[idx_view * 4, 0], convert_image(data_sel['image'][idx_fig, idx_view]), xlabel=xlabel)
                plot_image(axes[idx_view * 4 + 1, 0], convert_image(data_sel['recon'][idx_fig, idx_view]))
                plot_image(axes[idx_view * 4 + 2, 0], convert_image(data_sel['recon_aux'][idx_fig, idx_view]))
                plot_image(axes[idx_view * 4 + 3, 0], convert_image(data_sel['recon_aux_imp'][idx_fig, idx_view]))
                for idx_slot in range(num_slots):
                    xlabel = 'obj_{}'.format(idx_slot) if idx_view == 0 else None
                    color = [1.0, 0.5, 0.0] if data_sel['pres'][idx_fig, idx_slot] >= 0.5 else [0.0, 0.5, 1.0]
                    plot_image(
                        axes[idx_view * 4, idx_slot + 1],
                        convert_image(data_sel['apc'][idx_fig, idx_view, idx_slot]),
                        xlabel=xlabel,
                        color=color,
                    )
                    plot_image(
                        axes[idx_view * 4 + 1, idx_slot + 1],
                        convert_image(data_sel['apc_aux'][idx_fig, idx_view, idx_slot]),
                    )
                    plot_image(
                        axes[idx_view * 4 + 2, idx_slot + 1],
                        convert_image(data_sel['shp'][idx_fig, idx_view, idx_slot]),
                    )
                    plot_image(
                        axes[idx_view * 4 + 3, idx_slot + 1],
                        convert_image(data_sel['sdw_apc'][idx_fig, idx_view, idx_slot]),
                    )
                    for offset in [2]:
                        axes[idx_view * 4 + offset, idx_slot + 1].add_patch(
                            plt.Circle((
                                data_sel['trs'][idx_fig, idx_view, idx_slot][1],
                                data_sel['trs'][idx_fig, idx_view, idx_slot][0],
                            ), image_wd / 64, color='r')
                        )
                xlabel = 'bck' if idx_view == 0 else None
                plot_image(axes[idx_view * 4, -1], convert_image(data_sel['bck_imp'][idx_fig, idx_view]), xlabel=xlabel)
                plot_image(axes[idx_view * 4 + 1, -1], convert_image(data_sel['bck'][idx_fig, idx_view]))
                plot_image(axes[idx_view * 4 + 2, -1], convert_image(data_sel['mask_bck'][idx_fig, idx_view]))
                for offset in range(3, 4):
                    axes[idx_view * 4 + offset, -1].set_visible(False)
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            out_wd, out_ht = fig.canvas.get_width_height()
            out = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(out_ht, out_wd, -1)
            plt.close(fig)
            return out

        overview_list = [get_overview(idx) for idx in range(num_images)]
        overview = np.concatenate(overview_list, axis=0)
        imwrite(os.path.join(folder, f'step_{global_step}.png'), overview)
        return

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        temp_pres: torch.Tensor,
        temp_shp: torch.Tensor,
        noise_scale: torch.Tensor,
        noise_min: torch.Tensor,
        noise_max: torch.Tensor,
        ratio_stick_breaking: torch.Tensor,
        num_slots: int,
    ) -> Dict[str, torch.Tensor]:
        outputs_enc = self.enc(batch['image'], num_slots)
        view_latent = reparameterize_normal(outputs_enc['view_mu'], outputs_enc['view_logvar'])
        bck_latent = reparameterize_normal(outputs_enc['bck_mu'], outputs_enc['bck_logvar'])
        obj_latent = reparameterize_normal(outputs_enc['obj_mu'], outputs_enc['obj_logvar'])
        logits_pres = torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli(
            temperature=temp_pres, logits=outputs_enc['logits_zeta']).rsample()
        pres = torch.sigmoid(logits_pres)
        outputs_dec = self.dec(
            view_latent, bck_latent, obj_latent, pres, logits_pres, temp_shp, noise_scale, noise_min, noise_max,
            ratio_stick_breaking)
        outputs = {
            **outputs_enc,
            **outputs_dec,
            'pres': torch.ge(pres.squeeze(-1), 0.5).to(pres.dtype),
            'logits_pres': logits_pres,
            'view_latent': view_latent,
            'bck_latent': bck_latent,
            'obj_latent': obj_latent,
        }
        return outputs

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> Dict[str, torch.Tensor]:
        num_views_sel = torch.randint(self.num_views_min, self.num_views_max + 1, size=[]).item()
        batch = {key: val[:, :num_views_sel] for key, val in batch.items()}
        global_step = self.global_step + 1
        loss_coef = self.compute_loss_coef(global_step)
        outputs = self(batch, loss_coef['temp_pres'], loss_coef['temp_shp'], loss_coef['noise_scale'],
                       loss_coef['noise_min'], loss_coef['noise_max'], loss_coef['ratio_stick_breaking'],
                       self.num_slots_train)
        losses = self.compute_losses(batch, outputs, loss_coef)
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        self.manual_backward(losses['opt'])
        if 'ratio_dec' in loss_coef:
            with torch.no_grad():
                for param in self.enc.net_img.parameters():
                    param.grad *= loss_coef['ratio_dec']
                for param in self.enc.net_feat.parameters():
                    param.grad *= loss_coef['ratio_dec']
                for param in self.dec.parameters():
                    param.grad *= loss_coef['ratio_dec']
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        outputs.update({f'loss_{key}': val for key, val in losses.items()})
        if global_step % self.val_check_interval == 0:
            metrics = self.compute_metrics(batch, outputs)
            outputs.update({f'metric_{key}': val for key, val in metrics.items()})
        return outputs

    def on_train_batch_end(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        global_step = self.global_step + 1
        self.log_dict({
            'train/' + key: val
            for key, val in outputs.items() if key.startswith('loss_') or key.startswith('metric_')
        }, rank_zero_only=True)
        if global_step % self.val_check_interval == 0:
            self.log_image(batch, outputs, split='train', global_step=global_step)
        return

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        loss_coef = self.compute_loss_coef()
        outputs = self(batch, loss_coef['temp_pres'], loss_coef['temp_shp'], loss_coef['noise_scale'],
                       loss_coef['noise_min'], loss_coef['noise_max'], loss_coef['ratio_stick_breaking'],
                       self.num_slots_val)
        losses = self.compute_losses(batch, outputs, loss_coef)
        metrics = self.compute_metrics(batch, outputs)
        outputs.update({f'loss_{key}': val for key, val in losses.items()})
        outputs.update({f'metric_{key}': val for key, val in metrics.items()})
        return outputs

    def on_validation_batch_end(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.log_dict({
            'val/' + key: val
            for key, val in outputs.items() if key.startswith('loss_') or key.startswith('metric_')
        }, sync_dist=True)
        if batch_idx == 0:
            self.log_image(batch, outputs, split='val', global_step=self.global_step)
        return

    def predict_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        num_slots = getattr(self, f'num_slots_predict_{dataloader_idx}')
        loss_coef = self.compute_loss_coef()
        outputs = self(batch, loss_coef['temp_pres'], loss_coef['temp_shp'], noise_scale=0, noise_min=0, noise_max=0,
                       ratio_stick_breaking=1, num_slots=num_slots)
        apc_all = torch.cat([outputs['apc'], outputs['bck_imp'][:, :, None]], dim=2)
        ones = torch.ones(*outputs['shp'].shape[:2], 1, *outputs['shp'].shape[3:], device=outputs['shp'].device)
        shp_all = torch.cat([outputs['shp'], ones], dim=2)
        shp_imp_all = torch.cat([outputs['shp_imp'], ones], dim=2)
        pres = outputs['pres']
        pres_all = torch.cat([pres, torch.ones([pres.shape[0], 1], device=pres.device)], dim=1)
        outputs_extra = {
            'recon': torch.clamp((outputs['recon'] + 1) * 0.5, 0, 1),
            'bck_sdw': torch.clamp((outputs['bck'] + 1) * 0.5, 0, 1),
            'apc': torch.clamp((apc_all + 1) * 0.5, 0, 1),
            'shp': shp_all,
            'shp_imp': shp_imp_all,
            'pres': pres_all,
            'sdw_apc': torch.clamp((outputs['sdw_apc'] + 1) * 0.5, 0, 1),
            'sdw_shp': outputs['sdw_shp'],
            'mask': torch.cat([outputs['mask_obj'], outputs['mask_bck'][:, :, None]], dim=2),
        }
        outputs_extra = {key: (val * 255).to(torch.uint8) for key, val in outputs_extra.items()}
        outputs = {
            key: val for key, val in outputs.items()
                if key in ['logits_pres', 'view_latent', 'bck_latent', 'obj_latent', 'log_ord', 'trs']
        }
        outputs.update(outputs_extra)
        return outputs

    def configure_optimizers(self):
        def lr_lambda(x):
            decay_rate = self.cfg.scheduler.lr_decay
            decay_steps = self.cfg.scheduler.decay_steps
            warmup_steps = self.cfg.scheduler.warmup_steps
            decay_ratio = 0 if decay_steps == 0 else x / decay_steps
            decay_coef = pow(decay_rate, decay_ratio)
            warmup_ratio = 1 if warmup_steps == 0 else x / warmup_steps
            warmup_coef = min(warmup_ratio, 1)
            coef = decay_coef * warmup_coef
            return coef

        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.optimizer.lr, fused=True)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        outputs = {'optimizer': optimizer, 'lr_scheduler': scheduler}
        return outputs
