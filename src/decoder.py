from typing import Dict, Tuple
from omegaconf import DictConfig

import torch

from building_block import (
    get_grid,
    LinearBlock,
    DecoderBasic,
    DecoderComplex,
)


def select_by_index(x, index):
    x_ndim = x.ndim
    index_ndim = index.ndim
    index = index.reshape(list(index.shape) + [1] * (x_ndim - index_ndim))
    index = index.expand([-1] * index_ndim + list(x.shape[index_ndim:]))
    x = torch.gather(x, index_ndim - 1, index)
    return x


class Decoder(torch.nn.Module):
    def __init__(self, cfg: DictConfig, image_ht: int, image_wd: int, image_ch: int) -> None:
        super().__init__()
        self.register_buffer('pos_grid_noise', get_grid(image_ht, image_wd), persistent=False)
        self.image_ch = image_ch
        self.use_shadow = cfg.use_shadow
        self.max_shadow_val = cfg.max_shadow_val

        # Background
        if cfg.dec_bck.use_complex:
            self.net_bck = DecoderComplex(
                in_features=cfg.latent_view_size + cfg.latent_bck_size,
                out_shape=[image_ch, image_ht, image_wd],
                channel_list_rev=cfg.dec_bck.channel_list_rev,
                kernel_list_rev=cfg.dec_bck.kernel_list_rev,
                stride_list_rev=cfg.dec_bck.stride_list_rev,
                feature_list_rev=cfg.dec_bck.feature_list_rev,
                num_layers=cfg.dec_bck.num_layers,
                d_model=cfg.dec_bck.d_model,
                nhead=cfg.dec_bck.nhead,
                dim_feedforward=cfg.dec_bck.dim_feedforward,
                activation=cfg.dec_bck.activation,
            )
        else:
            self.net_bck = DecoderBasic(
                in_features=cfg.latent_view_size + cfg.latent_bck_size,
                out_shape=[image_ch, image_ht, image_wd],
                channel_list_rev=cfg.dec_bck.channel_list_rev,
                kernel_list_rev=cfg.dec_bck.kernel_list_rev,
                stride_list_rev=cfg.dec_bck.stride_list_rev,
                feature_list_rev=cfg.dec_bck.feature_list_rev,
                activation=cfg.dec_bck.activation,
            )

        # Objects
        in_features = cfg.latent_view_size + cfg.latent_obj_size
        out_channels = image_ch + 3 if self.use_shadow else image_ch + 1
        self.net_obj_misc = LinearBlock(
            in_features=in_features,
            feature_list=cfg.dec_obj_misc.feature_list + [3],
            act_inner=cfg.dec_obj_misc.activation,
            act_out=None,
        )
        if cfg.dec_obj_img.use_complex:
            self.net_obj_img = DecoderComplex(
                in_features=in_features,
                out_shape=[out_channels, image_ht, image_wd],
                channel_list_rev=cfg.dec_obj_img.channel_list_rev,
                kernel_list_rev=cfg.dec_obj_img.kernel_list_rev,
                stride_list_rev=cfg.dec_obj_img.stride_list_rev,
                feature_list_rev=cfg.dec_obj_img.feature_list_rev,
                num_layers=cfg.dec_obj_img.num_layers,
                d_model=cfg.dec_obj_img.d_model,
                nhead=cfg.dec_obj_img.nhead,
                dim_feedforward=cfg.dec_obj_img.dim_feedforward,
                activation=cfg.dec_obj_img.activation,
            )
        else:
            self.net_obj_img = DecoderBasic(
                in_features=in_features,
                out_shape=[out_channels, image_ht, image_wd],
                channel_list_rev=cfg.dec_obj_img.channel_list_rev,
                kernel_list_rev=cfg.dec_obj_img.kernel_list_rev,
                stride_list_rev=cfg.dec_obj_img.stride_list_rev,
                feature_list_rev=cfg.dec_obj_img.feature_list_rev,
                activation=cfg.dec_obj_img.activation,
            )

    @staticmethod
    def compute_mask(
        shp: torch.Tensor,                   # [B, V, S, H, W, 1]
        log_shp: torch.Tensor,               # [B, V, S, H, W, 1]
        pres: torch.Tensor,                  # [B, S, 1]
        logits_pres: torch.Tensor,           # [B, S, 1]
        log_ord: torch.Tensor,               # [B, V, S, 1]
        ratio_stick_breaking: torch.Tensor,  # []
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        index_sel = torch.argsort(log_ord.squeeze(-1), dim=2, descending=True)
        log_pres = torch.nn.functional.logsigmoid(logits_pres)[:, None, :, None, None]
        log_ord = log_ord[:, :, :, None, None]
        shp_mul_pres = shp * pres[:, None, :, None, None]
        mask_bck = (1 - shp_mul_pres).prod(2)                                   # [B, V, H, W, 1]
        mask_obj_aux_rel = torch.softmax(log_shp + log_pres + log_ord, dim=2)  # [B, V, S, H, W, 1]
        mask_obj_aux = (1 - mask_bck[:, :, None]) * mask_obj_aux_rel          # [B, V, S, H, W, 1]
        shp_mul_pres = select_by_index(shp_mul_pres, index_sel)
        ones = torch.ones([*shp_mul_pres.shape[:2], 1, *shp_mul_pres.shape[3:]], device=shp_mul_pres.device)
        mask_obj = shp_mul_pres * torch.cat([ones, (1 - shp_mul_pres[:, :, :-1]).cumprod(2)], dim=2)
        index_sel = torch.argsort(index_sel, dim=2)
        mask_obj = select_by_index(mask_obj, index_sel)
        mask_obj = mask_obj_aux - mask_obj_aux.detach() + mask_obj.detach()
        mask_obj = ratio_stick_breaking * mask_obj + (1 - ratio_stick_breaking) * mask_obj_aux
        return mask_bck, mask_obj

    def forward(
        self,
        view_latent: torch.Tensor,           # [B, V, D_v]
        bck_latent: torch.Tensor,            # [B, D_b]
        obj_latent: torch.Tensor,            # [B, S, D_o]
        pres: torch.Tensor,                  # [B, S, 1]
        logits_pres: torch.Tensor,           # [B, S, 1]
        temp_shp: torch.Tensor,              # []
        noise_scale: torch.Tensor,           # []
        noise_min: torch.Tensor,             # []
        noise_max: torch.Tensor,             # []
        ratio_stick_breaking: torch.Tensor,  # []
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_views = view_latent.shape[:2]
        num_slots = obj_latent.shape[1]

        # Latent Variables
        full_bck_latent = torch.cat([
            view_latent,
            bck_latent[:, None].expand(-1, num_views, -1),
        ], dim=-1).flatten(end_dim=1)  # [B * V, D_v + D_b]
        full_obj_latent = torch.cat([
            view_latent[:, :, None].expand(-1, -1, num_slots, -1),
            obj_latent[:, None].expand(-1, num_views, -1, -1),
        ], dim=-1).flatten(end_dim=2)  # [B * V * S, D_v + D_o]

        # Background
        x = self.net_bck(full_bck_latent).permute(0, 2, 3, 1)  # [B * V, H, W, C]
        bck_imp = x.unflatten(0, [batch_size, num_views])      # [B, V, H, W, C]

        # Objects
        x = self.net_obj_misc(full_obj_latent).unflatten(0, [batch_size, num_views, num_slots])
        log_ord = x[..., :1].contiguous() / temp_shp  # [B, V, S, 1]
        trs = torch.tanh(x[..., 1:].contiguous())     # [B, V, S, 2]
        x = self.net_obj_img(full_obj_latent).permute(0, 2, 3, 1)  # [B * V * S, H, W, C + 3]
        x = x.unflatten(0, [batch_size, num_views, num_slots])     # [B, V, S, H, W, C + 3]
        apc = x[..., :self.image_ch].contiguous()                                  # [B, V, S, H, W, C]
        logits_shp_imp = x[..., self.image_ch:self.image_ch + 1].contiguous() - 3  # [B, V, S, H, W, 1]
        logits_shp_imp_rand = torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli(
            temperature=temp_shp, logits=logits_shp_imp).rsample()
        logits_shp_imp = ratio_stick_breaking * logits_shp_imp + (1 - ratio_stick_breaking) * logits_shp_imp_rand
        shp_imp = torch.sigmoid(logits_shp_imp)                                    # [B, V, S, H, W, 1]
        log_shp_imp = torch.nn.functional.logsigmoid(logits_shp_imp)               # [B, V, S, H, W, 1]

        # Shadows
        if self.use_shadow:
            sdw_logits_apc_raw = x[..., self.image_ch + 1:self.image_ch + 2].contiguous()    # [B, V, S, H, W, 1]
            sdw_apc_raw = (torch.sigmoid(sdw_logits_apc_raw) - 1) * self.max_shadow_val + 1  # [B, V, S, H, W, 1]
            sdw_apc = (bck_imp[:, :, None] + 1) * sdw_apc_raw - 1                            # [B, V, S, H, W, C]
            sdw_logits_shp = x[..., self.image_ch + 2:].contiguous() - 3                     # [B, V, S, H, W, 1]
            sdw_logits_shp_rand = torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli(
                temperature=temp_shp, logits=sdw_logits_shp).rsample()
            sdw_logits_shp = ratio_stick_breaking * sdw_logits_shp + (1 - ratio_stick_breaking) * sdw_logits_shp_rand
            sdw_shp = torch.sigmoid(sdw_logits_shp)                                          # [B, V, S, H, W, 1]
            sdw_log_shp = torch.nn.functional.logsigmoid(sdw_logits_shp)                     # [B, V, S, H, W, 1]
            sdw_mask_bck, sdw_mask_obj = self.compute_mask(
                sdw_shp, sdw_log_shp, pres, logits_pres, torch.zeros_like(log_ord), ratio_stick_breaking=0)
            bck = sdw_mask_bck * bck_imp + (sdw_mask_obj * sdw_apc).sum(2)                   # [B, V, S, H, W, C]
            noise_coef = self.pos_grid_noise[:, None, None] - trs[:, :, :, None, None]       # [B, V, S, H, W, 2]
            noise_coef = -noise_scale * noise_coef.square().sum(-1, keepdims=True)           # [B, V, S, H, W, 1]
            noise_coef = noise_min + (noise_max - noise_min) * (1 - torch.exp(noise_coef))   # [B, V, S, H, W, 1]
            apc_aux = apc + noise_coef * torch.randn_like(apc)                               # [B, V, S, H, W, C]
            shp = shp_imp * (1 - sdw_shp)                                                    # [B, V, S, H, W, 1]
            log_shp = log_shp_imp + sdw_log_shp - sdw_logits_shp                             # [B, V, S, H, W, 1]
        else:
            sdw_apc = torch.zeros_like(apc)
            sdw_shp = sdw_logits_shp = torch.zeros_like(shp_imp)
            bck = bck_imp
            apc_aux = apc
            shp = shp_imp
            log_shp = log_shp_imp

        # Composition
        mask_bck, mask_obj = self.compute_mask(shp, log_shp, pres, logits_pres, log_ord, ratio_stick_breaking)
        mask_bck_imp, mask_obj_imp = self.compute_mask(shp_imp, log_shp_imp, pres, logits_pres, log_ord,
                                                       ratio_stick_breaking)
        recon = mask_bck * bck + (mask_obj * apc).sum(2)
        recon_aux = mask_bck * bck + (mask_obj * apc_aux).sum(2)
        recon_aux_imp = mask_bck_imp * bck_imp + (mask_obj_imp * apc_aux).sum(2)

        # Outputs
        outputs = {
            'recon': recon, 'recon_aux': recon_aux, 'recon_aux_imp': recon_aux_imp,
            'mask_bck': mask_bck, 'mask_obj': mask_obj, 'log_ord': log_ord, 'trs': trs,
            'bck': bck, 'bck_imp': bck_imp, 'apc': apc, 'apc_aux': apc_aux, 'shp': shp, 'shp_imp': shp_imp,
            'sdw_apc': sdw_apc, 'sdw_shp': sdw_shp, 'logits_shp_imp': logits_shp_imp, 'sdw_logits_shp': sdw_logits_shp,
        }
        return outputs
