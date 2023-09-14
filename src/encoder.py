from typing import Dict
from omegaconf import DictConfig

import torch

from building_block import (
    LinearBlock,
    EncoderPos,
    SlotAttentionMulti,
)


class Encoder(torch.nn.Module):
    def __init__(self, cfg: DictConfig, image_ht: int, image_wd: int, image_ch: int) -> None:
        super().__init__()
        self.net_img = EncoderPos(
            in_shape=[image_ch, image_ht, image_wd],
            channel_list=cfg.enc_img.channel_list,
            kernel_list=cfg.enc_img.kernel_list,
            stride_list=cfg.enc_img.stride_list,
            activation=cfg.enc_img.activation,
        )
        self.net_feat = torch.nn.Sequential(
            torch.nn.LayerNorm(self.net_img.out_shape[0]),
            LinearBlock(
                in_features=self.net_img.out_shape[0],
                feature_list=cfg.enc_feat.feature_list,
                act_inner=cfg.enc_feat.activation,
                act_out=None,
            ),
        )
        self.net_slot = SlotAttentionMulti(
            num_steps=cfg.enc_slot.num_steps,
            qry_size=cfg.enc_slot.qry_size,
            slot_view_size=cfg.enc_slot.slot_view_size,
            slot_attr_size=cfg.enc_slot.slot_attr_size,
            in_features=self.net_feat[-1].out_features,
            feature_res_list=cfg.enc_slot.feature_res_list,
            activation=cfg.enc_slot.activation,
        )

        # Viewpoint
        self.net_view = LinearBlock(
            in_features=cfg.enc_slot.slot_view_size,
            feature_list=cfg.enc_view.feature_list + [cfg.latent_view_size * 2],
            act_inner=cfg.enc_view.activation,
            act_out=None,
        )

        # Background
        self.net_bck_in = LinearBlock(
            in_features=cfg.enc_slot.slot_attr_size,
            feature_list=cfg.enc_bck_in.feature_list[:-1] + [cfg.enc_bck_in.feature_list[-1] + 1],
            act_inner=cfg.enc_bck_in.activation,
            act_out=None,
        )
        self.net_bck_out = LinearBlock(
            in_features=cfg.enc_bck_in.feature_list[-1],
            feature_list=cfg.enc_bck_out.feature_list + [cfg.latent_bck_size * 2],
            act_inner=cfg.enc_bck_out.activation,
            act_out=None,
        )

        # Objects
        self.split_obj = [cfg.latent_obj_size] * 2 + [1] * 3
        self.net_obj = LinearBlock(
            in_features=cfg.enc_slot.slot_attr_size,
            feature_list=cfg.enc_obj.feature_list + [sum(self.split_obj)],
            act_inner=cfg.enc_obj.activation,
            act_out=None,
        )

    def forward(
        self,
        image: torch.Tensor,      # [B, V, H, W, C]
        num_slots: int,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_views = image.shape[:2]
        x = image.flatten(end_dim=1).permute(0, 3, 1, 2)            # [B * V, C, H, W]
        x = self.net_img(x).flatten(start_dim=2).transpose(1, 2)    # [B * V, H' * W', C']
        x = self.net_feat(x).unflatten(0, [batch_size, num_views])  # [B, V, H' * W', D]
        slot_view, slot_attr = self.net_slot(x, num_slots)

        # Viewpoint
        view_param_list = self.net_view(slot_view).chunk(2, dim=-1)
        view_param_list = [param.contiguous() for param in view_param_list]
        view_mu, view_logvar = view_param_list

        # Background
        x = self.net_bck_in(slot_attr)                # [B, S, D' + 1]
        attn_sel = torch.softmax(x[..., -1:], dim=1)  # [B, S, 1]
        x = (x[..., :-1] * attn_sel).sum(1)           # [B, D']
        bck_param_list = self.net_bck_out(x).chunk(2, dim=-1)
        bck_param_list = [param.contiguous() for param in bck_param_list]
        bck_mu, bck_logvar = bck_param_list

        # Objects
        obj_param_list = self.net_obj(slot_attr).split(self.split_obj, dim=-1)
        obj_param_list = [param.contiguous() for param in obj_param_list]
        obj_mu, obj_logvar, logits_tau1, logits_tau2, logits_zeta = obj_param_list
        tau1 = torch.nn.functional.softplus(logits_tau1)
        tau2 = torch.nn.functional.softplus(logits_tau2)
        zeta = torch.sigmoid(logits_zeta)

        # Outputs
        outputs = {
            'view_mu': view_mu, 'view_logvar': view_logvar,
            'bck_mu': bck_mu, 'bck_logvar': bck_logvar,
            'obj_mu': obj_mu, 'obj_logvar': obj_logvar,
            'tau1': tau1, 'tau2': tau2, 'zeta': zeta, 'logits_zeta': logits_zeta,
        }
        return outputs
