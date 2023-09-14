from typing import List, Tuple

import math
import numpy as np

import torch


def get_grid(ht, wd):
    row_list = torch.linspace(-1, 1, ht)
    col_list = torch.linspace(-1, 1, wd)
    row_grid, col_grid = torch.meshgrid(row_list, col_list)
    grid = torch.stack([col_grid, row_grid], dim=-1)[None]  # [1, H, W, 2]
    return grid


def get_grid_dual(ht, wd):
    grid = get_grid(ht, wd)
    grid = torch.cat([1 + grid, 1 - grid], dim=-1) * 0.5
    return grid


def get_activation(activation):
    if activation == 'relu':
        net = torch.nn.ReLU(inplace=True)
    elif activation == 'silu':
        net = torch.nn.SiLU(inplace=True)
    else:
        raise ValueError
    return net


def get_net_list(net, activation):
    torch.nn.init.xavier_uniform_(net.weight)
    if net.bias is not None:
        torch.nn.init.zeros_(net.bias)
    net_list = [net]
    if activation is not None:
        net_list.append(get_activation(activation))
    return net_list


class Interpolate(torch.nn.Module):

    def __init__(self, in_size, out_size):
        super().__init__()
        assert len(in_size) == len(out_size) == 2
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x):
        if self.in_size != self.out_size:
            x = torch.nn.functional.interpolate(x, size=self.out_size, mode='nearest')
        return x

    def extra_repr(self):
        if self.in_size == self.out_size:
            return 'identity'
        else:
            return 'in_size={in_size}, out_size={out_size}'.format(**self.__dict__)


class SinusoidPosEmbedLayer(torch.nn.Module):
    def __init__(self, ht, wd, out_features) -> None:
        super().__init__()
        self.register_buffer('pos_grid', get_grid(ht, wd), persistent=False)
        self.net = torch.nn.Linear(self.pos_grid.shape[-1], out_features)
        init_scale = math.sqrt(6 / self.pos_grid.shape[-1])
        torch.nn.init.uniform_(self.net.weight, -init_scale, init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + torch.sin(self.net(self.pos_grid)).permute(0, 3, 1, 2)
        return x


class GRULayer(torch.nn.GRUCell):

    def __init__(self, input_size, hidden_size):
        super().__init__(input_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.weight_ih)
        torch.nn.init.orthogonal_(self.weight_hh)
        torch.nn.init.zeros_(self.bias_ih)
        torch.nn.init.zeros_(self.bias_hh)


class LinearLayer(torch.nn.Sequential):

    def __init__(self, in_features, out_features, activation, bias=True):
        net = torch.nn.Linear(in_features, out_features, bias=bias)
        net_list = get_net_list(net, activation)
        super().__init__(*net_list)


class ConvLayer(torch.nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size, stride, activation):
        net = self.get_net(in_channels, out_channels, kernel_size, stride)
        net_list = get_net_list(net, activation)
        super().__init__(*net_list)

    @staticmethod
    def get_net(in_channels, out_channels, kernel_size, stride):
        assert (kernel_size - stride) % 2 == 0
        padding = (kernel_size - stride) // 2
        net = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        return net


class UpConvLayer(torch.nn.Sequential):

    def __init__(self, in_shape, out_shape, kernel_size, activation):
        net = self.get_net(in_shape, out_shape, kernel_size)
        net_list = get_net_list(net, activation)
        net_list = [Interpolate(in_shape[1:], out_shape[1:])] + net_list
        super().__init__(*net_list)

    @staticmethod
    def get_net(in_shape, out_shape, kernel_size):
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        net = torch.nn.Conv2d(in_shape[0], out_shape[0], kernel_size, stride=1, padding=padding)
        return net


class ConvTLayer(torch.nn.Sequential):

    def __init__(self, in_channels, out_shape, kernel_size, stride, activation):
        net = self.get_net(in_channels, out_shape, kernel_size, stride)
        net_list = get_net_list(net, activation)
        super().__init__(*net_list)

    @staticmethod
    def get_net(in_channels, out_shape, kernel_size, stride):
        assert (kernel_size - stride) % 2 == 0
        padding = (kernel_size - stride) // 2
        net = torch.nn.ConvTranspose2d(in_channels, out_shape[0], kernel_size, stride=stride, padding=padding)
        return net


class LinearBlock(torch.nn.Sequential):

    def __init__(self, in_features, feature_list, act_inner, act_out):
        net_list = []
        for idx, num_features in enumerate(feature_list):
            activation = act_inner if idx < len(feature_list) - 1 else act_out
            net = LinearLayer(
                in_features=in_features,
                out_features=num_features,
                activation=activation,
            )
            net_list.append(net)
            in_features = num_features
        self.out_features = in_features
        super().__init__(*net_list)


class ConvBlock(torch.nn.Sequential):

    def __init__(self, in_shape, channel_list, kernel_list, stride_list, act_inner, act_out):
        assert len(channel_list) == len(kernel_list) == len(stride_list)
        net_list = []
        in_ch, in_ht, in_wd = in_shape
        for idx, (num_channels, kernel_size, stride) in enumerate(zip(channel_list, kernel_list, stride_list)):
            activation = act_inner if idx < len(channel_list) - 1 else act_out
            net = ConvLayer(
                in_channels=in_ch,
                out_channels=num_channels,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
            )
            net_list.append(net)
            in_ch = num_channels
            in_ht = (in_ht - 1) // stride + 1
            in_wd = (in_wd - 1) // stride + 1
        self.out_shape = [in_ch, in_ht, in_wd]
        super().__init__(*net_list)


class UpConvBlock(torch.nn.Sequential):

    def __init__(self, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, act_inner, act_out):
        assert len(channel_list_rev) == len(kernel_list_rev) == len(stride_list_rev)
        net_list_rev = []
        for idx, (num_channels, kernel_size, stride) in enumerate(
                zip(channel_list_rev, kernel_list_rev, stride_list_rev)):
            activation = act_out if idx == 0 else act_inner
            in_shape = [num_channels] + [(val - 1) // stride + 1 for val in out_shape[1:]]
            net = UpConvLayer(
                in_shape=in_shape,
                out_shape=out_shape,
                kernel_size=kernel_size,
                activation=activation,
            )
            net_list_rev.append(net)
            out_shape = in_shape
        self.in_shape = out_shape
        super().__init__(*reversed(net_list_rev))


class ConvTBlock(torch.nn.Sequential):

    def __init__(self, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, act_inner, act_out):
        assert len(channel_list_rev) == len(kernel_list_rev) == len(stride_list_rev)
        net_list_rev = []
        for idx, (num_channels, kernel_size, stride) in enumerate(
                zip(channel_list_rev, kernel_list_rev, stride_list_rev)):
            activation = act_out if idx == 0 else act_inner
            net = ConvTLayer(
                in_channels=num_channels,
                out_shape=out_shape,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
            )
            net_list_rev.append(net)
            out_shape = [num_channels] + [(val - 1) // stride + 1 for val in out_shape[1:]]
        self.in_shape = out_shape
        super().__init__(*reversed(net_list_rev))


class EncoderPos(torch.nn.Module):

    def __init__(self, in_shape, channel_list, kernel_list, stride_list, activation):
        super().__init__()
        self.net_image = ConvBlock(
            in_shape=in_shape,
            channel_list=channel_list,
            kernel_list=kernel_list,
            stride_list=stride_list,
            act_inner=activation,
            act_out=activation,
        )
        self.register_buffer('grid', get_grid_dual(*self.net_image.out_shape[1:]).permute(0, 3, 1, 2), persistent=False)
        self.net_grid = ConvLayer(
            in_channels=4,
            out_channels=self.net_image.out_shape[0],
            kernel_size=1,
            stride=1,
            activation=None,
        )
        self.out_shape = self.net_image.out_shape

    def forward(self, x):
        x = self.net_image(x)
        x = x + self.net_grid(self.grid)
        return x


class DecoderBasic(torch.nn.Module):

    def __init__(self, in_features, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, feature_list_rev,
                 activation):
        super().__init__()
        self.conv = ConvTBlock(
            out_shape=out_shape,
            channel_list_rev=channel_list_rev,
            kernel_list_rev=kernel_list_rev,
            stride_list_rev=stride_list_rev,
            act_inner=activation,
            act_out=None,
        )
        self.linear = LinearBlock(
            in_features=in_features,
            feature_list=[*reversed(feature_list_rev)] + [np.prod(self.conv.in_shape)],
            act_inner=activation,
            act_out=None if len(channel_list_rev) == 0 else activation,
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], *self.conv.in_shape)
        x = self.conv(x)
        return x


class DecoderComplex(torch.nn.Module):

    def __init__(self, in_features, out_shape, channel_list_rev, kernel_list_rev, stride_list_rev, feature_list_rev,
                 num_layers, d_model, nhead, dim_feedforward, activation):
        super().__init__()
        self.conv_out = ConvTBlock(
            out_shape=out_shape,
            channel_list_rev=channel_list_rev,
            kernel_list_rev=kernel_list_rev,
            stride_list_rev=stride_list_rev,
            act_inner=activation,
            act_out=None,
        )
        self.conv_cvt = ConvTLayer(
            in_channels=d_model,
            out_shape=self.conv_out.in_shape,
            kernel_size=3,
            stride=1,
            activation=activation,
        )
        self.xformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=0,
                activation=get_activation(activation),
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.pos_embed = SinusoidPosEmbedLayer(
            ht=self.conv_out.in_shape[1],
            wd=self.conv_out.in_shape[2],
            out_features=d_model,
        )
        self.linear = LinearBlock(
            in_features=in_features,
            feature_list=[*reversed(feature_list_rev)] + [d_model],
            act_inner=activation,
            act_out=activation,
        )

    def forward(self, x):
        x = self.linear(x)[..., None, None]
        x = self.pos_embed(x).flatten(start_dim=2).transpose(1, 2)
        x = self.xformer(x).unflatten(1, self.conv_out.in_shape[1:]).permute(0, 3, 1, 2)
        x = self.conv_cvt(x)
        x = self.conv_out(x)
        return x


class SlotAttentionMulti(torch.nn.Module):

    def __init__(
        self,
        num_steps: int,
        qry_size: int,
        slot_view_size: int,
        slot_attr_size: int,
        in_features: int,
        feature_res_list: List[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.slot_view_size = slot_view_size
        self.slot_attr_size = slot_attr_size
        slot_full_size = slot_view_size + slot_attr_size
        self.view_loc = torch.nn.Parameter(torch.zeros([slot_view_size]))
        self.view_log_scl = torch.nn.Parameter(torch.zeros([slot_view_size]))
        self.attr_loc = torch.nn.Parameter(torch.zeros([slot_attr_size]))
        self.attr_log_scl = torch.nn.Parameter(torch.zeros([slot_attr_size]))
        self.coef_key = 1 / math.sqrt(qry_size)
        self.split_key_val = [qry_size, slot_full_size]
        self.net_key_val = torch.nn.Sequential(
            torch.nn.LayerNorm(in_features),
            LinearLayer(
                in_features=in_features,
                out_features=sum(self.split_key_val),
                activation=None,
                bias=False,
            ),
        )
        self.net_qry = torch.nn.Sequential(
            torch.nn.LayerNorm(slot_full_size),
            LinearLayer(
                in_features=slot_full_size,
                out_features=qry_size,
                activation=None,
                bias=False,
            ),
        )
        self.net_upd = GRULayer(slot_full_size, slot_full_size)
        self.net_res = torch.nn.Sequential(
            torch.nn.LayerNorm(slot_full_size),
            LinearBlock(
                in_features=slot_full_size,
                feature_list=feature_res_list + [slot_full_size],
                act_inner=activation,
                act_out=None,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,          # [B, V, N, D]
        num_slots: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_views = x.shape[:2]
        x = x.flatten(end_dim=1)  # [B * V, N, D]
        x_key, x_val = self.net_key_val(x).split(self.split_key_val, dim=-1)
        x_key = x_key.transpose(-1, -2).contiguous() * self.coef_key  # [B * V, D_k, N]
        x_val = x_val.contiguous()                                    # [B * V, N, D_s]

        def compute_slot_full() -> torch.Tensor:
            slot_full = torch.cat([
                slot_view[:, :, None].expand(-1, -1, num_slots, -1),
                slot_attr[:, None].expand(-1, num_views, -1, -1),
            ], dim=-1).flatten(end_dim=1)    # [B * V, S, D_s]
            x_qry = self.net_qry(slot_full)  # [B * V, S, D_q]
            logits_attn = torch.bmm(x_qry, x_key)                                 # [B * V, S, N]
            attn = torch.softmax(torch.log_softmax(logits_attn, dim=-2), dim=-1)  # [B * V, S, N]
            x_upd = torch.bmm(attn, x_val).flatten(end_dim=1)  # [B * V * S, D_s]
            slot_full = slot_full.flatten(end_dim=1)           # [B * V * S, D_s]
            x_main = self.net_upd(x_upd, slot_full).unflatten(0, [batch_size, num_views, num_slots])  # [B, V, S, D_s]
            slot_full = x_main + self.net_res(x_main)                                                 # [B, V, S, D_s]
            return slot_full

        noise_attr = torch.randn([batch_size, num_slots, self.slot_attr_size], device=x.device)
        slot_attr = self.attr_loc + torch.exp(self.attr_log_scl) * noise_attr
        noise_view = torch.randn([batch_size, num_views, self.slot_view_size], device=x.device)
        slot_view = self.view_loc + torch.exp(self.view_log_scl) * noise_view
        for _ in range(self.num_steps):
            slot_full = compute_slot_full()
            slot_view_raw, slot_attr_raw = slot_full.split([self.slot_view_size, self.slot_attr_size], dim=-1)
            slot_view = slot_view_raw.mean(2)
            slot_attr = slot_attr_raw.mean(1)
        return slot_view, slot_attr
