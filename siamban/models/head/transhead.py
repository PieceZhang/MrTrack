from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import random
from functools import reduce, partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from timm.layers import DropPath

from .transhead_utils import *
from siamban.core.config import cfg
from mamba1_arch.mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref



class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        '''
            Args:
                x (torch.Tensor): (B, L, C), input tensor
            Returns:
                torch.Tensor: (B, L, C), output tensor
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class _ConvHead(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(_ConvHead, self).__init__()
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_channels, kernel_size=1)
                )


    def forward(self, search):
        search = self.conv_search(search)
        out = self.head(search)
        return out




class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv, attn_pos, _):
        '''
            Args:
                q (torch.Tensor): (B, L_q, C)
                kv (torch.Tensor): (B, L_kv, C)
            Returns:
                torch.Tensor: (B, L_q, C)
        '''
        B, q_N, C = q.shape
        kv_N = kv.shape[1]

        q = self.q(q).reshape(B, q_N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(kv).reshape(B, kv_N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = q @ k.transpose(-2, -1)
        attn = attn * self.scale
        if attn_pos is not None:
            if attn_pos.shape[-1] != attn.shape[-1]:  # expand the length of attn_pos with 0
                zero = torch.zeros([*attn_pos.shape[:-1], attn.shape[-1] - attn_pos.shape[-1]], device=attn.device, dtype=attn.dtype)
                attn_pos = torch.cat([attn_pos, zero], dim=-1)
            attn = attn + attn_pos
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, q_N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Decoder(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=nn.Identity(), act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Decoder, self).__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_kv = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)

        self.drop_path = drop_path
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, kv, attn_pos, mt):
        '''
            Args:
                mt:
                q (torch.Tensor): (B, L_q, C)
                kv (torch.Tensor): (B, L_kv, C)
                q_ape (torch.Tensor | None): (1 or B, L_q, C), absolute positional encoding for q
                k_ape (torch.Tensor | None): (1 or B, L_kv, C), absolute positional encoding for k
                attn_pos (torch.Tensor | None): (1 or B, num_heads, L_q, L_kv), untied positional encoding
            Returns:
                torch.Tensor: (B, L_q, C)
        '''
        q = q + self.drop_path(self.attn(self.norm1_q(q), self.norm1_kv(kv), attn_pos, None))
        q = q + self.drop_path(self.mlp(self.norm2(q)))

        return q


class SS2Dreg(nn.Module):
    # ref https://github.com/csguoh/MambaIR
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_norm_r = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_r = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.Lreg = cfg.MOTION.KWARGS.Lreg  # cfg.MOTION.KWARGS.Lreg
        self.reg = nn.Parameter(torch.zeros(1, self.d_inner, self.Lreg))
        # init reg with xavier
        trunc_normal_(self.reg, std=.02)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError


        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D


    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        r = torch.stack([self.reg for _ in range(B)], dim=0)
        r = torch.stack([r for _ in range(K)], dim=1).permute(0, 1, 3, 4, 2)
        r = r.squeeze(-1)

        xs = torch.cat([xs, r], dim=-1)
        L = L + self.Lreg

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        r = out_y[:, :, :, -self.Lreg:]  # B, K, C, Lreg
        L = L - self.Lreg
        r = r.sum(dim=1)

        out_y = out_y[:, :, :, :-self.Lreg]
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y, r

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4, r = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        r = self.out_norm_r(r.permute(0, 2, 1))
        r = self.out_proj_r(r).permute(0, 2, 1)
        return out, r


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class RegisterMamba(nn.Module):
    # from VSSBlock in MambaIR
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_x = norm_layer(hidden_dim)
        self.ssd = SS2Dreg(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 3, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 3, hidden_dim, kernel_size=3, padding=1),
            ChannelAttention(hidden_dim)
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,H,W,C]
        xn = self.ln_x(x)
        xn, r = self.ssd(xn)
        xn = self.drop_path(xn)
        x = x*self.skip_scale + xn
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.permute(0, 3, 1, 2)
        return x, r


class SS2Dretrival(nn.Module):
    # ref https://github.com/csguoh/MambaIR
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_r = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.Lreg = cfg.MOTION.KWARGS.Lreg  # cfg.MOTION.KWARGS.Lreg


    def forward_core(self, x: torch.Tensor, r):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        r = r.permute(0, 2, 1)
        r = torch.stack([r for _ in range(K)], dim=1)
        xs = torch.cat([r, xs], dim=-1)
        Lreg = r.shape[-1]
        L = L + Lreg

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        out_y = out_y[:, :, :, Lreg:]
        L = L - Lreg

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, r, **kwargs):
        B, H, W, C = x.shape
        # _, L, _ = r.shape  # B, C, Lreg
        r = self.in_proj_r(r)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x, r)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError


        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D


class RetrivalMamba(nn.Module):
    # from VSSBlock in MambaIR
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_x = norm_layer(hidden_dim)
        self.ln_r = norm_layer(hidden_dim)
        self.ssd = SS2Dretrival(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 3, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 3, hidden_dim, kernel_size=3, padding=1),
            ChannelAttention(hidden_dim)
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, r, x):
        B, H, W, C = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,H,W,C]
        xn = self.ln_x(x)
        r = r.permute(0, 2, 1)  # B, K, Lreg, C
        r = self.ln_r(r)
        xn = self.ssd(xn, r)
        xn = self.drop_path(xn)
        x = x*self.skip_scale + xn
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.permute(0, 3, 1, 2)
        return x


class Register(nn.Module):
    def __init__(self, Lreg, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super(Register, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.norm_x = norm_layer(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.Lreg = Lreg
        self.reg = nn.Parameter(torch.randn(1, Lreg, dim))


    def forward(self, x):
        x_sc = x  # shortcut

        B, C, Hx, Wx = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # B, N, C
        B, Nx, C = x.shape
        Nr = self.Lreg

        x = self.norm_x(x)

        x = torch.concatenate([x, self.reg.repeat(B, 1, 1)], dim=1)

        q = self.q(x).reshape(B, Nx+Nr, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = self.k(x).reshape(B, Nx+Nr, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = self.v(x).reshape(B, Nx+Nr, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn1 = k @ q.transpose(-2, -1)  # HxWx * HzWz
        attn1 = attn1 * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        x = attn1 @ v

        x = x.transpose(1, 2).reshape(B, Nx+Nr, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        reg = x[:, -Nr:, :]
        x = x[:, :-Nr, :]

        x, reg = x.permute(0, 2, 1).view(B, -1, Hx, Wx), reg.permute(0, 2, 1)
        x = x + x_sc

        return x, reg


class AnchorAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super(AnchorAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.norm_r = norm_layer(dim)
        self.norm_z = norm_layer(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.a = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, r, z):
        z_sc = z  # shortcut

        B, C, Hz, Wz = z.shape
        r = r.flatten(2).permute(0, 2, 1)  # B, N, C
        z = z.flatten(2).permute(0, 2, 1)  # B, N, C
        B, Nr, C = r.shape  # r is in variable length
        B, Nz, C = z.shape

        r = self.norm_r(r)
        z = self.norm_z(z)

        q = self.q(r).reshape(B, Nr, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k = self.k(r).reshape(B, Nr, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        v = self.v(z).reshape(B, Nz, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        a = self.a(z).reshape(B, Nz, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn1 = k @ a.transpose(-2, -1)  # HxWx * HzWz
        attn1 = attn1 * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        v = attn1 @ v

        attn2 = a @ q.transpose(-2, -1)
        attn2 = attn2 * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        z = attn2 @ v

        z = z.transpose(1, 2).reshape(B, Nz, C)
        z = self.proj(z)
        z = self.proj_drop(z)
        z = z.permute(0, 2, 1).view(B, -1, Hz, Wz)
        z = z + z_sc

        return z


class TransHead(nn.Module):
    def __init__(self, in_channels, enc_num, dec_num=1, cls_out_channels=2, num_heads=8, norm_layer=nn.LayerNorm):
        super(TransHead, self).__init__()
        assert dec_num == 1
        dim = in_channels

        self.reger = RegisterMamba(dim)

        self.reg_retrival = RetrivalMamba(dim)

        self.dec = Decoder(dim, num_heads)
        self.out_norm = norm_layer(dim)

        self.cls_mlp = _ConvHead(256, 256, cls_out_channels)
        self.reg_mlp = _ConvHead(256, 256, 4)

        # self.loc_scale = nn.Parameter(torch.ones(len(in_channels)))
        self.loc_scale = nn.Parameter(torch.tensor([1.]))

        self.untied_z_pos_dec = Untied2DPositionalEncoder(dim, num_heads, 12, 12, with_q=False)
        self.untied_x_pos_dec = Untied2DPositionalEncoder(dim, num_heads, 24, 24)
        self.dec_rpe_index = generate_2d_concatenated_cross_attention_relative_positional_encoding_index((12, 12), (24, 24))
        self.dec_rpe_bias_table = RelativePosition2DEncoder(num_heads, self.dec_rpe_index.max() + 1)

        # self.init_parameters()

    def init_parameters(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def forward(self, z, x, iftrain, regbank=None):
        if iftrain:
            assert regbank is None
            assert isinstance(x, list)
            z = z.clone()
            xlist = x

            # register
            regbank = None
            for i, x in enumerate(xlist):
                x, reg = self.reger(x)
                enlarge_factor = random.randint(cfg.MOTION.KWARGS.Enlarge_factor_range[0], cfg.MOTION.KWARGS.Enlarge_factor_range[1])
                reg = reg.repeat(1, 1, enlarge_factor)  # [B, C, L]
                if regbank is None:
                    regbank = reg
                else:
                    regbank = torch.concatenate([regbank, reg], dim=-1)

            B, _, H, W = x.shape  # this is the last x, predict the last x, the other for reg

            # register information retrival
            z = self.reg_retrival(regbank, z)
        else:
            assert not isinstance(x, list)
            z = z.clone()

            # register
            x, reg = self.reger(x)
            if regbank is not None:
                regbank = torch.concatenate([regbank, reg], dim=-1)
            else:
                regbank = reg

            if regbank.shape[-1] > cfg.MOTION.KWARGS.Enlarge_factor_range[-1] * cfg.MOTION.KWARGS.L_range[-1] * cfg.MOTION.KWARGS.Lreg:
                # regbank [B, C, L]
                regbank = regbank[:, :, -cfg.MOTION.KWARGS.Enlarge_factor_range[-1] * cfg.MOTION.KWARGS.L_range[-1] * cfg.MOTION.KWARGS.Lreg:]

            B, _, H, W = x.shape

            # register information retrival
            z = self.reg_retrival(regbank, z)

        z = z.flatten(2).permute(0, 2, 1)
        x = x.flatten(2).permute(0, 2, 1)

        # decoder
        attn_pos_dec = None
        z_learned_pos_k = self.untied_z_pos_dec()
        x_learned_pos_q, x_learned_pos_k = self.untied_x_pos_dec()
        attn_pos_dec = x_learned_pos_q @ torch.cat((z_learned_pos_k, x_learned_pos_k), dim=1).transpose(-2, -1).unsqueeze(0)
        attn_pos_dec = attn_pos_dec + self.dec_rpe_bias_table(self.dec_rpe_index)  # (1, num_heads, Lx, Lx+Lz)

        x = self.dec(x, torch.cat([z, x], dim=1), attn_pos_dec, None)

        x = self.out_norm(x)
        x = x.permute(0, 2, 1).view(B, -1, H, W)

        # head
        cls = self.cls_mlp(x)
        bbox = self.reg_mlp(x)  # .sigmoid()
        bbox = torch.exp(bbox * self.loc_scale)

        return cls, bbox, regbank


if __name__ == '__main__':
    z = torch.randn(2, 256, 12, 12).cuda()
    x = torch.randn(2, 256, 24, 24).cuda()
    xlist = [x, x, x]
    model = TransHead(256, 0, 1).cuda()
    cls, bbox, regbank = model(z, xlist, True)
    print(cls.shape, bbox.shape, regbank.shape)
