
import jittor as jt
from jittor import init
from jittor import nn
from typing import Optional, TypeVar, Union, Tuple
T = TypeVar('T')
_scalar_or_tuple_any_t = Union[T, Tuple[T, ...]]

_size_any_t = _scalar_or_tuple_any_t[int]
_ratio_any_t = _scalar_or_tuple_any_t[float]
class SiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def execute(self, x: jt.Var) -> jt.Var:
        return jt.nn.silu(x)
class UpSample(nn.Module):
    scale_factor: Optional[_ratio_any_t]
    def __init__(self, scale_factor: Optional[_ratio_any_t] = None, mode:str = 'nearest', 
        align_corners: Optional[bool] = None) -> None:
        super().__init__() 
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def execute(self, x:jt.Var) -> jt.Var:
        return jt.nn.interpolate(x, size=None, 
            scale_factor=self.scale_factor, 
            mode=self.mode, align_corners=self.align_corners)

class SelfAttention(nn.Module):

    def __init__(self, channels: int, size: int):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = jt.attention.MultiheadAttention(channels, 4, self_attention=True)
        #raise RuntimeError('original source: <nn.MultiheadAttention(channels, 4, batch_first=True)>, MultiheadAttention is not supported in Jittor yet. We will appreciate it if you provide an implementation of MultiheadAttention and make pull request at https://github.com/Jittor/jittor.')
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]), 
            nn.Linear(channels, channels), 
            nn.GELU(), 
            nn.Linear(channels, channels)
        )

    def execute(self, x: jt.Var) -> jt.Var:
        x = jt.transpose(x.view(((- 1), self.channels, (self.size * self.size))), 1, 2)
        x_ln:jt.Var = self.ln(x)
        # TODO: Check consistency of definitions: pay attention
        # whether we need transpose here
        (attention_value, _) = self.mha(x_ln)
        attention_value:jt.Var = attention_value + x #residual network that bypass a self-attention
        attention_value = self.ff_self(attention_value) + attention_value #another residual network
        attention_value_tr:jt.Var = jt.transpose(attention_value, 2, 1)
        # TODO: Check definition
        return attention_value_tr.view((-1, self.channels, self.size, self.size))

class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mid_channels:Optional[int]=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv(in_channels, mid_channels, 3, padding=1, bias=False), 
            nn.GroupNorm(1, mid_channels, affine=None), 
            nn.GELU(), 
            nn.Conv(mid_channels, out_channels, 3, padding=1, bias=False), 
            nn.GroupNorm(1, out_channels, affine=None)
        )

    def execute(self, x:jt.Var) -> jt.Var:
        if self.residual:
            return jt.nn.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Pool(2, op='maximum'), 
            DoubleConv(in_channels, in_channels, residual=True), 
            DoubleConv(in_channels, out_channels)
        )
        self.emb_layer = nn.Sequential(
            SiLU(), 
            nn.Linear(emb_dim, out_channels)
        )

    def execute(self, x: jt.Var, t) -> jt.Var:
        x = self.maxpool_conv(x)
        emb:jt.Var = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up(nn.Module):

    def __init__(self, in_channels: int, out_channels:int, emb_dim=256):
        super().__init__()
        
        self.up = UpSample(scale_factor=2, mode='bilinear', align_corners=True)
        #raise AttributeError("origin source: <nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)>, Upsample in Jittor has no Attribute align_corners")
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True), 
            DoubleConv(in_channels, out_channels, (in_channels // 2)), 
        )
        self.emb_layer = nn.Sequential(
            SiLU(),
            nn.Linear(emb_dim, out_channels)
        )

    def execute(self, x:jt.Var, skip_x:jt.Var, t:jt.Var) -> jt.Var:
        x = self.up(x)
        x = jt.contrib.concat([skip_x, x], dim=1)
        x = self.conv(x)
        emb:jt.Var = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class UNet(nn.Module):

    def __init__(self, c_in=3, c_out=3, time_dim=256, device='cuda'):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv(64, c_out, 1)

    def pos_encoding(self, t:jt.Var, channels: int):
        inv_freq:jt.Var = (1.0 / (10000 ** (jt.arange(0, channels, 2).float() / channels)))
        pos_enc_a = jt.sin((t.repeat(1, (channels // 2)) * inv_freq))
        pos_enc_b = jt.cos((t.repeat(1, (channels // 2)) * inv_freq))
        pos_enc = jt.contrib.concat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def execute(self, x:jt.Var, t:jt.Var) -> jt.Var:
        t = jt.unsqueeze(t, -1)
        #t = t.type(jt.float)
        t = self.pos_encoding(t, self.time_dim)
        x1:jt.Var = self.inc(x)
        x2:jt.Var = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3:jt.Var = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4:jt.Var = self.down3(x3, t)
        x4 = self.sa3(x4)
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output:jt.Var = self.outc(x)
        return output
