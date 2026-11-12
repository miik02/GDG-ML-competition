import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval
from cnn_common import IN_CH


def optimize_net(m: nn.Module, dummy_inp: torch.Tensor):
    """Fusiona Conv1d + BatchNorm1d en una sola capa y compila el grafo en C++ (JIT)."""
    m.eval()
    if hasattr(m, "f"):
        for blk in m.f:
            if isinstance(blk, nn.Sequential) and len(blk) >= 2:
                if isinstance(blk[0], nn.Conv1d) and isinstance(blk[1], nn.BatchNorm1d):
                    blk[0] = fuse_conv_bn_eval(blk[0], blk[1])
                    blk[1] = nn.Identity()
    with torch.inference_mode():
        return torch.jit.trace(m, dummy_inp)


class SpecNet(nn.Module):
    def __init__(self, w=48, in_ch=IN_CH, drop=0.25):
        super().__init__()
        def blk(i, o, k, s, d=1):
            pad = (k // 2) * d
            return nn.Sequential(
                nn.Conv1d(i, o, k, s, pad, dilation=d, bias=False),
                nn.BatchNorm1d(o),
                nn.SiLU(inplace=True)
            )
        self.f = nn.Sequential(
            blk(in_ch + 4, w, 7, 1),
            blk(w, w * 3 // 2, 5, 2),
            blk(w * 3 // 2, w * 2, 5, 2),
            blk(w * 2, w * 3, 3, 2, d=2),
            blk(w * 3, w * 4, 3, 2)
        )
        # 64 (FFT 16 bandas x 4 canales) + 16 (ACF 4 lags x 4 canales) + 16 (Corr 4x4) = 96 features
        self.spec_mlp = nn.Sequential(
            nn.BatchNorm1d(96),
            nn.Linear(96, w * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(drop)
        )
        self.head = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(w * 8 + w * 2, w * 2),
            nn.SiLU(inplace=True),
            nn.Linear(w * 2, 2)
        )

    def forward(self, x):
        B = x.shape[0]
        z = x[:, :4, :]
        nan_m = x[:, 4:8, :]

        valid = (nan_m[:, :, 1:] == 0) & (nan_m[:, :, :-1] == 0)
        dz = torch.where(valid, z[:, :, 1:] - z[:, :, :-1], torch.zeros_like(z[:, :, 1:]))
        dz = F.pad(dz, (1, 0))

        p = (torch.fft.rfft(z, dim=-1).abs()[:, :, 1:] ** 2)
        fft_bands = torch.log1p(p.reshape(B, 4, 16, 8).mean(dim=-1)).reshape(B, 64)

        var_z = (z ** 2).mean(dim=-1) + 1e-5
        acfs = torch.cat([(z[:, :, l:] * z[:, :, :-l]).mean(dim=-1) / var_z for l in (1, 2, 4, 8)], dim=1)

        zn = z / torch.sqrt(var_z.unsqueeze(-1))
        corr = (torch.bmm(zn, zn.transpose(1, 2)) / 256.0).reshape(B, 16)

        spec_feats = self.spec_mlp(torch.cat([fft_bands, acfs, corr], dim=1))
        h = self.f(torch.cat([x, dz], dim=1))
        pooled = torch.cat([h.mean(-1), h.amax(-1), spec_feats], dim=1)
        return self.head(pooled)