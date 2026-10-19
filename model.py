import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def mlp(a, b, c, dropout=0.2):
    return nn.Sequential(nn.Linear(a, b), nn.SiLU(), nn.Dropout(dropout), nn.Linear(b, c))


class Residual(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv3d(cout, cout, 3, 1, 1, bias=False), nn.GroupNorm(8, cout)
        )
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv3d(cin, cout, 1, stride, bias=False), nn.GroupNorm(8, cout)
        )

    def forward(self, x):
        return F.silu(self.body(x) + self.skip(x))


class Encoder(nn.Module):
    def __init__(self, tokens=False):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, 32, 3, 2, 1, bias=False), nn.GroupNorm(8, 32), nn.SiLU())
        self.blocks = nn.Sequential(Residual(32, 32, 1), Residual(32, 64, 2),
                                    Residual(64, 128, 2), Residual(128, 256, 2))
        self.global_projection = nn.Linear(256, 128)
        self.token_projection = nn.Conv3d(256, 128, 1) if tokens else None

    def forward(self, x):
        x = self.blocks(self.stem(x))
        z = self.global_projection(x.mean((2, 3, 4)))
        if self.token_projection is None:
            return z
        tokens = F.adaptive_avg_pool3d(self.token_projection(x), (6, 6, 6)).flatten(2).transpose(1, 2)
        return z, tokens


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(128, 4, dropout=0.2, batch_first=True)
        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(128)
        self.ff = mlp(128, 256, 128)

    def forward(self, query, context):
        x = self.norm1(query + self.attention(query, context, context, need_weights=False)[0])
        return self.norm2(x + self.ff(x))


class AMMRSurv(nn.Module):
    def __init__(self, checkpoint_encoders=True):
        super().__init__()
        self.checkpoint_encoders = checkpoint_encoders
        self.pet = Encoder()
        self.ct = Encoder(tokens=True)
        self.timing = mlp(4, 32, 16)
        self.event = nn.Embedding(3, 8)
        self.anatomy = nn.Embedding(4, 8)
        self.pet_response = mlp(536, 256, 128)
        self.prototypes = nn.Parameter(torch.randn(4, 128))
        self.pet_combined = mlp(260, 128, 128)
        self.ct_attention = nn.MultiheadAttention(128, 4, dropout=0.2, batch_first=True)
        self.ct_response = mlp(536, 256, 128)
        self.pet_projection = nn.Linear(128, 128)
        self.ct_projection = nn.Linear(128, 128)
        self.gate = mlp(408, 128, 128)
        self.fusion = mlp(800, 256, 128)
        self.set_blocks = nn.ModuleList([AttentionBlock(), AttentionBlock()])
        self.pool = AttentionBlock()
        self.seed = nn.Parameter(torch.empty(1, 1, 128))
        self.survival = mlp(133, 64, 5)
        self.metabolic = mlp(128, 64, 4)
        self.apply(self.initialize)
        nn.init.xavier_uniform_(self.seed)
        with torch.no_grad():
            self.prototypes.copy_(F.normalize(self.prototypes, dim=-1))

    @staticmethod
    def initialize(module):
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, (nn.Conv3d, nn.Linear)) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def encode(self, encoder, x):
        if self.training and self.checkpoint_encoders:
            return checkpoint(encoder, x, use_reentrant=False)
        return encoder(x)

    def forward(self, patient, warmup=False):
        n = len(patient["event"])
        t = self.timing(patient["timing"].view(1, 4)).expand(n, -1)
        e = self.event(patient["event"])
        v = self.anatomy(patient["anatomy"])
        responses, lesions, assignments = [], [], []
        for i in range(n):
            zp = self.encode(self.pet, patient["pet"][i])
            zc, tokens = self.encode(self.ct, patient["ct"][i])
            ti, ei, vi = t[i:i + 1], e[i:i + 1], v[i:i + 1]
            r = self.pet_response(torch.cat((zp[0:1], zp[1:2], zp[1:2] - zp[0:1],
                                           zp[0:1] * zp[1:2], ti, ei), -1))
            if warmup:
                a = r.new_full((1, 4), 0.25)
                reconstruction = torch.zeros_like(r)
            else:
                similarity = F.normalize(r.float(), dim=-1) @ F.normalize(self.prototypes.float(), dim=-1).T
                a = torch.softmax(similarity / 0.1, dim=-1).to(r.dtype)
                reconstruction = a @ self.prototypes.to(r.dtype)
            up = self.pet_combined(torch.cat((r, reconstruction, a), -1))
            attended = self.ct_attention(tokens[1:2], tokens[0:1], tokens[0:1], need_weights=False)[0].mean(1)
            uc = self.ct_response(torch.cat((attended, zc[0:1], zc[1:2], zc[1:2] - zc[0:1], ti, ei), -1))
            p, c = self.pet_projection(up), self.ct_projection(uc)
            g = torch.sigmoid(self.gate(torch.cat((p, c, (p - c).abs(), ti, ei), -1)))
            fused = g * p + (1 - g) * c
            lesion = self.fusion(torch.cat((p, c, p * c, p - c, (p - c).abs(), fused, vi, ei, ti), -1))
            responses.append(r)
            assignments.append(a)
            lesions.append(lesion)
        r, a = torch.cat(responses), torch.cat(assignments)
        x = torch.cat(lesions).unsqueeze(0)
        for block in self.set_blocks:
            x = block(x, x)
        s = self.pool(self.seed, x).squeeze(1)
        q = a.float().mean(0, keepdim=True)
        entropy = -(q * torch.log(q + 1e-8)).sum(-1, keepdim=True)
        logits = self.survival(torch.cat((s, q.to(s.dtype), entropy.to(s.dtype)), -1)).float().squeeze(0)
        return {"logits": logits, "response": r.float(), "assignments": a.float(),
                "metabolic": self.metabolic(r).float()}


def survival_loss(logits, time, event, boundaries):
    boundaries = torch.as_tensor(boundaries, device=logits.device, dtype=torch.float32)
    logits = logits.float()
    if event and time <= float(boundaries[-1]):
        index = int(torch.searchsorted(boundaries, boundaries.new_tensor(time), right=False))
        return F.softplus(logits[:index]).sum() + F.softplus(-logits[index])
    return F.softplus(logits[boundaries <= time]).sum()


def auxiliary_loss(output, patient):
    valid = patient["target_valid"]
    if not bool(valid.any()):
        return output["metabolic"].sum() * 0, False
    loss = F.smooth_l1_loss(output["metabolic"], patient["targets"], beta=1.0, reduction="none")
    counts = valid.sum(-1)
    per_lesion = (loss * valid).sum(-1) / counts.clamp_min(1)
    return per_lesion[counts > 0].mean(), True


def prototype_loss(output, prototypes):
    r = F.normalize(output["response"], dim=-1)
    mu = F.normalize(prototypes.float(), dim=-1)
    compactness = (output["assignments"] * (1 - r @ mu.T)).sum(-1).mean()
    off_diagonal = ~torch.eye(4, dtype=torch.bool, device=mu.device)
    diversity = F.relu((mu @ mu.T)[off_diagonal] - 0.2).square().mean()
    return compactness + diversity


def augment(patient, geometry_probability=0.5, ct_noise_probability=0.2):
    for i in range(len(patient["event"])):
        if torch.rand(()) < geometry_probability:
            angles = (torch.rand(3) * 20 - 10) * math.pi / 180
            x, y, z = angles.tolist()
            rx = torch.tensor([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
            ry = torch.tensor([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
            rz = torch.tensor([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
            scale = float(torch.rand(()) * 0.1 + 0.95)
            rotation = rz @ ry @ rx
            translation = (torch.rand(3) * 10 - 5) / 96
            inverse = rotation.T / scale
            theta = torch.cat((inverse, (-inverse @ translation).unsqueeze(1)), 1)
            for key, padding in (("pet", 0.0), ("ct", -1.0)):
                volume = patient[key][i]
                transform = theta.to(volume).unsqueeze(0).expand(2, -1, -1)
                grid = F.affine_grid(transform, volume.shape, align_corners=False)
                patient[key][i] = F.grid_sample(volume - padding, grid, mode="bilinear",
                                               padding_mode="zeros", align_corners=False) + padding
        for timepoint in range(2):
            if torch.rand(()) < ct_noise_probability:
                volume = patient["ct"][i, timepoint]
                patient["ct"][i, timepoint] = (volume + torch.randn_like(volume) * 0.01).clamp(-1, 1)
    return patient
