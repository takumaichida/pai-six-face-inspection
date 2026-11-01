"""Defect detection learned from good parts only (a small PatchCore-style detector).

Real lines rarely have enough defect images, and new kinds of defects appear. This detector keeps a memory of
local image features of good faces for each camera and scores a new face by how far its most unusual patch is
from that memory. No defect image is used for training.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PatchFeatures(torch.nn.Module):
    """ImageNet ResNet-18, layer2 + layer3 features on a 16x16 grid (frozen, no training)."""

    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.l2, self.l3 = m.layer2, m.layer3
        self.eval()

    @torch.no_grad()
    def forward(self, imgs):
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0
        x = ((x - MEAN) / STD).to(next(self.parameters()).device)
        f2 = self.l2(self.stem(x))
        f3 = F.interpolate(self.l3(f2), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = F.avg_pool2d(torch.cat([f2, f3], 1), 3, 1, 1)          # local neighbourhood
        return f.permute(0, 2, 3, 1).reshape(len(imgs), -1, f.shape[1])   # (B, patches, C)


class AnomalyDetector:
    def __init__(self, device="cpu"):
        self.dev = torch.device(device)
        self.feat = PatchFeatures().to(self.dev)
        self.bank, self.th = {}, {}

    def features(self, imgs, bs=64):
        return torch.cat([self.feat(imgs[k:k + bs]) for k in range(0, len(imgs), bs)])

    def fit(self, view, good_imgs, n_keep=30000, seed=0):
        f = self.features(good_imgs).reshape(-1, self.feat.l3[-1].conv2.out_channels + 128)
        idx = torch.from_numpy(np.random.default_rng(seed).choice(len(f), min(n_keep, len(f)), replace=False))
        self.bank[view] = f[idx.to(f.device)].contiguous()

    def score(self, view, imgs, chunk=4096):
        """Per image: distance of its most unusual patch to the nearest good patch."""
        f = self.features(imgs)
        b, p, c = f.shape
        flat = f.reshape(-1, c)
        d = torch.cat([torch.cdist(flat[k:k + chunk], self.bank[view]).min(1).values
                       for k in range(0, len(flat), chunk)])
        return d.reshape(b, p).max(1).values.cpu().numpy()

    def calibrate(self, view, good_imgs, q=0.995):
        """Threshold = a high quantile of scores on good faces not used for the memory."""
        self.th[view] = float(np.quantile(self.score(view, good_imgs), q))

    def save(self, path):
        torch.save(dict(bank={k: v.cpu() for k, v in self.bank.items()}, th=self.th), path)

    def load(self, path):
        z = torch.load(path, map_location="cpu")
        self.bank = {k: v.to(self.dev) for k, v in z["bank"].items()}
        self.th = z["th"]
        return self


class AnomalyJudge:
    """Judge interface for the line: returns a defect 'probability' with 0.5 at the calibrated threshold."""

    def __init__(self, path="models/anomaly.pt", device="cpu"):
        self.det = AnomalyDetector(device).load(path)
        self.last = None

    def __call__(self, cam, img, part):
        s = float(self.det.score(cam, img[None])[0])
        p = float(1 / (1 + np.exp(-(s / self.det.th[cam] - 1) * 12)))
        self.last = dict(cam=cam, img=img, probs=np.array([1 - p, p, 0, 0]), part=part)
        return p
