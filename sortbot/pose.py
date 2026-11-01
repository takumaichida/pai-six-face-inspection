"""Part pose from the camera-1 image: where the part is under the camera and how it is turned.

A ResNet-18 regresses (dx, dy, cos 4θ, sin 4θ): the offset of the part centre from the point under the camera
and its yaw modulo 90 deg (a square part looks the same every quarter turn). The line uses it to grab parts
that pass by at unknown positions and angles.
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SCALE = np.array([0.02, 0.04, 1.0, 1.0], np.float32)   # dx, dy [m] are normalised by these


def build_model(pretrained=True):
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=w)
    m.fc = nn.Linear(m.fc.in_features, 4)
    return m


def to_tensor(imgs):
    x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN) / STD


def decode(out):
    """Network output (N, 4) -> dx, dy [m], yaw [rad] in (-pi/4, pi/4]."""
    out = np.asarray(out) * SCALE
    yaw = np.arctan2(out[:, 3], out[:, 2]) / 4
    return out[:, 0], out[:, 1], yaw


def bright_fraction(img):
    """Share of bright pixels: a part (bright aluminium) under the camera, as a simple presence sensor."""
    return float((img.mean(2) > 170).mean())


class PoseEstimator:
    def __init__(self, weights="models/pose.pt", device="cpu"):
        self.dev = torch.device(device)
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model = self.model.to(self.dev).eval()

    def __call__(self, img):
        with torch.no_grad():
            out = self.model(to_tensor(img[None]).to(self.dev)).cpu().numpy()
        dx, dy, yaw = decode(out)
        return float(dx[0]), float(dy[0]), float(yaw[0])
