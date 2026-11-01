"""Face-image judges used by the line: the defect classifier, and classifier + good-only anomaly detector."""
import numpy as np
import torch
import torch.nn as nn
import torchvision

CLASSES = ["ok", "scratch", "stain", "chip"]

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FaceJudge:
    def __init__(self, weights="models/face_cls.pt", device="cpu"):
        self.dev = torch.device(device)
        m = torchvision.models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
        m.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model = m.to(self.dev).eval()
        self.last = None

    def probs(self, img):
        x = torch.from_numpy(img[None]).permute(0, 3, 1, 2).float() / 255.0
        x = ((x - MEAN) / STD).to(self.dev)
        with torch.no_grad():
            return torch.softmax(self.model(x), 1)[0].cpu().numpy()

    def __call__(self, cam, img, part):
        p = self.probs(img)
        self.last = dict(cam=cam, img=img, probs=p, part=part)
        return float(1.0 - p[0])          # probability that this face is defective


class CombinedJudge:
    """Classifier for the known defect types, plus the good-only detector for anything else unusual.
    A face is NG when either says so; `last["by"]` tells which one decided."""

    def __init__(self, weights="models/face_cls.pt", anomaly="models/anomaly.pt", device="cpu"):
        from .anomaly import AnomalyJudge
        self.cls, self.anom = FaceJudge(weights, device), AnomalyJudge(anomaly, device)
        self.last = None

    def __call__(self, cam, img, part):
        p1, p2 = self.cls(cam, img, part), self.anom(cam, img, part)
        self.last = dict(self.cls.last, by="anomaly" if p2 > max(p1, 0.5) else "classifier", p_anomaly=p2)
        return max(p1, p2)
