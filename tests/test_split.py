import torch
from torch import nn

from feature_probe.split import SplitClassifier


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU())
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.stem(x).mean((2, 3)))


def test_split_round_trip():
    torch.manual_seed(0)
    model = Network().eval()
    adapter = SplitClassifier(model)
    images = torch.rand(3, 3, 4, 4)
    errors = adapter.assert_split_consistency(images, ["stem"])
    assert errors["stem"] == 0
