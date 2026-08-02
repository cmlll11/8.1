import torch

from feature_probe.artifacts import load_classifier
from feature_probe.models import CifarResNet18
from feature_probe.split import SplitClassifier


def test_cifar_resnet_split_boundaries():
    model = CifarResNet18().eval()
    images = torch.rand(2, 3, 32, 32)
    errors = SplitClassifier(model).assert_split_consistency(images, ["stem", "layer1.0", "layer1", "layer2"])
    assert max(errors.values()) == 0


def test_self_contained_checkpoint_round_trip(tmp_path):
    model = CifarResNet18().eval()
    path = tmp_path / "model.pt"
    torch.save(
        {
            "metadata": {"protocol": "MDL-FEATURE-v1", "architecture": "cifar_resnet18"},
            "model": model.state_dict(),
            "epoch": 1,
        },
        path,
    )
    restored = load_classifier(path)
    images = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(model(images), restored(images))
