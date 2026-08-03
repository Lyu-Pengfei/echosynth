"""CompactCNN, transforms, and two-stage composition tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from echosynth.classify.models import CompactCNN
from echosynth.classify.transforms import EvalTransform, TrainTransform, build_transform
from echosynth.classify.two_step import compose_two_step


def test_compactcnn_shape() -> None:
    model = CompactCNN(channels=(4, 8, 16, 32), dropout=0.1)
    assert model(torch.randn(3, 1, 112, 112)).shape == (3, 2)


def test_eval_transform_is_structurally_nonaugmenting() -> None:
    image = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    transform = build_transform(False, 32)
    assert isinstance(transform, EvalTransform)
    assert not isinstance(transform, TrainTransform)
    assert torch.equal(transform(image), transform(image))


@pytest.mark.parametrize("mode", ["product", "stage2_only", "logit_sum"])
def test_two_step_hand_computation(mode: str) -> None:
    first = np.array([0.2, 0.8])
    second = np.array([0.5, 0.75])
    result = compose_two_step(first, second, mode)
    if mode == "product":
        expected = np.array([0.1, 0.6])
    elif mode == "stage2_only":
        expected = second
    else:
        expected = np.array([0.2, 12 / 13])
    assert result.p_ca == pytest.approx(expected)


@pytest.mark.slow
def test_compactcnn_can_overfit_twenty_samples() -> None:
    torch.manual_seed(3)
    images = torch.zeros(20, 1, 32, 32)
    labels = torch.tensor([0] * 10 + [1] * 10)
    images[:10, :, 8:24, 8:24] = 0.2
    images[10:, :, 8:24, 8:24] = 1
    model = CompactCNN(channels=(2, 4, 8, 16), dropout=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(30):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
    probabilities = torch.softmax(model(images), dim=1)[:, 1].detach().numpy()
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(labels.numpy(), probabilities) > 0.95
