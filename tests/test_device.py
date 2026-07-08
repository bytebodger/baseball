import pytest
import torch

from src.device import resolve_device


def test_resolve_device_passes_through_explicit_cpu_request(monkeypatch):
    """--device cpu is an explicit, intentional choice (small tests, local
    debugging) and must never raise, regardless of CUDA availability."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cpu") == torch.device("cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_returns_cuda_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == torch.device("cuda")


def test_resolve_device_raises_loudly_instead_of_silently_falling_back_to_cpu(monkeypatch):
    """This is the whole point: an unrequested absence of CUDA must be a hard
    error, not a silent downgrade to a much slower (and checkpoint-device-
    mismatched) CPU run."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")
