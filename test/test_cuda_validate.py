import torch


def test_cuda_validate():
    assert torch.cuda.is_available() is True


def test_cuda_validate_fail():
    assert torch.cuda.is_available() is not None


def test_cuda_get_device_name():
    device_name = torch.cuda.get_device_name()
    assert device_name == 'NVIDIA GeForce RTX 4060'
    assert device_name != 'not found'
