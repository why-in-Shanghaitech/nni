"""
HPO Quickstart with PyTorch
===========================
This script is for downloading datset on the login node of a slurm system.
"""

from torchvision import datasets
from torchvision.transforms import ToTensor
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
