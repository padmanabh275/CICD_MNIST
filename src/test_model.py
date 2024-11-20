import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from model import create_model, count_parameters

def test_model_architecture():
    model = create_model()
    
    # Test 1: Check parameter count
    params = count_parameters(model)
    assert params < 25000, f"Too many parameters: {params}"
    
    # Test 2: Check input processing
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (1, 10), f"Invalid output shape: {output.shape}"

def test_model_predictions():
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    model = create_model()
    model.eval()
    
    # Get one test sample
    data, _ = next(iter(test_loader))
    
    # Test 3: Check prediction shape
    with torch.no_grad():
        predictions = model(data)
    assert predictions.shape == (1, 10), f"Invalid prediction shape: {predictions.shape}"
    
    # Test 4: Check probability sum
    prob_sum = torch.sum(predictions[0]).item()
    assert np.abs(prob_sum - 1.0) < 1e-6, f"Probabilities don't sum to 1: {prob_sum}"

def test_model_training():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32)
    
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Test 5: Check if model trains without errors
    model.train()
    data, target = next(iter(train_loader))
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    assert not torch.isnan(loss), "Training produced NaN loss"