import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import datetime
import os
from model import create_model, count_parameters
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

def show_batch(images, title):
    """Display a batch of images"""
    plt.figure(figsize=(10, 10))
    plt.title(title)
    grid_imgs = vutils.make_grid(images[:16], nrow=4, normalize=True)
    plt.imshow(grid_imgs.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(f'augmented_samples_{title}.png')
    plt.close()

def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define augmentations
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomApply([
            transforms.RandomRotation(15),  # Random rotation up to 15 degrees
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # Random translation up to 10%
                scale=(0.9, 1.1),      # Random scaling between 90-110%
                shear=10               # Random shearing up to 10 degrees
            )
        ], p=0.7),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))  # Random blur
        ], p=0.3),
        transforms.RandomErasing(p=0.2),  # Random erasing
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Visualize augmented samples
    # Get a batch of original images
    orig_dataset = datasets.MNIST('./data', train=True, download=True, 
                                transform=transforms.ToTensor())
    orig_loader = DataLoader(orig_dataset, batch_size=16, shuffle=True)
    orig_images, _ = next(iter(orig_loader))
    show_batch(orig_images, "Original Images")
    
    # Get a batch of augmented images
    aug_images, _ = next(iter(train_loader))
    show_batch(aug_images, "Augmented Images")
    
    # Create and train model
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.015, 
                                                  steps_per_epoch=len(train_loader), 
                                                  epochs=1,
                                                  pct_start=0.2)
    
    # Training
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Save model with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"model_mnist_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    
    return accuracy, count_parameters(model)

if __name__ == "__main__":
    accuracy, params = train()
    assert params < 25000, f"Model has {params} parameters, should be < 25000"
    assert accuracy > 0.95, f"Model accuracy {accuracy:.4f} is below 0.95" 