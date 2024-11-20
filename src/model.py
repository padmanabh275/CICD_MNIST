import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(8 * 7 * 7, 24),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(24, 10)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 8 * 7 * 7)
        x = self.fc_layers(x)
        return x

def create_model():
    model = MNISTModel()
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 