import torch
import torch.nn as nn

class SimpleConv(nn.Module):

    def __init__(self, in_channels, num_classes, dropout=False):
        super(SimpleConv, self).__init__()
        layers = []

        layers.append(nn.Linear(in_channels, in_channels))

        layers.append(nn.Conv2d(in_channels, 16, 5, stride=1, padding=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2, stride=2))

        layers.append(nn.Conv2d(16, 32, 5, stride=1, padding=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2, stride=2))

        layers.append(nn.Conv2d(32, 64, 5, stride=1, padding=2))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2, stride=2))

        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(64, 128))
        layers.append(nn.ReLU(inplace=True))

        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(128, 64))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(64, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y
