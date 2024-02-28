import torch
import torch.nn as nn

def linear_relu(dim_in, dim_out):
    return [nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True)]

class LinearNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearNet, self).__init__()
        self.input_size = input_size
        self.classifier = nn.Linear(self.input_size, num_classes)

    def forward(self, x):
        # Reshape the image into a D vector for the Linear class
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y

class FullyConnected(nn.Module):

    def __init__(self, input_size, num_classes):
        super(FullyConnected, self).__init__()
        self.classifier =  nn.Sequential(
            *linear_relu(input_size, 256),
            *linear_relu(256, 256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y

class FullyConnectedRegularized(nn.Module):

    def __init__(self, input_size, num_classes, l2_reg, dropout=False):
        super(FullyConnectedRegularized, self).__init__()
        layers = []
        self.l2_reg = l2_reg
        self.lin1 = nn.Linear(input_size, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, num_classes)

        if dropout:
            layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(input_size, 256))
        layers.append(nn.Linear(256, 256))
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(256, num_classes))

        self.classifier = nn.Sequential(*layers)

    def penalty(self):
        return self.l2_reg * (self.lin1.weight.norm(2) + self.lin2.weight.norm(2) + self.lin3.weight.norm(2))

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        y = self.lin3(x)
        return y
