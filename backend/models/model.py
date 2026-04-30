import torch.nn as nn
import torchvision.models as models


class DeepfakeDetector(nn.Module):

    def __init__(self):
        super(DeepfakeDetector, self).__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)

        # Replace final layer
        self.model.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.model(x)