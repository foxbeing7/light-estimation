import torch
import torch.nn as nn
import torchvision.models as models
from cbam import *

class LightSourceEstimationModel(nn.Module):
    def __init__(self, num_outputs):
        super(LightSourceEstimationModel, self).__init__()
        # self.resnet18 = ResNet18(num_outputs=4)
        self.resnet18 = models.resnet34(pretrained=False)
        num_features = self.resnet18.fc.in_features
        # self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.resnet18.fc = nn.Identity()  # 移除原始的全连接层

        self.fc_layers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        features = self.resnet18(x)
        predictions = self.fc_layers(features)
        return predictions
        # return features


class ResNet18(nn.Module):
    def __init__(self,  num_outputs):
        super(ResNet18, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=3, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.cbam0 = CBAM(channels=64)
        # self.cbam1 = CBAM(channels=64)
        # self.cbam2 = CBAM(channels=128)
        # self.cbam3 = CBAM(channels=256)
        # self.cbam4 = CBAM(channels=512)
        self.layer1 = self.resblock(64, 64, num_blocks=2, stride=1)
        self.layer2 = self.resblock(64, 128, num_blocks=2, stride=2)
        self.layer3 = self.resblock(128, 256, num_blocks=2, stride=2)
        self.layer4 = self.resblock(256, 512, num_blocks=2, stride=2)
        # self.cbam = CBAM()
        # Global average pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512,num_outputs)
        )

    def resblock(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(1, num_blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.cbam0(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.cbam1(x)
        x = self.layer2(x)
        # x = self.cbam2(x)
        x = self.layer3(x)
        # x = self.cbam3(x)
        x = self.layer4(x)
        # x = self.cbam4(x)
        x = self.avgpool(x)
        predictions = self.fc_layers(x)
        return predictions