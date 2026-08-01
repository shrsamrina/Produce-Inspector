import torch
import torch.nn as nn


class FruitCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 2
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 3
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # Adaptive Pooling - keeps a 7x7 spatial grid
            nn.AdaptiveAvgPool2d((7, 7))
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                128 * 7 * 7,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.4),


            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),


            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.classifier(x)

        return x
