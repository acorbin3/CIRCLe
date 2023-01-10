import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.base = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLu(),
            nn.MaxPool2d(2),
            # Second convolutional layer
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLu(),
            nn.MaxPool2d(2),
            # Third convolutional layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLu(),
        nn.MaxPool2d(2),
        # Flatten the feature map
        nn.Flatten(),
        nn.Linear(64 * 16 * 16, 512))

        self.output_layer = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 7)
        )

    def forward(self, x):
        base_output = self.base(x)
        logist = self.output_layer(base_output)
        return logist, base_output


# Initialize the model
model = CNN()
