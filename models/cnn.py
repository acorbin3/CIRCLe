import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        # Pass input through first convolutional layer
        x = self.pool1(F.relu(self.conv1(x)))

        # Pass output through second convolutional layer
        x = self.pool2(F.relu(self.conv2(x)))

        # Pass output through third convolutional layer
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten the feature map
        x = self.flatten(x)
        x = self.fc1(x)
        base = self.dropout(x)
        logits = self.fc2(base)

        return logits, base


# Initialize the model
model = CNN()
