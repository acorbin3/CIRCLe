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

        # Fully connected layer
        self.fc = nn.Linear(64 * 8 * 8, 512)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # Output layer
        self.out = nn.Linear(512, 7)

    def forward(self, x):
        # Pass input through first convolutional layer
        x = self.pool1(F.relu(self.conv1(x)))

        # Pass output through second convolutional layer
        x = self.pool2(F.relu(self.conv2(x)))

        # Pass output through third convolutional layer
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten the feature map
        x = x.view(-1, 64 * 8 * 8)

        # Pass output through the fully connected layer
        base = F.relu(self.fc(x))

        # Apply dropout
        x = self.dropout(base)

        # Pass output through the output layer
        x = self.out(x)

        return x, base


# Initialize the model
model = CNN()
