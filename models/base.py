import torchvision.models as models
from torch import nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, hidden_dim=256, base='resnet50'):
        super(BaseModel, self).__init__()
        if base == 'alexnet':
            self.base = models.alexnet(pretrained=True)
            self.base.classifier[6] = nn.Linear(self.base.classifier[6].in_features, hidden_dim)
        elif base == 'resnet50':
            self.base = models.resnet50(pretrained=True)
            self.base.fc = nn.Linear(self.base.fc.in_features, hidden_dim)
        elif base == 'resnet18':
            self.base = models.resnet18(pretrained=True)
            self.base.fc = nn.Linear(self.base.fc.in_features, hidden_dim)

        elif base == "resnext101_32x8d":
            self.base = models.resnext101_32x8d(pretrained=True)
            self.base.fc = nn.Linear(in_features=2048, out_features=7)

        elif base == 'vgg16':
            self.base = models.vgg16(pretrained=True)
            self.base.classifier[6] = nn.Linear(self.base.classifier[6].in_features, hidden_dim)
        elif base == 'densenet121':
            self.base = models.densenet121(pretrained=True)
            self.base.classifier = nn.Linear(in_features=self.base.classifier.in_features, out_features=hidden_dim)
        elif base == 'mobilenetv2':
            self.base = models.mobilenet_v2(pretrained=True)
            self.base.classifier[1] = nn.Linear(in_features=self.base.classifier[1].in_features,
                                                out_features=hidden_dim)
            # Freeze all the layers
            for param in self.base.parameters():
                param.requires_grad = False

            # Unfreeze the last layer
            for param in self.base.classifier[1].parameters():
                param.requires_grad = True
        elif base == 'mobilenetv3l':
            self.base = models.mobilenet_v3_large(pretrained=True)
            self.base.classifier[3] = nn.Linear(in_features=self.base.classifier[3].in_features,
                                                out_features=hidden_dim)
        elif base == "alexnet":
            self.base = models.alexnet(pretrained=True)
            self.base.classifier[6] = nn.Linear(self.base.classifier[6], hidden_dim)
        elif base == "resnext50_32x4d":
            self.base = models.resnext50_32x4d(pretrained=True)
            self.base.fc = nn.Linear(in_features=self.base.fc.in_features, out_features=hidden_dim)
        elif base == "squeezenet1_0":
            self.base = models.squeezenet1_0(pretrained=True)
            self.base.classifier[1] = nn.Conv2d(
                in_channels=self.base.classifier[1].in_channels,
                out_channels=hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )
            self.base.classifier[3] = nn.Linear(self.base.classifier[3], hidden_dim)
        elif base == "shufflenet_v2_x1_0":
            self.base = models.shufflenet_v2_x1_0(pretrained=True)
            self.base.fc = nn.Linear(in_features=self.base.fc.in_features, out_features=hidden_dim)
        elif base == "wide_resnet50_2":
            self.base = models.wide_resnet50_2(pretrained=True)
            self.base.fc = nn.Linear(in_features=self.base.fc.in_features, out_features=hidden_dim)
        elif base == "mnasnet":
            self.base = models.mnasnet1_0(pretrained=True)
            self.base.classifier[-1] = nn.Linear(in_features=self.base.classifier[-1].in_features,
                                                 out_features=hidden_dim)