import torch
from torch import nn
import torch.nn.functional as F
from models.base import BaseModel


class Model(BaseModel):
    def __init__(self, config, hidden_dim=256, base='vgg16'):
        super(Model, self).__init__(hidden_dim, base)

        #self.base_output = nn.Sequential(
         #   self.base(),
          #  nn.ReLu(),
        #)

        self.output_layer = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor):
        #base_output = F.relu(self.base(x))
        #logist = self.output_layer(base_output)
        return self.base(x)

