import torch
from torch import nn
import torch.nn.functional as F
from models.base import BaseModel
from models.stargan import load_stargan


class Model(BaseModel):
    def __init__(self, config, hidden_dim=256, base='vgg16', use_reg=True):
        super(Model, self).__init__(hidden_dim, base)

        self.out_layer = nn.Linear(hidden_dim, config.num_classes)
        self.trans = load_stargan(
            config.gan_path + 'stargan_last-G.ckpt')
        self.trans.eval()

        self.alpha = config.alpha

        self.use_reg = use_reg

    def forward(self, x, y, d=None):
        # run the input into the base model
        z = F.relu(self.base(x))
        print(f"orig_z {z}")
        # run the output of the base into the output layer to determine the class
        logits = self.out_layer(z)
        print(f"logits {logits}")
        # compute the loss based on the expected y value
        loss = F.cross_entropy(logits, y)

        # some kind of correction, not quite sure
        correct = (torch.argmax(logits, 1) == y).sum().float() / x.shape[0]
        # empty regurlization
        reg = loss.new_zeros([1])
        if self.training:
            if self.use_reg:
                with torch.no_grad():


                    # encode fitzpatrick label of current image
                    d_onehot = d.new_zeros([d.shape[0], 6])
                    d_onehot.scatter_(1, d[:, None], 1)

                    # generate random class to pick from
                    d_new = torch.randint(0, 6, (d.size(0),)).to(d.device)
                    d_new_onehot = d.new_zeros([d.shape[0], 6])
                    d_new_onehot.scatter_(1, d_new[:, None], 1)

                    # New generated image
                    x_new = self.trans(x, d_onehot, d_new_onehot)
                    # TODO - figure out dimentions
                    print(f"x_new.shape : {x_new.shape} x_new.dim():{x_new.dim()} x_new.size():{x_new.size()} x_new.dtype:{x_new.dtype}")
                    #print(x_new)

                z_new = F.relu(self.base(x_new))
                print(f"z_new.shape : {z_new.shape} z_new.dim():{z_new.dim()} z_new.size():{z_new.size()} z_new.dtype:{z_new.dtype}")
                reg = self.alpha * F.mse_loss(z_new, z)

        return loss, reg, correct