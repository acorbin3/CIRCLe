import torch
from torch import nn
import torch.nn.functional as F
from models.base import BaseModel
from models.stargan import load_stargan
from skin_transformer.skin_transformer import transform_image


def debug_it(name, it, print_object=False):
    print(f"{name}.shape : {it.shape} {name}.dim():{it.dim()} {name}.size():{it.size()} {name}.dtype:{it.dtype}")
    if print_object:
        print(it)
    print()


class Model(BaseModel):
    def __init__(self, config, hidden_dim=256, base='vgg16', use_reg=True):
        super(Model, self).__init__(hidden_dim, base)

        self.out_layer = nn.Linear(hidden_dim, config.num_classes)
        # self.trans = load_stargan(
        #    config.gan_path + 'stargan_last-G.ckpt')
        # self.trans.eval()

        self.alpha = config.alpha

        self.use_reg = use_reg

    def custom_transformer(img):
        pass

    def forward(self, input_image, expected_classification, d=None, input_mask=None):
        debugging = False
        # run the input into the base model
        z = F.relu(self.base(input_image))
        if debugging: print("################")
        if debugging: debug_it("orig_z", z)
        # run the output of the base into the output layer to determine the class
        logits = self.out_layer(z)
        if debugging: debug_it("logits", logits, True)
        if debugging: debug_it("y", expected_classification, True)

        # compute the loss based on the expected y value
        loss = F.cross_entropy(logits, expected_classification)
        if debugging: debug_it("loss", loss, True)

        # some kind of correction, not quite sure
        correct = (torch.argmax(logits, 1) == expected_classification).sum().float() / input_image.shape[0]
        if debugging: debug_it("correct", correct, True)

        # empty regurlization
        reg = loss.new_zeros([1])
        if debugging: debug_it("reg", reg, True)
        if self.training:
            if self.use_reg:
                with torch.no_grad():
                    # encode fitzpatrick label of current image
                    d_onehot = d.new_zeros([d.shape[0], 6])
                    d_onehot.scatter_(1, d[:, None], 1)

                    # generate random fitzpatrick skin type class to transform
                    #d_new = torch.randint(0, 6, (d.size(0),)).to(d.device)
                    #d_new_onehot = d.new_zeros([d.shape[0], 6])
                    #d_new_onehot.scatter_(1, d_new[:, None], 1)
                    # TODO - update to new image transformer
                    x_new = transform_image(input_image, input_mask)
                    #x_new = input_image
                    # New generated image
                    # x_new = self.trans(x, d_onehot, d_new_onehot)

                    # x_new = self.custom_transformer(x)

                    if debugging: debug_it("x_new", x_new, False)

                    # print(x_new)

                z_new = F.relu(self.base(x_new))

                reg = self.alpha * F.mse_loss(z_new, z)
                if debugging: debug_it("reg", reg, True)
        if debugging: print("------------------")
        return loss, reg, correct
