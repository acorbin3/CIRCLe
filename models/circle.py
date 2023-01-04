import torch
from torch import nn
import torch.nn.functional as F
from models.base import BaseModel
from models.stargan import load_stargan
from skin_transformer.skin_transformer import transform_image
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from skimage import util
import cv2
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix


def debug_it(name, it, print_object=False):
    print(f"{name}.shape : {it.shape} {name}.dtype:{it.dtype}")
    if print_object:
        print(it)
    print()


class Model(BaseModel):
    def __init__(self, config, hidden_dim=256, base='vgg16', use_reg=True):
        super(Model, self).__init__(hidden_dim, base)

        #self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        #self.dropout = nn.Dropout(p=0.5)

        self.out_layer = nn.Linear(hidden_dim, config.num_classes)
        # self.trans = load_stargan(
        #    config.gan_path + 'stargan_last-G.ckpt')
        # self.trans.eval()

        self.alpha = config.alpha

        self.use_reg = use_reg

    def custom_transformer(self, input_image_batches, input_mask_batches, input_image_ita_batches):
        """
        Method wrapping the skin transformer since that method doesnt support batches.
        @param input_image_batches:
        @param input_mask_batches:
        @return: array of modified images
        """
        transformed_image_batches = []
        for batch in range(len(input_image_batches)):
            input_image = input_image_batches[batch][0]
            input_image_ita = input_image_ita_batches[batch][0]
            # print(f"input_image: {input_image.dtype}")
            # print(f"input_image: {input_image.shape}")
            to_pil = transforms.ToPILImage()
            input_image = to_pil(input_image.type(torch.float32))

            # input_image.save("test_after_batch_conversion1.png")

            # image_array = np.array(input_image)
            # rgb_image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            # input_image2 = Image.fromarray(rgb_image_array)
            # input_image2.save("test_after_batch_conversion2.png")

            input_mask = input_mask_batches[batch][0]
            input_mask = to_pil(input_mask.type(torch.float32))
            # input_mask.save("test_batch_mask.png")

            transformed_image = transform_image(input_image, input_mask, image_ita=input_image_ita, verbose=False)
            # print(f"transformed_image: {transformed_image.shape}")
            # pil_image = Image.fromarray(util.img_as_ubyte(transformed_image))
            to_tensor = transforms.ToTensor()
            transformed_image_tensor = to_tensor(transformed_image)
            # todo - this is a hack for cuda, look a ways to fix this
            transformed_image_tensor = transformed_image_tensor.to("cuda")
            # transformed_image_tensor = transformed_image_tensor.permute(1, 2, 0)
            # print(f"transformed_image_tensor: {transformed_image_tensor.shape}")
            # print(f"transformed_image_tensor: {transformed_image_tensor.dtype}\n")
            transformed_image_batches.append(transformed_image_tensor)

        return transformed_image_batches

    def forward(self, input_image, expected_classification, d=None, input_mask=None, input_image_ita=None):
        debugging = False
        # run the input into the base model
        z = F.relu(self.base(input_image))

        #TODO - evaluate if we should add these extra 2 layers to help prevent overfitting to the traning dataset
        #z = self.fc1(z)
        #z = self.dropout(z)

        if debugging: print("################")
        if debugging: debug_it("orig_z", z)
        # run the output of the base into the output layer to determine the class
        logits = self.out_layer(z)
        # logits basically returns a list of probilities of each class.
        # we only care about the highest probility thus we would do this "torch.argmax(logits, 1)"
        # to assocate that the model's highest probility will be the "predicted" class
        if debugging: debug_it("logits", logits, True)
        if debugging: debug_it("y", expected_classification, True)

        # compute the loss based on the expected y value
        loss = F.cross_entropy(logits, expected_classification)
        if debugging: debug_it("loss", loss, True)

        # This is calculating accuracy
        correct = (torch.argmax(logits, 1) == expected_classification).sum().float() / input_image.shape[0]
        if debugging: debug_it("correct", correct, True)

        # Calculate precision and recall
        true_labels = []
        predicted_labels = []
        predictions = torch.argmax(logits, 1).cpu().numpy()
        labels = expected_classification.cpu().numpy()

        if debugging: debug_it("predictions", predictions, True)
        if debugging: debug_it("labels", labels, True)
        # Compute the micro-average precision and recall
        cm = confusion_matrix(labels, predictions)
        precision = cm.diagonal().sum() / cm.sum(axis=0).sum()
        recall = cm.diagonal().sum() / cm.sum(axis=1).sum()

        # empty regularization
        reg = loss.new_zeros([1])
        if debugging: debug_it("reg", reg, True)
        if self.training:
            if self.use_reg:
                with torch.no_grad():
                    if debugging: debug_it("input_image", input_image, False)
                    output_skin_transformer = self.custom_transformer(np.array_split(input_image, len(input_image)),
                                                                      np.array_split(input_mask, len(input_mask)),
                                                                      np.array_split(input_image_ita, len(input_image_ita)))
                    x_new = torch.stack(output_skin_transformer)

                    if debugging: debug_it("x_new", x_new, False)

                z_new = F.relu(self.base(x_new))

                reg = self.alpha * F.mse_loss(z_new, z)
                if debugging: debug_it("reg", reg, True)
        if debugging: print("------------------")
        return loss, reg, correct, precision, recall
