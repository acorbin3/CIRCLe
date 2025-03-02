import torch
import random
import argparse
import os, importlib
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary

from Metrics import Metrics
from models.cnn import CNN
from datasets.isic_2018.dataset import get_isic_2018_dataloaders, download_isic_2018_datasets, get_cached_dataframe
from util import AverageMeter
from datasets.fitzpatrick_17k_dataset.dataset import get_fitz_dataloaders
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='DG')
parser.add_argument('--dataset', type=str, default='FitzPatrick17k')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--num_classes', type=int, default=114)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--data_dir', type=str, default='../data/fitz17k/images/all/')
parser.add_argument('--gan_path', type=str, default='saved/stargan/')
parser.add_argument('--model', type=str, default='circle')
parser.add_argument('--base', type=str, default='vgg16')
parser.add_argument('--model_save_dir', type=str, default='saved/model/')
parser.add_argument('--use_reg_loss', type=bool, default=False)
flags = parser.parse_args()

if flags.dataset == 'FitzPatrick17k':
    flags.num_classes = 114
elif flags.dataset == "isic2018":
    flags.num_classes = 7  # ['NV', 'BKL', 'MEL', 'AKIEC', 'BCC', 'VASC', 'DF']

# print setup
print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))

device = 'cuda'
# set seed
random.seed(flags.seed)
np.random.seed(flags.seed)
torch.manual_seed(flags.seed)
torch.cuda.manual_seed(flags.seed)
torch.cuda.manual_seed_all(flags.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Data loader.
if flags.dataset == "FitzPatrick17k":
    train_loader, val_loader, _ = get_fitz_dataloaders(root=flags.data_dir,
                                                       holdout_mode='random_holdout',
                                                       batch_size=flags.batch_size,
                                                       shuffle=False,
                                                       partial_skin_types=[],
                                                       partial_ratio=1.0
                                                       )
elif flags.dataset == "isic2018":
    download_isic_2018_datasets()
    isic_df = get_cached_dataframe()
    train_loader, val_loader, test_loader, class_weights = get_isic_2018_dataloaders(isic_df)

# load models
if flags.model != "cnn":
    model = importlib.import_module('models.' + flags.model).Model(flags, flags.hidden_dim, flags.base).to(device)
else:
    model = CNN(8, .4)
    model.cuda()

summary(model, (3, 128, 128))
# TODO - consider changing the optimizer. Here are some notes:
"""
Stochastic Gradient Descent (SGD): This is a simple and widely-used optimizer that updates the model weights by computing the gradient of the loss function with respect to the weights, and moving the weights in the opposite direction. SGD is sensitive to the learning rate, which controls the step size of the weight updates.

Adam (Adaptive Moment Estimation): This is a popular optimizer that combines the ideas of momentum and learning rate decay. It keeps track of an exponentially weighted moving average of the gradients and the second moments of the gradients, and uses these to adapt the learning rate of each weight. Adam is generally a good default choice for many tasks.

RMSProp (Root Mean Square Propagation): This optimizer is similar to Adam, but it only keeps track of the second moments of the gradients, and uses these to scale the learning rate of each weight. RMSProp can be faster than Adam and is often used in deep learning models.

LBFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno): This is an optimizer that uses the L-BFGS algorithm, which is a quasi-Newton method that approximates the Hessian matrix of the loss function. LBFGS is more computationally expensive than other optimizers, but can be effective for training deep learning models.

There are many other optimizers available in PyTorch, such as Adagrad, Adadelta, and others
"""
# optim = torch.optim.SGD(model.parameters(), lr=flags.lr, weight_decay=flags.weight_decay, momentum=0.9)
optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)


def to_device(data):
    for i in range(len(data)):
        data[i] = data[i].to(device)
    return data


best_by_val = 0
best_val_acc = 0.0
best_val_loss = float('inf')
best_by_test = 0
best_test_loss = float('inf')

best_val_precision = 0
best_val_recall = 0

# TODO- update to use pytorch metrics: https://torchmetrics.readthedocs.io/en/stable/pages/overview.html
loss_function = nn.CrossEntropyLoss(weight=class_weights.float()).to(device)
metrics = Metrics()
val_metrics = Metrics()
total_loss_train, total_acc_train = [],[]
total_loss_val, total_acc_val = [],[]
for epoch in range(flags.epochs):
    print(
        f'Epoch {epoch}: Best val loss {best_val_loss:.4f}, Best val acc {best_val_acc:.4f}, best val recall {best_val_recall:.4f}, best val precision {best_val_precision:.4f}')
    model.train()
    i = 0
    for x, y, transformed_image in tqdm(train_loader, ncols=75, leave=False):

        optim.zero_grad()
        # insead of optim.zero_grad(), the below lines are supposed to be faster
        #for param in model.parameters():
        #    param.grad = None

        x, y, transformed_image = Variable(x).to(device), Variable(y).to(device), Variable(transformed_image).to(device)
        inputs, labels = x, y
        inputs_transformed, labels_transformed = transformed_image, y

        # pass inputs through the model
        outputs = model(inputs)

        # Compute metrics for main input image
        metrics.compute_metrics(outputs, labels, loss_function(outputs, labels))

        if False:
            logits_transformed, base_output_transformed = model(inputs_transformed)
            reg = flags.alpha * F.mse_loss(base_output_transformed, base_output)
            (metrics.loss + reg).backward()
            metrics.reg = reg
        else:
            metrics.reg = None
            metrics.loss.float().backward()

        metrics.update_metrics(x[0].shape[0])
        optim.step()
        i += 1
        if (i + 1) % 50 == 0:
            print()
            print(f'[epoch {epoch}], [iter {i + 1} / {len(train_loader)}], [train loss {metrics.loss_meter}], [train acc {metrics.accuracy_meter}]')
            total_loss_train.append(metrics.loss_meter.float())
            total_acc_train.append(metrics.accuracy_meter.float())

    print(
        f'\t>>> Training: Loss {metrics.loss_meter}, Reg {metrics.regularization_meter}, Acc {metrics.accuracy_meter}, precision: {metrics.precision_meter}, recall{metrics.recall_meter}')

    model.eval()
    with torch.no_grad():
        for x, y, transformed_image in tqdm(val_loader, ncols=75, leave=False):
            x, y, transformed_image = x.to(device), y.to(device), transformed_image.to(device)

            outputs = model(x)

            if False:
                # TODO - need to update model(once its working) to return features and logists
                logits_transformed, base_output_transformed = model(transformed_image)
                reg = flags.alpha * F.mse_loss(base_output_transformed, base_output)
                val_metrics.reg = reg

            val_metrics.compute_metrics(outputs, y, loss_function(outputs, y))
            val_metrics.update_metrics(x.shape[0])
        total_loss_val.append(val_metrics.loss_meter.float())
        total_acc_val.append(val_metrics.accuracy_meter.float())


    if flags.use_reg_loss:
        print(
            f'\t>>> Val.    : Loss {val_metrics.loss_meter}, Reg {val_metrics.regularization_meter}, Acc {val_metrics.accuracy_meter}, precision: {val_metrics.precision_meter}, recall{val_metrics.recall_meter}')
    else:
        print(
            f'\t>>> Val.    : Loss {val_metrics.loss_meter}, Acc {val_metrics.accuracy_meter}, precision: {val_metrics.precision_meter}, recall{val_metrics.recall_meter}')
    # Compute the confusion matrix
    # cm = confusion_matrix(y_true, y_pred)

    # Display the confusion matrix
    # print(cm)

    if val_metrics.loss_meter.float() < best_val_loss:
        best_val_loss = val_metrics.loss_meter.float()

    if val_metrics.recall_meter.float() > best_val_recall:
        best_val_recall = val_metrics.recall_meter.float()
    if val_metrics.precision_meter.float() > best_val_precision:
        best_val_precision = val_metrics.precision_meter.float()
    if val_metrics.accuracy_meter.float() > best_val_acc:
        best_val_acc = val_metrics.accuracy_meter.float()
        save_path = os.path.join(flags.model_save_dir, 'epoch{}_acc_{:.4f}.ckpt'.format(epoch, best_val_acc))
        torch.save(model.state_dict(), save_path)
        print('Saved model with highest acc ...')

    torch.cuda.empty_cache()
