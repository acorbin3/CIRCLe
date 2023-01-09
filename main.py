import torch
import random
import argparse
import os, importlib
from tqdm import tqdm
import numpy as np
from torch import nn
import torch.nn.functional as F

from organize_data.isic_2018.dataset import get_isic_2018_dataloaders, download_isic_2018_datasets, get_cached_dataframe
from util import AverageMeter
from organize_data.fitzpatrick_17k_dataset.dataset import get_fitz_dataloaders
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
parser.add_argument('--use_reg_loss', type=bool, default=True)
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
    train_loader, val_loader, test_loader = get_isic_2018_dataloaders(isic_df)

# load models
model = importlib.import_module('models.' + flags.model).Model(flags, flags.hidden_dim, flags.base).to(device)

#TODO - consider changing the optimizer. Here are some notes:
"""
Stochastic Gradient Descent (SGD): This is a simple and widely-used optimizer that updates the model weights by computing the gradient of the loss function with respect to the weights, and moving the weights in the opposite direction. SGD is sensitive to the learning rate, which controls the step size of the weight updates.

Adam (Adaptive Moment Estimation): This is a popular optimizer that combines the ideas of momentum and learning rate decay. It keeps track of an exponentially weighted moving average of the gradients and the second moments of the gradients, and uses these to adapt the learning rate of each weight. Adam is generally a good default choice for many tasks.

RMSProp (Root Mean Square Propagation): This optimizer is similar to Adam, but it only keeps track of the second moments of the gradients, and uses these to scale the learning rate of each weight. RMSProp can be faster than Adam and is often used in deep learning models.

LBFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno): This is an optimizer that uses the L-BFGS algorithm, which is a quasi-Newton method that approximates the Hessian matrix of the loss function. LBFGS is more computationally expensive than other optimizers, but can be effective for training deep learning models.

There are many other optimizers available in PyTorch, such as Adagrad, Adadelta, and others
"""
optim = torch.optim.SGD(model.parameters(), lr=flags.lr, weight_decay=flags.weight_decay, momentum=0.9)


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


for epoch in range(flags.epochs):
    print(f'Epoch {epoch}: Best val loss {best_val_loss}, Best val acc {best_val_acc}, best val recall {best_val_recall}, best val precision {best_val_precision}')
    lossMeter = AverageMeter()
    regMeter = AverageMeter()
    correctMeter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    model.train()
    for data in tqdm(train_loader, ncols=75, leave=False):
        data = to_device(data)
        def closure():
            optim.zero_grad()
            x, y, transformed_image = data[0] , data[1], data[2]
            inputs, labels = x, y
            inputs_transformed, labels_transformed = transformed_image, y
            logits, base_output = model(inputs)


            loss = F.cross_entropy(logits, labels)

            if flags.use_reg_loss:
                logits_transformed, base_output_transformed = model(inputs_transformed)
                reg = flags.alpha * nn.MSELoss(base_output_transformed, base_output)
                (loss + reg).backward()
                regMeter.update(reg.detach().item(), data[0].shape[0])
            else:
                loss.backward()


            lossMeter.update(loss.detach().item(), data[0].shape[0])
            correctMeter.update(correct.detach().item(), data[0].shape[0])
            precision_meter.update(precision, data[0].shape[0])
            recall_meter.update(recall, data[0].shape[0])


        optim.step(closure)

    print(f'>>> Training: Loss {lossMeter}, Reg {regMeter}, Acc {correctMeter}, precision: {precision_meter}, recall{recall_meter}')

    vallossMeter = AverageMeter()
    valregMeter = AverageMeter()
    valcorrectMeter = AverageMeter()
    val_precision_meter = AverageMeter()
    val_recall_meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        if flags.dataset == "FitzPatrick17k":
            for x, y, fst in tqdm(val_loader, ncols=75, leave=False):
                x, y, fst = x.to(device), y.to(device), fst.to(device)
                loss, reg, correct = model(x, y)

                vallossMeter.update(loss.detach().item(), x.shape[0])
                valregMeter.update(reg.detach().item(), x.shape[0])
                valcorrectMeter.update(correct.detach().item(), x.shape[0])
                del loss, reg, correct
        elif flags.dataset == "isic2018":
            for x, y, transformed_image in tqdm(val_loader, ncols=75, leave=False):
                x, y, transformed_image = x.to(device), y.to(device), transformed_image.to(device)

                logits, base_output = model(x)
                loss = nn.CrossEntropyLoss(logits, y)
                y_true.append(y.cpu().numpy())
                predictions = torch.argmax(logits, 1).cpu().numpy()
                labels = y.cpu().numpy()
                y_pred.append(predictions)


                cm = confusion_matrix(labels, predictions)
                precision = cm.diagonal().sum() / cm.sum(axis=0).sum()
                recall = cm.diagonal().sum() / cm.sum(axis=1).sum()

                vallossMeter.update(loss.detach().item(), x.shape[0])
                valregMeter.update(reg.detach().item(), x.shape[0])
                valcorrectMeter.update(correct.detach().item(), x.shape[0])
                val_precision_meter.update(precision, x.shape[0])
                val_recall_meter.update(recall, x.shape[0])

                del loss, reg, correct, precision, recall
    print(f'>>> Val: Loss {vallossMeter}, Reg {valregMeter}, Acc {valcorrectMeter}, precision: {val_precision_meter}, recall{val_recall_meter}')
    # Compute the confusion matrix
    #cm = confusion_matrix(y_true, y_pred)

    # Display the confusion matrix
    #print(cm)

    if vallossMeter.float() < best_val_loss:
        best_val_loss = vallossMeter.float()

    if val_recall_meter.float() > best_val_recall:
        best_val_recall = val_recall_meter.float()
    if val_precision_meter.float() > best_val_precision:
        best_val_precision = val_precision_meter.float()
    if valcorrectMeter.float() > best_val_acc:
        best_val_acc = valcorrectMeter.float()
        save_path = os.path.join(flags.model_save_dir, 'epoch{}_acc_{:.3f}.ckpt'.format(epoch, best_val_acc))
        torch.save(model.state_dict(), save_path)
        print('Saved model with highest acc ...')

    torch.cuda.empty_cache()
