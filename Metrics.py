from util import AverageMeter
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


class Metrics:
    def __init__(self):
        self.loss_meter = AverageMeter()
        self.accuracy_meter = AverageMeter()
        self.precision_meter = AverageMeter()
        self.recall_meter = AverageMeter()
        self.regularization_meter = AverageMeter()
        self.reg = None

    def compute_metrics(self, logits, labels, loss):
        # Compute metrics for main input image
        #self.loss = F.cross_entropy(logits, labels)
        self.loss = loss
        self.predictions = torch.argmax(logits, 1)
        # accuracy = (torch.argmax(logits, 1) == labels).sum().float() / inputs.shape[0]
        #self.accuracy = torch.mean(torch.eq(self.predictions, labels).float())
        #self.accuracy = logits.eq(labels.view_as(logits)).sum().item()/logits[0]
        self.accuracy = (logits.argmax(dim=1) == labels).float().mean()

        _predictions = self.predictions.cpu().numpy()
        _labels = labels.cpu().numpy()

        self.confusion_matrix = confusion_matrix(_labels, _predictions)
        _diagonal = self.confusion_matrix.diagonal().sum()

        self.precision = _diagonal / self.confusion_matrix.sum(axis=0).sum()
        self.recall = _diagonal / self.confusion_matrix.sum(axis=1).sum()
        return

    def update_metrics(self, size):
        # data  = data[0].shape[0]
        self.loss_meter.update(self.loss.detach().item(), size)
        self.accuracy_meter.update(self.accuracy.detach().item(), size)
        self.precision_meter.update(self.precision, size)
        self.recall_meter.update(self.recall, size)
        if self.reg is not None:
            self.regularization_meter.update(self.reg.detach().item(), size)
        return
