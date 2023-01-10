from util import AverageMeter
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


class Metrics():
    def __init__(self):
        self.loss_meter = AverageMeter()
        self.accuracy_meter = AverageMeter()
        self.precision_meter = AverageMeter()
        self.recall_meter = AverageMeter()
        self.regularization_meter = AverageMeter()

    def compute_metrics(self, logits, labels):
        # Compute metrics for main input image
        self.loss = F.cross_entropy(logits, labels)
        self.predictions = torch.argmax(logits, 1)
        # accuracy = (torch.argmax(logits, 1) == labels).sum().float() / inputs.shape[0]
        self.accuracy = torch.mean(torch.eq(self.predictions, labels).float())

        _predictions = self.predictions.cpu().numpy()
        _labels = labels.cpu().numpy()

        self.confusion_matrix = confusion_matrix(_labels, _predictions)
        _diagonal = self.confusion_matrix.diagonal().sum()

        self.precision = _diagonal / self.confusion_matrix.sum(axis=0).sum()
        self.recall = _diagonal / self.confusion_matrix.sum(axis=1).sum()
        return


    def update_metrics(self, data, reg=None):
        # data  = data[0].shape[0]
        self.loss_meter.update(self.loss.detach().item(), data)
        self.accuracy_meter.update(self.accuracy.detach().item(), data)
        self.precision_meter.update(self.precision, data)
        self.recall_meter.update(self.recall, data)
        if reg:
            self.regularization_meter.update(reg.detach().item(), data)
        return