class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0

    def update(self, val, n=1):
        self.count += n
        self.sum += val * n

    def float(self):
        if self.count > 0:
            return self.sum / self.count
        else:
            return 0


    def __repr__(self):
        if self.count > 0:
            return '%.4f' % (self.sum / self.count)
        else:
            return "0.000"
