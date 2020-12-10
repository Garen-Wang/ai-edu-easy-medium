import numpy as np
import matplotlib.pyplot as plt


class TrainingHistory(object):
    def __init__(self):
        self.loss_train = []
        self.accuracy_train = []
        self.loss_vld = []
        self.accuracy_vld = []
        self.iteration_seq = []
        self.epoch_seq = []

    def Add(self, loss_train, accuracy_train, loss_vld, accuracy_vld, total_iteration, epoch):
        self.loss_train.append(loss_train)
        self.accuracy_train.append(accuracy_train)
        if loss_vld is not None:
            self.loss_vld.append(loss_vld)
        if accuracy_vld is not None:
            self.accuracy_vld.append(accuracy_vld)
        self.iteration_seq.append(total_iteration)
        self.epoch_seq.append(epoch)

    def ShowLossHistory(self, params):
        plt.figure(figsize=(12, 5))
        axes = plt.subplot(1, 2, 1)
        p1, = axes.plot(self.epoch_seq, self.loss_train)
        p2, = axes.plot(self.epoch_seq, self.loss_vld)
        axes.legend([p1, p2], ['train', 'validation'])
        axes.set_title('Loss')
        axes.set_xlabel('epoch')
        axes.set_ylabel('loss')

        axes = plt.subplot(1, 2, 2)
        p1, = axes.plot(self.epoch_seq, self.accuracy_train)
        p2, = axes.plot(self.epoch_seq, self.accuracy_vld)
        axes.legend([p1, p2], ['train', 'validation'])
        axes.set_title('Accuracy')
        axes.set_xlabel('epoch')
        axes.set_ylabel('accuracy')

        title = params.toString()
        plt.suptitle(title)
        plt.show()

