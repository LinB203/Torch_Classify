import torch

class ConfusionMatrix():
    def __init__(self, num_classes, device):
        self.device = device
        self.num_classes = num_classes
        self.confusionmat = torch.zeros((num_classes, num_classes)).to(device)
        self.per_recall = torch.zeros((1, num_classes))
        self.per_precision = torch.zeros((1, num_classes))
        self.per_F1 = torch.zeros((1, num_classes))

        self.mean_val_accuracy = 0
        self.mean_precision = 0
        self.mean_recall = 0
        self.mean_F1 = 0

    def update(self, val_labels, predict_y):
        mask = (predict_y >= 0) & (predict_y < self.num_classes)
        label = self.num_classes * val_labels[mask].int() + predict_y[mask]
        bincount = torch.bincount(label, minlength=self.num_classes ** 2)
        self.confusionmat += bincount.reshape(self.num_classes, self.num_classes)
        # print(self.confusionmat)

    def acc_p_r_f1(self):
        tp = torch.diag(self.confusionmat)
        tp_fp = torch.sum(self.confusionmat, dim=0)
        tp_fn = torch.sum(self.confusionmat, dim=1)
        self.per_recall = tp / tp_fn
        self.per_precision = tp / tp_fp
        self.per_F1 = 2 * self.per_recall * self.per_precision / (self.per_recall + self.per_precision)
        self.mean_val_accuracy = torch.sum(tp) / torch.sum(self.confusionmat)

        # 精度调整
        self.per_recall = torch.nan_to_num(self.per_recall * 100)
        self.per_precision = torch.nan_to_num(self.per_precision * 100)
        self.per_F1 = torch.nan_to_num(self.per_F1 * 100)
        self.mean_val_accuracy = torch.nan_to_num(self.mean_val_accuracy * 100)

        self.mean_precision = torch.mean(self.per_precision)
        self.mean_recall = torch.mean(self.per_recall)
        self.mean_F1 = torch.mean(self.per_F1)
        # self.mean_val_accuracy = self.mean_val_accuracy

        # return val_accuracy.mean(), precision.mean(), recall.mean(), F1.mean()

    def save(self, results_file, epoch):
        self.per_precision = ['%7.3f'%i for i in self.per_precision]
        self.per_recall = ['%7.3f'%i for i in self.per_recall]
        self.per_F1 = ['%7.3f'%i for i in self.per_F1]
        with open(results_file, 'a') as f:
            f.write('\n%5d'%epoch+' '+'%8.3f'%self.mean_val_accuracy+' '+'%9.3f'%self.mean_precision+' '+'%6.3f'%self.mean_recall+' '\
                    +'%8.3f'%self.mean_F1+' '+' '.join(self.per_precision)+' '+' '.join(self.per_recall)+' '+' '.join(self.per_F1))
        self.__init__(self.num_classes, self.device)
