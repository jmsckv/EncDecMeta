import torch
import math
from collections import OrderedDict

class MetricsAggregator:
    """
    Similar to TF, we instantiate (and reset) a MetricsAggregator object at the beginning of an epoch.
    With every batch, we update three components of its internal state.
    1. Confusion matrix per class, summed up for all batches observed so far.
    2. Actually observed pixel count per class, summed up for all batches observed so far.
    3. Number of samples evaluated so far.
    """
    def __init__(self, fold, config):
        n_cl = config['classes']
        self.debug = config['debug']
        self.fold = fold
        self.res = int(config['H'] * config['W'])
        self.tensors = dict()

        # confusion matrix and pixel count per class
        for i in ['TP','TN','FP','FN','PX_CNT']:
            self.tensors[i] = torch.cuda.DoubleTensor([0]*n_cl) if torch.cuda.is_available() else torch.DoubleTensor([0]*n_cl)

        # background classes can get 'M'asked
        self.tensors['M'] = torch.cuda.BoolTensor(n_cl * [True]) if torch.cuda.is_available() else torch.BoolTensor(n_cl * [True])
        if config.get('bg_classes', False):
            for i in config['bg_classes']:
                self.tensors['M'][i] = False

        # this scalar gets continuously updated, treat it as a tensor
        self.tensors['samples_evaluated'] = torch.cuda.DoubleTensor([0]) if torch.cuda.is_available() else torch.DoubleTensor([0])

        if self.debug:
            assert fold in ['train', 'val', 'test']
            if torch.cuda.is_available():
                for t in self.tensors: assert self.tensors[t].is_cuda

    def reset_state(self):
        for t in ['TP','TN','FP','FN','PX_CNT','samples_evaluated']: # don't reset mask!
            if isinstance(self.tensors[t], (torch.DoubleTensor, torch.cuda.DoubleTensor)):
                self.tensors[t][:] = 0
            elif isinstance(self.tensors[t], (torch.BoolTensor, torch.cuda.BoolTensor)):
                self.tensors[t][:] = True
            else:
                raise NotImplementedError


    def update_state(self, logits, labels):
        """
        After each predicted batch, update state of internal confusion matrix from which metrics can be dynamically derived.
        :param logits: nn outputs with shape N,C,H,W
        :param labels: gt labels with shape N,H,W; for every value v in this tensor must hold: 0<=v<C
        """
        labels = labels.unsqueeze(dim=1) #.cuda()  # N,1,H,W
        predictions = logits.max(axis=1)[1].unsqueeze(dim=1) #.cuda()  # N,1,H,W
        preds_1hot = torch.zeros_like(logits, dtype=bool).scatter_(1, predictions, True)#.cuda()
        labels_1hot = torch.zeros_like(logits, dtype=bool).scatter_(1, labels, True)#.cuda()

        if self.debug:
            if torch.cuda.is_available():
                for t in ['labels','preditions','preds_1hot','labels_1hot']:
                    assert t.is_cuda
            assert logits.shape[0] ==  labels.shape[0]
            assert logits.shape == preds_1hot.shape
            assert logits.shape[1] >= labels.max()
            assert preds_1hot.shape == labels_1hot.shape
            assert preds_1hot.max() == labels_1hot.max() == 1
            assert preds_1hot.min() == labels_1hot.min() == 0
            assert preds_1hot.sum() == labels_1hot.sum() == logits.shape[0]*logits.shape[2]* logits.shape[3]

        self.tensors['TP'] += torch.sum(preds_1hot & labels_1hot , dim=(0, 2, 3)).double() # casting to double required for Cuda 10.0, not required for Cuda 10.1
        self.tensors['TN'] += torch.sum(~ preds_1hot & ~ labels_1hot , dim=(0, 2, 3)).double()
        self.tensors['FP'] += torch.sum(preds_1hot & ~ labels_1hot, dim=(0, 2, 3)).double()
        self.tensors['FN'] += torch.sum(~ preds_1hot & labels_1hot, dim=(0, 2, 3)).double()
        self.tensors['PX_CNT'] += torch.sum(labels_1hot , dim=(0, 2, 3)).double()
        self.tensors['samples_evaluated'] += logits.shape[0]
        
        if self.debug:
            if torch.cuda.is_available():
                for t in self.tensors: assert self.tensors[t].is_cuda


    def get_metrics(self, add_metrics={}):
        """
        Usually called at epoch end. In principle, calling during epoch should also be possible.
        In this case, metrics would get calculated based on classes observed so far.
        """
        tot_px_cnt = self.res * int(self.tensors['samples_evaluated'][0])

        if self.debug:
            sum_per_class = self.tensors['TP'] + self.tensors['TN'] + self.tensors['FP'] + self.tensors['FN']
            unique = sum_per_class.unique()
            assert len(unique) == 1, 'Expect to observe the exact same number for all classes.'
            assert unique[0] == self.tensors['PX_CNT'].sum() == tot_px_cnt, 'Expect exactly one type of prediction per pixel.'

        mask_non_observed = (self.tensors['PX_CNT']).bool()
        mask_bg = self.tensors['M']
        mask_combined = (self.tensors['M'] * mask_non_observed).bool() # in PyTorch 1.4 no logical AND

        if self.debug:
            assert mask_combined.sum() <= mask_bg.sum()
            assert mask_combined.sum() <= mask_non_observed.sum()
                                                            
        accuracies = (self.tensors['TP'] + self.tensors['TN']) / tot_px_cnt
        acc = torch.mean(accuracies[mask_combined])
        acc_bg_included = torch.mean(accuracies[mask_non_observed])

        IoUs = self.tensors['TP'] / (tot_px_cnt - self.tensors['TN']) # per class: I/U, U = sum(TP,FP,FN) = all - TN
        mIoU = torch.mean(IoUs[mask_combined])
        mIoU_bg_included = torch.mean(IoUs[mask_non_observed])

        if self.debug:
            if torch.cuda.is_available():
                for i in [accuracies, acc, acc_bg_included, IoUs, mIoU, mIoU_bg_included]:
                    assert i.is_cuda

        results = OrderedDict()

        for i in ['acc','mIoU']:
            for j in ['','_bg_included']:
                results[ i + j + '_' + self.fold ] = float(eval(i+j+'.cpu()'))

        for i in range(self.tensors['TP'].shape[0]):
            results['IoU_class_' + str(i) + '_' + self.fold] = float(IoUs[i].cpu())
            results['acc_class_' + str(i) + '_' + self.fold] = float(accuracies[i].cpu())

        if self.debug:
            for k in results:
                if isinstance(results[k], float) and not math.isnan(results[k]):
                    # don't apply check to nans and str; we don't use exactly 1 due to smaller rounding error
                    assert results[k] <= 1.0001, f'Failure for {k,results[k],type(results[k])}: any metric derived from the confusion matrix should be <= 1.'

        #for t in self.tensors:
        #    results[t + '_' + self.fold] = self.tensors[t].cpu()

        if add_metrics:
            for k in add_metrics:
                results[k + '_' + self.fold] = float(add_metrics[k])

        return results



